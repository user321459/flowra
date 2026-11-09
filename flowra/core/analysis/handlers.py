"""
Architecture-specific handlers.

Each handler registers calibration hooks, evaluates the flow-change descriptor
for every adaptation target, partitions weight matrices and extracts blocks.

* :class:`FeedForwardHandler` - chains of fully connected / convolutional layers;
* :class:`TransformerHandler` - transformer layers with separate attention and
  FFN sub-layer states;
* :class:`ResidualCNNHandler` - residual blocks, using the residual branch ``F_l``;
* :class:`GatedRNNHandler` - GRU / LSTM gate divergences over time steps.

Head-level descriptors and the value/output rank split live at the end of the module.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from flowra.core.analysis.flow_descriptor import (
    DEFAULT_EPS,
    TERMS_ACCELERATION,
    TERMS_CONV,
    TERMS_FULL,
    FixedProjector,
    align_states,
    flow_change,
    flow_change_from_sequence,
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def first_tensor(output: Any) -> Optional[torch.Tensor]:
    """Return the primary tensor of a module output (tensor, tuple or dict-like)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            t = first_tensor(item)
            if t is not None:
                return t
        return None
    if isinstance(output, dict) or hasattr(output, "values"):
        for item in output.values():
            t = first_tensor(item)
            if t is not None:
                return t
    return None


def replace_first_tensor(output: Any, new: torch.Tensor) -> Any:
    """Return ``output`` with its primary tensor replaced by ``new``."""
    if isinstance(output, torch.Tensor):
        return new
    if isinstance(output, tuple):
        items = list(output)
        for i, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                items[i] = new
                return type(output)(items) if type(output) is not tuple else tuple(items)
        return output
    if isinstance(output, list):
        items = list(output)
        for i, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                items[i] = new
                return items
    return output


def name_matches(name: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    return any(re.search(p, name) for p in patterns)


def block_bounds(size: int, parts: int, unit: int = 1) -> List[Tuple[int, int]]:
    """
    Split ``range(size)`` into ``parts`` contiguous, nearly equal intervals
    whose boundaries are multiples of ``unit``.
    """
    units = size // unit
    parts = max(1, min(int(parts), units))
    edges = [round(i * units / parts) * unit for i in range(parts + 1)]
    return [(edges[i], edges[i + 1]) for i in range(parts)]


def default_partition(m: int, n: int) -> Tuple[int, int]:
    """Default partition: ``p = ceil(sqrt(m))``, ``q = ceil(sqrt(n))``."""
    return math.ceil(math.sqrt(m)), math.ceil(math.sqrt(n))


BlockLayout = Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]


def extract_block(weight_2d: torch.Tensor, layout: BlockLayout, i: int, j: int) -> torch.Tensor:
    """View of block ``(i, j)`` of a partitioned 2-D weight."""
    (r0, r1), (c0, c1) = layout[0][i], layout[1][j]
    return weight_2d[r0:r1, c0:c1]


@dataclass
class TargetInfo:
    """An adaptation target: a module (or an RNN gate) that can receive rank."""

    name: str
    kind: str  # "linear" | "conv" | "gate"
    module: nn.Module
    shape: Tuple[int, int]  # (m, n) of the flattened weight W_l
    group: str = ""  # flow point / block the target belongs to
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_rank(self) -> int:
        return min(self.shape)

    @property
    def cost_per_rank(self) -> int:
        return self.shape[0] + self.shape[1]


@dataclass
class DescriptorSettings:
    eps: float = DEFAULT_EPS
    alignment: str = "zero_pad"
    projector: FixedProjector = field(default_factory=FixedProjector)


class CaptureStore:
    """
    Per-batch storage for calibration hooks.

    ``states`` hold the representations entering the flow-change descriptor
    (batch-first).  ``grad_sq`` accumulates per-sample squared norms of the
    task-loss gradient with respect to target outputs.  The loss is assumed to
    be averaged over the batch, so the per-sample gradient is rescaled by the
    batch size.
    """

    def __init__(self, batch_dim: int = 0, keep_graph: bool = False, fp16: bool = False):
        self.batch_dim = batch_dim
        self.keep_graph = keep_graph
        self.fp16 = fp16
        self.states: Dict[str, torch.Tensor] = {}
        self.order: List[str] = []
        self.grad_sq: Dict[str, torch.Tensor] = {}
        self.grad_tensors: Dict[str, List[torch.Tensor]] = {}
        self.context: Dict[str, Any] = {}

    def batch_first(self, t: torch.Tensor) -> torch.Tensor:
        return t.movedim(self.batch_dim, 0) if self.batch_dim else t

    def save_state(self, key: str, tensor: torch.Tensor, batch_first: bool = False) -> None:
        t = tensor if batch_first else self.batch_first(tensor)
        if not self.keep_graph:
            t = t.detach()
            if self.fp16 and t.is_floating_point():
                t = t.half()
        self.states[key] = t
        if key not in self.order:
            self.order.append(key)

    def _grad_hook(self, key: str, batch_first: bool) -> Callable[[torch.Tensor], None]:
        bd = 0 if batch_first else self.batch_dim

        def hook(grad: torch.Tensor) -> None:
            g = grad.movedim(bd, 0) if bd else grad
            n = g.shape[0]
            sq = g.detach().float().reshape(n, -1).pow(2).sum(dim=1) * float(n * n)
            prev = self.grad_sq.get(key)
            self.grad_sq[key] = sq if prev is None else prev + sq

        return hook

    def add_grad_hook(self, key: str, tensor: torch.Tensor, batch_first: bool = False) -> None:
        """Track the gradient of an intermediate tensor that already requires grad."""
        if tensor.requires_grad:
            tensor.register_hook(self._grad_hook(key, batch_first))
            self.grad_tensors.setdefault(key, []).append(tensor)

    def track_output(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Make ``tensor`` part of the autograd graph (if needed), register a
        gradient hook on it and return a clone to be used downstream.  The
        clone keeps in-place operations of the following layers away from
        the hooked tensor.
        """
        if not torch.is_grad_enabled():
            return tensor
        src = tensor if tensor.requires_grad else tensor.detach().requires_grad_(True)
        src.register_hook(self._grad_hook(key, batch_first=False))
        self.grad_tensors.setdefault(key, []).append(src)
        return src.clone()

    def per_sample_grad_norm(self, key: str) -> Optional[torch.Tensor]:
        sq = self.grad_sq.get(key)
        return None if sq is None else sq.sqrt()


def _register_target_hooks(
    targets: Dict[str, TargetInfo],
    store: CaptureStore,
    save_states: bool = False,
) -> List[Any]:
    handles = []
    for name, info in targets.items():
        if info.kind == "gate":
            continue

        def hook(module, args, output, name=name):
            out = first_tensor(output)
            if out is None:
                return None
            if save_states:
                store.save_state(name, out)
            return replace_first_tensor(output, store.track_output(name, out))

        handles.append(info.module.register_forward_hook(hook))
    return handles


def _weight_2d(module: nn.Module) -> torch.Tensor:
    w = module.weight
    return w.reshape(w.shape[0], -1)


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------


class ArchitectureHandler:
    """Base class.  Subclasses override discovery, hooks and divergence."""

    name = "base"

    def __init__(self):
        self.batch_dim = 0

    # -- discovery ---------------------------------------------------------
    def matches(self, model: nn.Module) -> bool:
        return False

    def discover_targets(
        self,
        model: nn.Module,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> Dict[str, TargetInfo]:
        raise NotImplementedError

    @staticmethod
    def _filter(name: str, include, exclude) -> bool:
        if include and not name_matches(name, include):
            return False
        if exclude and name_matches(name, exclude):
            return False
        return True

    # -- calibration -------------------------------------------------------
    def attach(self, model: nn.Module, targets: Dict[str, TargetInfo], store: CaptureStore) -> List[Any]:
        raise NotImplementedError

    def detach(self, handles: List[Any]) -> None:
        for h in handles:
            h.remove()

    def compute_divergence(
        self, store: CaptureStore, targets: Dict[str, TargetInfo], settings: DescriptorSettings
    ) -> Dict[str, torch.Tensor]:
        """Signed per-sample descriptor ``[N]`` per target."""
        raise NotImplementedError

    def point_descriptors(self, store: CaptureStore, settings: DescriptorSettings) -> Dict[str, torch.Tensor]:
        """Optional per-flow-point descriptors (for reports)."""
        return {}

    # -- weights / partition -----------------------------------------------
    def weight_matrix(self, target: TargetInfo) -> torch.Tensor:
        return _weight_2d(target.module)

    def partition(self, target: TargetInfo) -> Tuple[int, int]:
        """Number of row / column blocks ``(p_l, q_l)``."""
        return default_partition(*target.shape)

    def column_unit(self, target: TargetInfo) -> int:
        """Granularity of column boundaries (whole kernels for convolutions)."""
        if target.kind == "conv":
            return target.module.weight[0, 0].numel()
        return 1

    def block_layout(self, target: TargetInfo) -> BlockLayout:
        p, q = self.partition(target)
        m, n = target.shape
        return block_bounds(m, p), block_bounds(n, q, self.column_unit(target))


# ---------------------------------------------------------------------------
# Fully connected and convolutional chains
# ---------------------------------------------------------------------------


class FeedForwardHandler(ArchitectureHandler):
    """
    Chains of linear / convolutional layers.

    The trajectory is ``[x_0, y_0, y_1, ..., y_{K-1}]``: the input of the
    first target followed by the outputs of all targets in execution order.
    Target ``k`` is centred on its input, so the stencil is
    ``(input of the previous layer, input, output)``. Linear layers use
    acceleration, velocity and alignment; convolutions drop alignment.
    """

    name = "feedforward"
    INPUT_KEY = "__input__"

    def matches(self, model: nn.Module) -> bool:
        return True

    def discover_targets(self, model, include=None, exclude=None):
        targets: Dict[str, TargetInfo] = {}
        for name, mod in model.named_modules():
            if not name or not self._filter(name, include, exclude):
                continue
            if isinstance(mod, nn.Linear):
                targets[name] = TargetInfo(name, "linear", mod, (mod.out_features, mod.in_features), group=name)
            elif isinstance(mod, (nn.Conv1d, nn.Conv2d)) and mod.groups == 1:
                n = mod.weight[0].numel()
                targets[name] = TargetInfo(name, "conv", mod, (mod.out_channels, n), group=name)
        return targets

    def attach(self, model, targets, store):
        handles = []

        def pre_hook(module, args):
            if self.INPUT_KEY not in store.states and args:
                x = first_tensor(args)
                if x is not None:
                    store.save_state(self.INPUT_KEY, x)

        for info in targets.values():
            handles.append(info.module.register_forward_pre_hook(pre_hook))
        handles.extend(_register_target_hooks(targets, store, save_states=True))
        return handles

    def _sequence(self, store, targets):
        keys = [self.INPUT_KEY] + [k for k in store.order if k in targets]
        return keys, [store.states[k] for k in keys]

    def compute_divergence(self, store, targets, settings):
        keys, states = self._sequence(store, targets)
        result = {}
        for idx, key in enumerate(keys[1:]):
            terms = TERMS_CONV if targets[key].kind == "conv" else TERMS_FULL
            result[key] = flow_change_from_sequence(
                states, center=idx, terms=terms, eps=settings.eps,
                alignment=settings.alignment, projector=settings.projector,
            )["D"]
        return result

    def point_descriptors(self, store, settings):
        keys = [k for k in store.order]
        states = [store.states[k] for k in keys]
        return {
            k: flow_change_from_sequence(states, i, TERMS_FULL, settings.eps, settings.alignment, settings.projector)["D"]
            for i, k in enumerate(keys)
        }


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------


ATTN_TOKENS = ("self_attn", "attention", "attn", "self_attention", "mha")
FINAL_NORM_TOKENS = ("norm", "ln_f", "final_layer_norm", "layernorm", "layer_norm", "final_norm")
QKV_PATTERN = re.compile(r"(^|[._])(q|k|v|q_proj|k_proj|v_proj|query|key|value|to_q|to_k|to_v|wq|wk|wv)$")
OUT_PATTERN = re.compile(r"(^|[._])(o_proj|out_proj|o|wo|to_out|c_proj|dense)$")


def _num_heads(module: nn.Module) -> Optional[int]:
    for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads", "nhead"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    cfg = getattr(module, "config", None)
    if cfg is not None:
        value = getattr(cfg, "num_attention_heads", None)
        if isinstance(value, int):
            return value
    for child in module.children():
        value = _num_heads(child)
        if value:
            return value
    return None


class TransformerHandler(ArchitectureHandler):
    """
    Transformer layers.

    ``h_l`` is the output of the entire layer (attention + FFN).  When the
    attention sub-module of a layer can be identified, the intermediate
    residual-stream state after attention is inserted into the trajectory,
    which yields the separate divergences ``D_attn`` and ``D_ffn``:

        [x_0, mid_0, h_0, mid_1, h_1, ..., h_{L-1}, final_norm(h_{L-1})]

    For attention modules returning a residual update (LLaMA, GPT, ViT) the
    intermediate state is ``x_l + attn_l``; for modules that already include
    the residual connection and normalisation (BERT/RoBERTa) it is the
    attention output itself.  Attention projections are centred on ``mid_l``,
    FFN projections on ``h_l``.

    ``nn.MultiheadAttention.out_proj`` is not a target: PyTorch reads its
    weight directly instead of calling the module.
    """

    name = "transformer"

    def __init__(self, split_attention_ffn: bool = True):
        super().__init__()
        self.split_attention_ffn = split_attention_ffn
        self.blocks_name: Optional[str] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.final_norm: Optional[nn.Module] = None
        self._attn: Dict[int, Tuple[str, nn.Module, bool]] = {}

    # -- structure ---------------------------------------------------------
    @staticmethod
    def find_blocks(model: nn.Module) -> Optional[Tuple[str, nn.ModuleList]]:
        best = None
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.ModuleList) or len(mod) < 2:
                continue
            if all(sum(isinstance(m, nn.Linear) for m in child.modules()) >= 2 for child in mod):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        return best

    def matches(self, model):
        return self.find_blocks(model) is not None

    @staticmethod
    def _attention_child(block: nn.Module) -> Optional[Tuple[str, nn.Module, bool]]:
        for child_name, child in block.named_children():
            if any(tok in child_name.lower() for tok in ATTN_TOKENS):
                includes_residual = any(
                    isinstance(m, nn.LayerNorm) or "norm" in type(m).__name__.lower()
                    for m in child.modules()
                    if m is not child
                )
                return child_name, child, includes_residual
        return None

    def _find_final_norm(self, model, blocks_name):
        candidate = None
        for name, mod in model.named_modules():
            if not name or name.startswith(blocks_name + ".") or name == blocks_name:
                continue
            leaf = name.split(".")[-1].lower()
            if leaf in FINAL_NORM_TOKENS and "norm" in type(mod).__name__.lower():
                candidate = mod
        return candidate

    def _setup(self, model):
        found = self.find_blocks(model)
        if found is None:
            raise ValueError("No transformer layer stack found")
        self.blocks_name, self.blocks = found
        self.final_norm = self._find_final_norm(model, self.blocks_name)
        self._attn = {}
        for l, block in enumerate(self.blocks):
            attn = self._attention_child(block)
            if attn is not None:
                self._attn[l] = attn
        first = self.blocks[0]
        attn = self._attn.get(0)
        if attn is not None and isinstance(attn[1], nn.MultiheadAttention) and not attn[1].batch_first:
            self.batch_dim = 1
        elif isinstance(first, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            self.batch_dim = 0 if first.self_attn.batch_first else 1

    def discover_targets(self, model, include=None, exclude=None):
        self._setup(model)
        targets: Dict[str, TargetInfo] = {}
        for l, block in enumerate(self.blocks):
            attn = self._attn.get(l)
            for sub_name, mod in block.named_modules():
                if not isinstance(mod, nn.Linear) or not sub_name:
                    continue
                parent_name = sub_name.rsplit(".", 1)[0] if "." in sub_name else ""
                parent = block.get_submodule(parent_name) if parent_name else block
                if isinstance(parent, nn.MultiheadAttention):
                    continue
                full = f"{self.blocks_name}.{l}.{sub_name}"
                if not self._filter(full, include, exclude):
                    continue
                in_attn = attn is not None and (sub_name == attn[0] or sub_name.startswith(attn[0] + "."))
                sub = "attn" if in_attn else "ffn"
                targets[full] = TargetInfo(
                    full, "linear", mod, (mod.out_features, mod.in_features),
                    group=f"layer{l}.{sub}",
                    meta={"layer": l, "sublayer": sub, "local_name": sub_name},
                )
        return targets

    # -- hooks -------------------------------------------------------------
    def attach(self, model, targets, store):
        if self.blocks is None:
            self._setup(model)
        store.batch_dim = self.batch_dim
        handles = []
        for l, block in enumerate(self.blocks):

            def pre_hook(module, args, kwargs, l=l):
                x = first_tensor(args) if args else None
                if x is None:
                    for key in ("hidden_states", "x", "src", "tgt", "input"):
                        if isinstance(kwargs.get(key), torch.Tensor):
                            x = kwargs[key]
                            break
                if x is not None:
                    store.save_state(f"in{l}", x)

            def post_hook(module, args, output, l=l):
                out = first_tensor(output)
                if out is not None:
                    store.save_state(f"out{l}", out)

            handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(block.register_forward_hook(post_hook))

            attn = self._attn.get(l) if self.split_attention_ffn else None
            if attn is not None:
                _, attn_mod, includes_residual = attn

                def attn_hook(module, args, output, l=l, includes_residual=includes_residual):
                    delta = first_tensor(output)
                    if delta is None:
                        return
                    delta = store.batch_first(delta)
                    x = store.states.get(f"in{l}")
                    if not includes_residual and x is not None and x.shape == delta.shape:
                        mid = x.to(delta.dtype) + delta
                    else:
                        mid = delta
                    store.save_state(f"mid{l}", mid, batch_first=True)

                handles.append(attn_mod.register_forward_hook(attn_hook))

        if self.final_norm is not None:

            def final_hook(module, args, output):
                out = first_tensor(output)
                if out is not None:
                    store.save_state("final", out)

            handles.append(self.final_norm.register_forward_hook(final_hook))

        handles.extend(_register_target_hooks(targets, store, save_states=False))
        return handles

    def _trajectory(self, store) -> Tuple[List[str], List[torch.Tensor]]:
        keys = ["in0"] if "in0" in store.states else []
        for l in range(len(self.blocks)):
            if f"mid{l}" in store.states:
                keys.append(f"mid{l}")
            if f"out{l}" in store.states:
                keys.append(f"out{l}")
        if "final" in store.states:
            keys.append("final")
        return keys, [store.states[k] for k in keys]

    def compute_divergence(self, store, targets, settings):
        keys, states = self._trajectory(store)
        index = {k: i for i, k in enumerate(keys)}
        cache: Dict[str, torch.Tensor] = {}
        result = {}
        for name, info in targets.items():
            l, sub = info.meta["layer"], info.meta["sublayer"]
            key = f"mid{l}" if sub == "attn" and f"mid{l}" in index else f"out{l}"
            if key not in cache:
                cache[key] = flow_change_from_sequence(
                    states, index[key], TERMS_FULL, settings.eps, settings.alignment, settings.projector
                )["D"]
            result[name] = cache[key]
        return result

    def point_descriptors(self, store, settings):
        keys, states = self._trajectory(store)
        return {
            k: flow_change_from_sequence(states, i, TERMS_FULL, settings.eps, settings.alignment, settings.projector)["D"]
            for i, k in enumerate(keys)
        }

    def compute_head_descriptors(self, store, settings) -> Dict[int, torch.Tensor]:
        """Per-layer head divergences, shape ``[N, H]``."""
        out: Dict[int, torch.Tensor] = {}
        for l, attn in self._attn.items():
            heads = _num_heads(attn[1])
            h = store.states.get(f"mid{l}")
            h_prev = store.states.get(f"in{l}")
            if not heads or h is None or h_prev is None:
                continue
            aligned = align_states([h_prev, h], settings.alignment, settings.projector)
            try:
                out[l] = head_divergence(aligned[1], aligned[0], heads, settings.eps)
            except ValueError:
                continue
        return out

    # -- partition ---------------------------------------------------------
    def partition(self, target):
        """
        Attention projections are partitioned by heads (rows of Q/K/V, columns
        of the output projection); all other layers use the default partition.
        """
        if target.meta.get("sublayer") == "attn" and self.blocks is not None:
            attn = self._attn.get(target.meta["layer"])
            heads = _num_heads(attn[1]) if attn else None
            local = target.meta.get("local_name", "")
            m, n = target.shape
            if heads:
                if QKV_PATTERN.search(local) and n % heads == 0 and m % (n // heads) == 0:
                    return m // (n // heads), 1
                if OUT_PATTERN.search(local) and m % heads == 0 and n % (m // heads) == 0:
                    return 1, n // (m // heads)
        return default_partition(*target.shape)


# ---------------------------------------------------------------------------
# Residual CNNs
# ---------------------------------------------------------------------------


class ResidualCNNHandler(ArchitectureHandler):
    """
    Residual blocks ``h_{l+1} = F_l(h_l) + h_l`` (torchvision BasicBlock /
    Bottleneck layout).  The trajectory consists of residual-branch outputs
    ``F_l(h_l)`` (output of the last batch-norm of the branch) and the
    descriptor of block ``l`` is the acceleration of the residual branch:

        D_res_l = ||F_l - 2 F_{l-1} + F_{l-2}|| / ||F_{l-1}||.

    Targets are the convolutions of the residual branch; shortcut
    (``downsample``) convolutions are not adapted.
    """

    name = "residual"

    def __init__(self):
        super().__init__()
        self.blocks: List[Tuple[str, nn.Module]] = []

    @staticmethod
    def find_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        return [
            (name, mod)
            for name, mod in model.named_modules()
            if all(hasattr(mod, a) for a in ("conv1", "bn1", "conv2", "bn2"))
        ]

    def matches(self, model):
        return len(self.find_blocks(model)) >= 2

    @staticmethod
    def _branch_end(block: nn.Module) -> nn.Module:
        return block.bn3 if hasattr(block, "bn3") else block.bn2

    def discover_targets(self, model, include=None, exclude=None):
        self.blocks = self.find_blocks(model)
        targets = {}
        for l, (bname, block) in enumerate(self.blocks):
            for sub_name, mod in block.named_modules():
                if not isinstance(mod, nn.Conv2d) or "downsample" in sub_name or mod.groups != 1:
                    continue
                full = f"{bname}.{sub_name}"
                if not self._filter(full, include, exclude):
                    continue
                targets[full] = TargetInfo(
                    full, "conv", mod, (mod.out_channels, mod.weight[0].numel()),
                    group=f"block{l}", meta={"layer": l},
                )
        return targets

    def attach(self, model, targets, store):
        if not self.blocks:
            self.blocks = self.find_blocks(model)
        handles = []
        for l, (_, block) in enumerate(self.blocks):

            def hook(module, args, output, l=l):
                store.save_state(f"F{l}", output)

            handles.append(self._branch_end(block).register_forward_hook(hook))
        handles.extend(_register_target_hooks(targets, store, save_states=False))
        return handles

    def compute_divergence(self, store, targets, settings):
        states = [store.states[f"F{l}"] for l in range(len(self.blocks)) if f"F{l}" in store.states]
        cache, result = {}, {}
        for name, info in targets.items():
            l = info.meta["layer"]
            if l not in cache:
                cache[l] = flow_change_from_sequence(
                    states, l - 1, TERMS_ACCELERATION, settings.eps, settings.alignment, settings.projector
                )["D"]
            result[name] = cache[l]
        return result

    def point_descriptors(self, store, settings):
        states = [store.states[f"F{l}"] for l in range(len(self.blocks)) if f"F{l}" in store.states]
        return {
            f"block{l}": flow_change_from_sequence(
                states, l - 1, TERMS_ACCELERATION, settings.eps, settings.alignment, settings.projector
            )["D"]
            for l in range(len(states))
        }

    def partition(self, target):
        """Partition of the flattened kernel ``C_out x (C_in k_h k_w)`` across output / input channels."""
        conv: nn.Conv2d = target.module
        return math.ceil(math.sqrt(conv.out_channels)), math.ceil(math.sqrt(conv.in_channels))


# ---------------------------------------------------------------------------
# Gated RNNs
# ---------------------------------------------------------------------------

GRU_GATES = ("r", "z", "n")
LSTM_GATES = ("i", "f", "g", "o")


def rnn_gates(rnn: nn.Module) -> Tuple[str, ...]:
    return GRU_GATES if isinstance(rnn, nn.GRU) else LSTM_GATES


def gate_weight(rnn: nn.Module, layer: int, gate_idx: int) -> torch.Tensor:
    """``W_g = [W_ig | W_hg]`` for gate ``gate_idx`` of layer ``layer``."""
    h = rnn.hidden_size
    w_ih = getattr(rnn, f"weight_ih_l{layer}")[gate_idx * h:(gate_idx + 1) * h]
    w_hh = getattr(rnn, f"weight_hh_l{layer}")[gate_idx * h:(gate_idx + 1) * h]
    return torch.cat([w_ih, w_hh], dim=1)


def unroll_rnn(
    rnn: nn.Module,
    x: torch.Tensor,
    hx: Any = None,
    weights: Optional[Dict[str, torch.Tensor]] = None,
    gate_callback: Optional[Callable[[int, str, List[torch.Tensor]], None]] = None,
):
    """
    Reference unrolled GRU / LSTM forward (unidirectional, unpacked input),
    numerically equivalent to the cuDNN kernel.  ``weights`` can override the
    module parameters (used by gate adapters); ``gate_callback(layer, gate,
    per_step_tensors)`` receives the gate activations of every layer.
    """
    is_lstm = isinstance(rnn, nn.LSTM)
    w = weights or {}
    get = lambda n: w.get(n, getattr(rnn, n, None))  # noqa: E731
    batch_first = rnn.batch_first
    seq = x.transpose(0, 1) if batch_first else x  # [T, N, I]
    unbatched = seq.dim() == 2
    if unbatched:
        seq = seq.unsqueeze(1)
    T, N = seq.shape[0], seq.shape[1]
    H = rnn.hidden_size

    if hx is None:
        h0 = seq.new_zeros(rnn.num_layers, N, H)
        c0 = seq.new_zeros(rnn.num_layers, N, H) if is_lstm else None
    elif is_lstm:
        h0, c0 = hx
        if unbatched:
            h0, c0 = h0.unsqueeze(1), c0.unsqueeze(1)
    else:
        h0, c0 = (hx.unsqueeze(1) if unbatched else hx), None

    layer_input = seq
    h_n, c_n = [], []
    for layer in range(rnn.num_layers):
        w_ih, w_hh = get(f"weight_ih_l{layer}"), get(f"weight_hh_l{layer}")
        b_ih, b_hh = get(f"bias_ih_l{layer}"), get(f"bias_hh_l{layer}")
        gi_all = torch.nn.functional.linear(layer_input, w_ih, b_ih)  # [T, N, G*H]
        h = h0[layer]
        c = c0[layer] if is_lstm else None
        outputs = []
        gates_steps: Dict[str, List[torch.Tensor]] = {g: [] for g in rnn_gates(rnn)}
        for t in range(T):
            gh = torch.nn.functional.linear(h, w_hh, b_hh)
            gi = gi_all[t]
            if is_lstm:
                i_g, f_g, g_g, o_g = (gi + gh).chunk(4, dim=-1)
                i_g, f_g, g_g, o_g = i_g.sigmoid(), f_g.sigmoid(), g_g.tanh(), o_g.sigmoid()
                c = f_g * c + i_g * g_g
                h = o_g * c.tanh()
                for g, v in zip(LSTM_GATES, (i_g, f_g, g_g, o_g)):
                    gates_steps[g].append(v)
            else:
                i_r, i_z, i_n = gi.chunk(3, dim=-1)
                h_r, h_z, h_n_ = gh.chunk(3, dim=-1)
                r = (i_r + h_r).sigmoid()
                z = (i_z + h_z).sigmoid()
                n = (i_n + r * h_n_).tanh()
                h = (1 - z) * n + z * h
                for g, v in zip(GRU_GATES, (r, z, n)):
                    gates_steps[g].append(v)
            outputs.append(h)
        if gate_callback is not None:
            for g, steps in gates_steps.items():
                gate_callback(layer, g, steps)
        layer_input = torch.stack(outputs, dim=0)
        if rnn.dropout and rnn.training and layer < rnn.num_layers - 1:
            layer_input = torch.nn.functional.dropout(layer_input, rnn.dropout, True)
        h_n.append(h)
        if is_lstm:
            c_n.append(c)

    output = layer_input
    h_n_t = torch.stack(h_n, 0)
    if unbatched:
        output, h_n_t = output.squeeze(1), h_n_t.squeeze(1)
    elif batch_first:
        output = output.transpose(0, 1)
    if is_lstm:
        c_n_t = torch.stack(c_n, 0)
        if unbatched:
            c_n_t = c_n_t.squeeze(1)
        return output, (h_n_t, c_n_t)
    return output, h_n_t


class GatedRNNHandler(ArchitectureHandler):
    """
    GRU / LSTM gates.  Each gate ``g`` of each layer is an adaptation target
    with weight ``W_g = [W_ig | W_hg]``.  The gate divergence is computed
    over time steps,

        D_g = mean_t ||g_t - 2 g_{t-1} + g_{t-2}|| / ||g_{t-1}||,

    and ``C_g = D_g * ||grad_{g_t} L||_2 * ||W_g||_F``.  During calibration
    the RNN forward is replaced by the numerically equivalent unrolled
    forward so that gate activations and their gradients are available.
    Bidirectional RNNs, projected LSTMs and packed sequences are not supported.
    """

    name = "rnn"

    def __init__(self):
        super().__init__()
        self._saved_flags: Dict[int, bool] = {}

    def matches(self, model):
        return any(isinstance(m, (nn.GRU, nn.LSTM)) for m in model.modules())

    def discover_targets(self, model, include=None, exclude=None):
        targets = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, (nn.GRU, nn.LSTM)) or mod.bidirectional or getattr(mod, "proj_size", 0):
                continue
            for layer in range(mod.num_layers):
                in_size = mod.input_size if layer == 0 else mod.hidden_size
                for gi, g in enumerate(rnn_gates(mod)):
                    full = f"{name}.l{layer}.{g}"
                    if not self._filter(full, include, exclude):
                        continue
                    targets[full] = TargetInfo(
                        full, "gate", mod, (mod.hidden_size, in_size + mod.hidden_size),
                        group=f"{name}.l{layer}", meta={"rnn": name, "layer": layer, "gate": g, "gate_idx": gi},
                    )
        return targets

    def attach(self, model, targets, store):
        handles = []
        rnns = {info.meta["rnn"]: info.module for info in targets.values()}
        for rnn_name, rnn in rnns.items():
            for p in rnn.parameters():
                self._saved_flags[id(p)] = p.requires_grad
                p.requires_grad_(True)

            def hook(module, args, output, rnn_name=rnn_name):
                if not args or not isinstance(args[0], torch.Tensor):
                    return None
                hx = args[1] if len(args) > 1 else None

                def on_gates(layer, gate, steps):
                    key = f"{rnn_name}.l{layer}.{gate}"
                    if key not in targets:
                        return
                    stacked = torch.stack(steps, dim=1)  # [N, T, H]
                    store.save_state(key, stacked, batch_first=True)
                    for step in steps:
                        store.add_grad_hook(key, step, batch_first=True)

                with torch.enable_grad():
                    return unroll_rnn(module, args[0], hx, gate_callback=on_gates)

            handles.append(rnn.register_forward_hook(hook))
            handles.append(_RestoreFlags(rnn, self._saved_flags))
        return handles

    def compute_divergence(self, store, targets, settings):
        result = {}
        for name in targets:
            g = store.states.get(name)
            if g is None:
                continue
            g = g.float()
            if g.shape[1] < 3:
                result[name] = torch.zeros(g.shape[0], device=g.device)
                continue
            acc = (g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]).norm(dim=-1)
            ref = g[:, 1:-1].norm(dim=-1)
            result[name] = (acc / (ref + settings.eps)).mean(dim=1)
        return result

    def weight_matrix(self, target):
        return gate_weight(target.module, target.meta["layer"], target.meta["gate_idx"])


class _RestoreFlags:
    """Hook-like handle that restores ``requires_grad`` flags and clears gradients."""

    def __init__(self, module: nn.Module, flags: Dict[int, bool]):
        self.module, self.flags = module, flags

    def remove(self):
        for p in self.module.parameters():
            p.requires_grad_(self.flags.get(id(p), False))
            p.grad = None


# ---------------------------------------------------------------------------
# Handler selection
# ---------------------------------------------------------------------------

HANDLERS = {
    "feedforward": FeedForwardHandler,
    "transformer": TransformerHandler,
    "residual": ResidualCNNHandler,
    "rnn": GatedRNNHandler,
}


def get_handler(model: nn.Module, name: str = "auto") -> ArchitectureHandler:
    if name != "auto":
        if name not in HANDLERS:
            raise ValueError(f"Unknown handler '{name}', available: {sorted(HANDLERS)}")
        return HANDLERS[name]()
    for cls in (GatedRNNHandler, TransformerHandler, ResidualCNNHandler):
        handler = cls()
        if handler.matches(model):
            return handler
    return FeedForwardHandler()


# ---------------------------------------------------------------------------
# Attention heads and transformer sub-layer split
# ---------------------------------------------------------------------------


def head_divergence(
    h: torch.Tensor,
    h_prev: torch.Tensor,
    num_heads: int,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """
    ``D_h = ||head_h(h) - 2 head_h(h')|| / ||head_h(h')||``.

    ``h`` and ``h_prev`` are batch-first representations whose last dimension
    is ``d_model``. Returns ``[N, H]``.
    """
    if h.shape[-1] % num_heads:
        raise ValueError(f"d_model={h.shape[-1]} is not divisible by num_heads={num_heads}")
    dh = h.shape[-1] // num_heads

    def split(t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        return t.reshape(t.shape[0], -1, num_heads, dh).movedim(-2, 1)

    hf, hp = split(h), split(h_prev)
    hf = hf.reshape(hf.shape[0], num_heads, -1)
    hp = hp.reshape(hp.shape[0], num_heads, -1)
    return (hf - 2.0 * hp).norm(dim=-1) / (hp.norm(dim=-1) + eps)


def head_flow_descriptors(
    states: Sequence[torch.Tensor],
    center: int,
    num_heads: int,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """
    Head-specific three-point flow-change descriptor.
    Returns ``[N, num_heads]``.
    """
    n = len(states)
    c = min(max(center, 1), n - 2)
    hp, hc, hn = (s.float() for s in align_states([states[c - 1], states[c], states[c + 1]]))
    d = hc.shape[-1]
    if d % num_heads:
        raise ValueError(f"d_model={d} is not divisible by num_heads={num_heads}")
    split = lambda t: t.reshape(*t.shape[:-1], num_heads, d // num_heads).movedim(-2, 1)  # noqa: E731
    hp, hc, hn = split(hp), split(hc), split(hn)  # [N, H, T, dh]
    out = []
    for h in range(num_heads):
        out.append(flow_change(hp[:, h], hc[:, h], hn[:, h], TERMS_CONV, eps)["D"])
    return torch.stack(out, dim=1)


def proportional_integer_split(weights: Sequence[float], total: int) -> List[int]:
    """Largest-remainder split of ``total`` proportional to non-negative ``weights``."""
    w = [max(float(x), 0.0) for x in weights]
    s = sum(w)
    if total <= 0 or not w:
        return [0] * len(w)
    if s <= 0:
        w, s = [1.0] * len(w), float(len(w))
    raw = [total * x / s for x in w]
    base = [int(math.floor(r)) for r in raw]
    order = sorted(range(len(w)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def allocate_head_ranks(head_criticality: Sequence[float], total_rank: int, gamma: float = 1.0) -> List[int]:
    """``r_h ∝ C_h^gamma`` for the heads of one attention layer."""
    return proportional_integer_split([max(c, 0.0) ** gamma for c in head_criticality], total_rank)


def split_attention_rank(rank: int, alpha: float = 0.7) -> Dict[str, int]:
    """Assign a fraction ``alpha`` of a head rank to V/O and the remainder to Q/K."""
    r_vo = int(round(alpha * rank))
    r_qk = rank - r_vo
    return {"v": r_vo - r_vo // 2, "o": r_vo // 2, "q": r_qk - r_qk // 2, "k": r_qk // 2}


def attention_ffn_rank_ratio(d_attn: float, g_attn: float, d_ffn: float, g_ffn: float, gamma: float = 1.0) -> float:
    """``r_attn / r_ffn = (D_attn ||grad Attn L|| / (D_ffn ||grad FFN L||))^gamma``."""
    num, den = d_attn * g_attn, d_ffn * g_ffn
    if den <= 0:
        return float("inf") if num > 0 else 1.0
    return (num / den) ** gamma
