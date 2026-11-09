"""
Attach a PEFT baseline or FLowRA to a pre-trained backbone.

    flowra   - calibration, greedy rank allocation, refinement
    lora     - uniform rank r=16, LoRA scaling, no refinement
    dora     - same placement as LoRA with DoRA factors
    adalora  - PEFT AdaLoRA on the same target names
    adapter  - Houlsby after attention and FFN / Conv-Adapter on CNN blocks
    prefix   - Prefix-Tuning (Transformers), VPT (ViT), padded visual prompt (CNN)
    bitfit   - bias-only
    full_ft  - all parameters trainable
    frozen   - classification / task head only
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowra import FLOWRA, FlowConfig, TrainingConfig
from flowra.algorithms.allocation import AllocationResult, uniform_rank_for_budget, uniform_ranks
from flowra.core.training import FlowraTrainer
from flowra.utils.logging_utils import ExperimentLogger

METHODS = ("flowra", "lora", "dora", "adalora", "adapter", "prefix", "bitfit", "full_ft", "frozen")


def flow_config(cfg: Dict[str, Any], seed: int, device: str, **overrides) -> FlowConfig:
    raw = dict(cfg.get("flowra") or {})
    raw.update(overrides)
    raw["seed"] = seed
    raw["device"] = device
    return FlowConfig.from_dict(raw)


def _uniform_allocation(fw: FLOWRA, rank: Optional[int], budget: Optional[int] = None) -> AllocationResult:
    shapes = {n: t.shape for n, t in fw.targets.items()}
    if rank is None:
        rank = max(uniform_rank_for_budget(shapes, budget or fw.budget_total), 1)
    ranks = uniform_ranks(shapes, int(rank))
    used = {n: r * (shapes[n][0] + shapes[n][1]) for n, r in ranks.items()}
    total = sum(used.values())
    return AllocationResult(ranks, total, used, dict(ranks), 0)


def _inject_uniform(model: nn.Module, cfg: FlowConfig, rank: Optional[int], logger: ExperimentLogger) -> FLOWRA:
    fw = FLOWRA(model, cfg, logger)
    fw.allocation = _uniform_allocation(fw, rank)
    fw.inject()
    return fw


def _peft_targets(model: nn.Module) -> List[str]:
    names = [n.split(".")[-1] for n, _ in model.named_modules()
             if n.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj", "query", "key", "value"}]
    return sorted(set(names)) or ["q_proj", "v_proj"]


def _peft_task_type(model: nn.Module):
    from peft import TaskType
    name = type(model).__name__.lower()
    if "causallm" in name or "conditionalgeneration" in name:
        return TaskType.CAUSAL_LM
    if "sequenceclassification" in name:
        return TaskType.SEQ_CLS
    return TaskType.FEATURE_EXTRACTION


class HoulsbyWrap(nn.Module):
    """Houlsby bottleneck on the residual stream after attention or FFN."""

    def __init__(self, module: nn.Module, dim: int, reduction: int = 16):
        super().__init__()
        self.module = module
        r = max(dim // reduction, 8)
        self.down = nn.Linear(dim, r, bias=True)
        self.up = nn.Linear(r, dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            hidden = out[0] + self.up(F.relu(self.down(out[0])))
            return (hidden,) + out[1:]
        return out + self.up(F.relu(self.down(out)))


class ResidualConvAdapter(nn.Module):
    """1x1 residual bottleneck after a CNN block."""

    def __init__(self, block: nn.Module, channels: int, reduction: int = 16):
        super().__init__()
        self.block = block
        r = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, r, 1)
        self.up = nn.Conv2d(r, channels, 1)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.block(x)
        return y + self.up(F.relu(self.down(y)))


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def _hidden_dim(mod: nn.Module) -> Optional[int]:
    for attr in ("hidden_size", "embed_dim", "n_embd", "d_model"):
        v = getattr(mod, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    cfg = getattr(mod, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "n_embd", "d_model"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    for child in mod.modules():
        if isinstance(child, nn.Linear):
            return child.in_features
    return None


def _unfreeze_peft_and_heads(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if any(k in n for k in ("down.", "up.", "vpt_prefix", "visual_prompt",
                                "score", "classifier", "head", "fc", "lm_head")):
            p.requires_grad_(True)


def _apply_adapter(model: nn.Module, reduction: int = 16) -> nn.Module:
    """Houlsby after attention and FFN; Conv-Adapter after ResNet residual blocks."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = False
    attn_names = ("self_attn", "attn", "attention")
    ffn_names = ("mlp", "ffn", "feed_forward")
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (HoulsbyWrap, ResidualConvAdapter)):
            continue
        attn_attr = next((a for a in attn_names if hasattr(mod, a)), None)
        ffn_attr = next((a for a in ffn_names if hasattr(mod, a)), None)
        if ffn_attr is None and hasattr(mod, "intermediate") and hasattr(mod, "output"):
            ffn_attr = "output"  # BERT FFN residual (hidden size, not intermediate)
        if attn_attr and ffn_attr:
            dim = _hidden_dim(mod)
            if dim is None:
                continue
            setattr(mod, attn_attr, HoulsbyWrap(getattr(mod, attn_attr), dim, reduction))
            setattr(mod, ffn_attr, HoulsbyWrap(getattr(mod, ffn_attr), dim, reduction))
            wrapped = True
            continue
        kind = type(mod).__name__
        if name and kind in ("Bottleneck", "BasicBlock"):
            last = getattr(mod, "conv3", None) or getattr(mod, "conv2", None)
            channels = getattr(last, "out_channels", None)
            if channels:
                _set_submodule(model, name, ResidualConvAdapter(mod, channels, reduction))
                wrapped = True
    if not wrapped:
        for name, mod in list(model.named_modules()):
            if name and isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) >= 32:
                _set_submodule(model, name, HoulsbyWrap(mod, mod.out_features, reduction))
    _unfreeze_peft_and_heads(model)
    return model


def _apply_bitfit(model: nn.Module) -> nn.Module:
    for n, p in model.named_parameters():
        p.requires_grad_("bias" in n or any(k in n for k in ("score", "classifier", "head", "fc")))
    return model


def _apply_prefix(model: nn.Module, n_tokens: int = 20, visual_prompt_pad: int = 16) -> nn.Module:
    """Prefix-Tuning (Li & Liang) on Transformers; VPT on ViT; padded visual prompt on CNN."""
    name = type(model).__name__.lower()
    if any(k in name for k in ("llama", "gpt", "bloom", "opt", "t5", "bert", "roberta", "blip",
                               "causal", "seq", "conditionalgeneration")):
        try:
            from peft import PrefixTuningConfig, get_peft_model
            cfg = PrefixTuningConfig(task_type=_peft_task_type(model), num_virtual_tokens=n_tokens)
            return get_peft_model(model, cfg)
        except Exception:
            pass
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "patch_embed") and hasattr(backbone, "blocks"):
        dim = getattr(backbone, "embed_dim", None) or getattr(backbone, "num_features")
        backbone.vpt_prefix = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.normal_(backbone.vpt_prefix, std=0.02)
        orig = backbone.forward_features

        def forward_features(x, orig=orig, bb=backbone):
            tokens = orig(x)
            if tokens.dim() == 3:
                p = bb.vpt_prefix.expand(tokens.size(0), -1, -1)
                tokens = torch.cat([tokens[:, :1], p, tokens[:, 1:]], dim=1)
            return tokens
        backbone.forward_features = forward_features
        for p in model.parameters():
            p.requires_grad_(False)
        backbone.vpt_prefix.requires_grad_(True)
        _unfreeze_peft_and_heads(model)
        return model
    pad = max(int(visual_prompt_pad), 1)
    model.visual_prompt_pad = pad
    model.visual_prompt = nn.Parameter(0.02 * torch.randn(1, 3, 32, 32))
    orig = model.forward

    def wrapped(*args, pixel_values=None, labels=None, orig=orig, m=model, **kwargs):
        x = pixel_values if isinstance(pixel_values, torch.Tensor) else (args[0] if args else None)
        if not (isinstance(x, torch.Tensor) and x.dim() == 4):
            return orig(*args, pixel_values=pixel_values, labels=labels, **kwargs)
        b, _, h, w = x.shape
        p = m.visual_prompt_pad
        canvas = F.interpolate(m.visual_prompt, size=(h + 2 * p, w + 2 * p), mode="bilinear",
                               align_corners=False).expand(b, -1, -1, -1).clone()
        canvas[:, :, p:p + h, p:p + w] = x
        if pixel_values is not None:
            return orig(*args, pixel_values=canvas, labels=labels, **kwargs)
        return orig(canvas, *args[1:], **kwargs)
    model.forward = wrapped
    for p in model.parameters():
        p.requires_grad_(False)
    model.visual_prompt.requires_grad_(True)
    _unfreeze_peft_and_heads(model)
    return model


def attach(method: str, model: nn.Module, calib, cfg: Dict[str, Any], seed: int, device: str,
           logger: ExperimentLogger) -> Tuple[nn.Module, Optional[FLOWRA], Callable[[], List[nn.Parameter]]]:
    """
    Return ``(model, flowra_or_none, params_fn)``.
    ``calib`` is used only by FLowRA.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")

    rank = None if cfg.get("match_budget") else int(cfg.get("lora_rank", 16))
    if method == "flowra":
        fw = FLOWRA(model, flow_config(cfg, seed, device), logger)
        fw.adapt(calib)
        return fw.model, fw, fw.trainable_parameters

    if method == "lora":
        fcfg = flow_config(cfg, seed, device, scaling="lora", refinement=False, polymorphic=False,
                           adapter_variant="lora", use_flow=False)
        fw = _inject_uniform(model, fcfg, rank, logger)
        return fw.model, fw, fw.trainable_parameters

    if method == "dora":
        fcfg = flow_config(cfg, seed, device, scaling="lora", refinement=False, polymorphic=False,
                           adapter_variant="dora", use_flow=False)
        fw = _inject_uniform(model, fcfg, rank, logger)
        return fw.model, fw, fw.trainable_parameters

    if method == "adalora":
        from peft import AdaLoraConfig, get_peft_model
        peft_cfg = AdaLoraConfig(
            task_type=_peft_task_type(model),
            init_r=max(rank or 16, 12), target_r=max((rank or 16) // 2, 4), lora_alpha=16.0,
            target_modules=_peft_targets(model), lora_dropout=0.0,
        )
        model = get_peft_model(model, peft_cfg)
        return model, None, lambda: [p for p in model.parameters() if p.requires_grad]

    if method == "adapter":
        model = _apply_adapter(model, reduction=int(cfg.get("adapter_reduction", 16)))
        return model, None, lambda: [p for p in model.parameters() if p.requires_grad]

    if method == "prefix":
        model = _apply_prefix(model, n_tokens=int(cfg.get("prefix_tokens", 20)),
                              visual_prompt_pad=int(cfg.get("visual_prompt_pad", 16)))
        return model, None, lambda: [p for p in model.parameters() if p.requires_grad]

    if method == "bitfit":
        model = _apply_bitfit(model)
        return model, None, lambda: [p for p in model.parameters() if p.requires_grad]

    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad_(True)
        return model, None, lambda: [p for p in model.parameters() if p.requires_grad]

    saved = list((cfg.get("flowra") or {}).get("modules_to_save") or [])
    for p in model.parameters():
        p.requires_grad_(False)
    if saved:
        for name, mod in model.named_modules():
            if name and any(name == s or name.endswith("." + s) for s in saved):
                for p in mod.parameters():
                    p.requires_grad_(True)
    else:
        for name, p in model.named_parameters():
            if any(k in name for k in ("score", "classifier", "lm_head", "fc")):
                p.requires_grad_(True)
    return model, None, lambda: [p for p in model.parameters() if p.requires_grad]


def train_attached(model: nn.Module, fw: Optional[FLOWRA], params_fn, train, tcfg: TrainingConfig,
                   eval_fn, logger: ExperimentLogger) -> None:
    if fw is not None:
        fw.train(train, tcfg, eval_fn=eval_fn)
        return
    FlowraTrainer(model, params_fn, train, tcfg, eval_fn=eval_fn, logger=logger,
                  device=next(model.parameters()).device).train()


def trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
