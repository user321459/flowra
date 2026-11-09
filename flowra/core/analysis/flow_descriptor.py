"""
Flow-change descriptor.

For three consecutive hidden representations ``h_{l-1}, h_l, h_{l+1}`` the
descriptor is

    D_l =  ||h_{l+1} - 2 h_l + h_{l-1}||_2 / (||h_l||_2 + eps)        (acceleration)
         * ||h_l - h_{l-1}||_2 / (||h_l||_2 + eps)                    (velocity)
         * <a, v> / (||a||_2 ||v||_2 + eps)                           (directional alignment)

with ``a = h_{l+1} - 2 h_l + h_{l-1}`` and ``v = h_l - h_{l-1}``.
Convolutional layers drop the alignment term; residual branches and gated
RNN divergences keep only the acceleration term.

All functions operate per sample: inputs are batch-first tensors ``[N, ...]``
and the result is a tensor of shape ``[N]``. The functions are differentiable
so that block sensitivities can be obtained by autograd.

The descriptor is used purely as an empirical calibration feature; no
information-theoretic interpretation is implied.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-8

TERMS_FULL: Tuple[str, ...] = ("acceleration", "velocity", "alignment")
TERMS_CONV: Tuple[str, ...] = ("acceleration", "velocity")
TERMS_ACCELERATION: Tuple[str, ...] = ("acceleration",)

ALIGNMENT_STRATEGIES = ("zero_pad", "linear", "conv1x1")


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).float()


def flow_change(
    h_prev: torch.Tensor,
    h: torch.Tensor,
    h_next: torch.Tensor,
    terms: Sequence[str] = TERMS_FULL,
    eps: float = DEFAULT_EPS,
) -> Dict[str, torch.Tensor]:
    """
    Per-sample flow-change descriptor.

    The three tensors must already have a common shape (see
    :func:`align_states`).  Norms are taken over all non-batch dimensions and
    normalised by the norm of the central representation ``h``.

    Returns a dict with the signed descriptor ``"D"`` and its components
    ``"acceleration"``, ``"velocity"`` and ``"alignment"`` (each ``[N]``).
    """
    hp, hc, hn = _flatten(h_prev), _flatten(h), _flatten(h_next)

    acc = hn - 2.0 * hc + hp
    vel = hc - hp

    center_norm = hc.norm(dim=1)
    acc_norm = acc.norm(dim=1)
    vel_norm = vel.norm(dim=1)

    acceleration = acc_norm / (center_norm + eps)
    velocity = vel_norm / (center_norm + eps)
    alignment = (acc * vel).sum(dim=1) / (acc_norm * vel_norm + eps)

    components = {
        "acceleration": acceleration,
        "velocity": velocity,
        "alignment": alignment,
    }
    unknown = set(terms) - set(components)
    if unknown:
        raise ValueError(f"Unknown descriptor terms: {sorted(unknown)}")

    descriptor = torch.ones_like(center_norm)
    for term in terms:
        descriptor = descriptor * components[term]

    components["D"] = descriptor
    return components


def flow_change_from_sequence(
    states: Sequence[torch.Tensor],
    center: int,
    terms: Sequence[str] = TERMS_FULL,
    eps: float = DEFAULT_EPS,
    alignment: str = "zero_pad",
    projector: Optional["FixedProjector"] = None,
) -> Dict[str, torch.Tensor]:
    """
    Descriptor at position ``center`` of an ordered trajectory of states.

    Boundary positions use the nearest complete three-point stencil, i.e. the
    window ``(center-1, center, center+1)`` is shifted inwards so that it lies
    inside the trajectory.  A trajectory with only two states has no
    second-order difference; in that case only the velocity term is returned.
    """
    n = len(states)
    if n == 0:
        raise ValueError("Empty trajectory")
    if n == 1:
        zeros = torch.zeros(states[0].shape[0], device=states[0].device)
        return {"D": zeros, "acceleration": zeros, "velocity": zeros, "alignment": zeros}
    if n == 2:
        a, b = align_states([states[0], states[1]], alignment, projector)
        vel = _flatten(b) - _flatten(a)
        velocity = vel.norm(dim=1) / (_flatten(b).norm(dim=1) + eps)
        zeros = torch.zeros_like(velocity)
        return {"D": velocity, "acceleration": zeros, "velocity": velocity, "alignment": zeros}

    c = min(max(center, 1), n - 2)
    hp, hc, hn = align_states([states[c - 1], states[c], states[c + 1]], alignment, projector)
    return flow_change(hp, hc, hn, terms=terms, eps=eps)


# ---------------------------------------------------------------------------
# Dimension alignment
# ---------------------------------------------------------------------------


class FixedProjector:
    """
    Fixed (non-trainable) semi-orthogonal projections used to align
    representations of different width during calibration.

    ``linear`` alignment projects the last dimension of vector-valued
    representations, ``conv1x1`` projects the channel dimension of feature
    maps (a fixed 1x1 convolution).  Projections are seeded, so repeated
    calibration runs are deterministic.  They are used only during
    calibration and never become part of the model.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def matrix(self, d_in: int, d_out: int, device: torch.device) -> torch.Tensor:
        key = (d_in, d_out, str(device))
        if key not in self._cache:
            gen = torch.Generator().manual_seed(self.seed + 7919 * d_in + d_out)
            g = torch.randn(max(d_in, d_out), min(d_in, d_out), generator=gen)
            q, _ = torch.linalg.qr(g)  # [max, min], orthonormal columns
            mat = q if d_out >= d_in else q.t()  # [d_out, d_in]
            self._cache[key] = mat.to(device)
        return self._cache[key]

    def project_dim(self, x: torch.Tensor, dim: int, d_out: int) -> torch.Tensor:
        d_in = x.shape[dim]
        if d_in == d_out:
            return x
        mat = self.matrix(d_in, d_out, x.device).to(x.dtype)
        x_moved = x.movedim(dim, -1)
        projected = x_moved @ mat.t()
        return projected.movedim(-1, dim)


def _zero_pad_to(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    pad: List[int] = []
    for dim in reversed(range(1, x.dim())):
        pad.extend([0, shape[dim] - x.shape[dim]])
    if any(pad):
        x = F.pad(x, pad)
    return x


def align_states(
    states: Sequence[torch.Tensor],
    strategy: str = "zero_pad",
    projector: Optional[FixedProjector] = None,
) -> List[torch.Tensor]:
    """
    Map batch-first representations to a common shape.

    * ``zero_pad`` (default): zero-pad every non-batch dimension to the
      maximum size.  Tensors of different rank are flattened first.
    * ``linear``: fixed semi-orthogonal projection of the last dimension to
      the largest width, remaining dimensions are zero-padded.
    * ``conv1x1``: fixed projection of the channel dimension (dim 1) of
      feature maps, spatial dimensions are zero-padded.
    """
    if strategy not in ALIGNMENT_STRATEGIES:
        raise ValueError(f"Unknown alignment strategy '{strategy}', expected one of {ALIGNMENT_STRATEGIES}")

    states = list(states)
    if all(s.shape == states[0].shape for s in states):
        return states

    if len({s.dim() for s in states}) > 1:
        states = [s.reshape(s.shape[0], -1) for s in states]

    if strategy != "zero_pad":
        projector = projector or FixedProjector()
        dim = -1 if strategy == "linear" else 1
        if states[0].dim() >= 2 and (strategy == "linear" or states[0].dim() >= 3):
            width = max(s.shape[dim] for s in states)
            states = [projector.project_dim(s, dim, width) for s in states]

    target = [max(s.shape[d] for s in states) for d in range(states[0].dim())]
    return [_zero_pad_to(s, target) for s in states]
