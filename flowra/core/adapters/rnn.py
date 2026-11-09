"""
Gate-wise adapters for GRU / LSTM.

Gate ``g`` of layer ``k`` has weight ``W_g = [W_ig | W_hg]`` and receives
its own factors ``B_g A_g`` with rank ``r_g ∝ C_g^gamma``.  The updates are
added to the corresponding rows of ``weight_ih_l{k}`` / ``weight_hh_l{k}``
and the frozen RNN is evaluated with the modified weights.
"""

from typing import Dict, Mapping

import torch
import torch.nn as nn
from torch.func import functional_call

from flowra.core.adapters.base import BaseAdapter, LowRankFactors
from flowra.core.analysis.handlers import rnn_gates


class GatedRNNAdapter(BaseAdapter):
    def __init__(self, base: nn.Module, gate_ranks: Mapping[str, int], scaling: Mapping[str, float]):
        """``gate_ranks`` / ``scaling`` are keyed by ``"l{layer}.{gate}"``."""
        if not isinstance(base, (nn.GRU, nn.LSTM)) or base.bidirectional:
            raise TypeError("GatedRNNAdapter supports unidirectional nn.GRU / nn.LSTM")
        super().__init__(base)
        self.gate_names = rnn_gates(base)
        h = base.hidden_size
        w = base.weight_ih_l0
        self.gates = nn.ModuleDict()
        for layer in range(base.num_layers):
            in_size = base.input_size if layer == 0 else h
            for g in self.gate_names:
                key = f"l{layer}.{g}"
                self.gates[key.replace(".", "_")] = LowRankFactors(
                    h, in_size + h, gate_ranks.get(key, 0), scaling.get(key, 1.0), w.dtype, w.device
                )

    def factors(self) -> Dict[str, LowRankFactors]:
        return {k.replace("_", ".", 1): f for k, f in self.gates.items()}

    def _weights(self) -> Dict[str, torch.Tensor]:
        b = self.base
        h = b.hidden_size
        weights = {}
        for layer in range(b.num_layers):
            in_size = b.input_size if layer == 0 else h
            d_ih, d_hh = [], []
            for g in self.gate_names:
                f = self.gates[f"l{layer}_{g}"]
                if f.rank:
                    delta = f.delta()
                else:
                    delta = torch.zeros(h, in_size + h, dtype=b.weight_ih_l0.dtype, device=b.weight_ih_l0.device)
                d_ih.append(delta[:, :in_size])
                d_hh.append(delta[:, in_size:])
            weights[f"weight_ih_l{layer}"] = getattr(b, f"weight_ih_l{layer}") + torch.cat(d_ih, 0)
            weights[f"weight_hh_l{layer}"] = getattr(b, f"weight_hh_l{layer}") + torch.cat(d_hh, 0)
        return weights

    def forward(self, *args, **kwargs):
        if self.merged:
            return self.base(*args, **kwargs)
        return functional_call(self.base, self._weights(), args, kwargs)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            for name, w in self._weights().items():
                getattr(self.base, name).copy_(w)
            self.merged = True
