"""
Small LLaMA-style decoder used by the usage example.

``TinyLlama`` mirrors the LLaMA layer structure (pre-norm RMSNorm, q/k/v/o
projections, SwiGLU FFN with gate/up/down projections) so that FLowRA
discovers exactly the same target modules as on LLaMA-2/3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (p(x).view(b, t, self.heads, self.dh).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d)
        self.self_attn = Attention(d, heads)
        self.post_attention_layernorm = nn.RMSNorm(d)
        self.mlp = MLP(d, hidden)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyLlama(nn.Module):
    """Decoder-only LLaMA-style classifier (last-token pooling)."""

    def __init__(self, vocab: int = 64, d: int = 64, layers: int = 4, heads: int = 4, hidden: int = 172,
                 num_classes: int = 4, max_len: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.layers = nn.ModuleList(Block(d, heads, hidden) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.score = nn.Linear(d, num_classes, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids) + self.pos[: input_ids.shape[1]]
        for block in self.layers:
            x = block(x)
        return self.score(self.norm(x)[:, -1])


# ----------------------------------------------------------------------
# Synthetic sequence-classification tasks
# ----------------------------------------------------------------------
def make_sequence_task(task: str, n: int, vocab: int = 64, length: int = 24, num_classes: int = 4, seed: int = 0,
                       task_token: int = None):
    """
    Token-sequence tasks with different decision rules:

    * ``count``   - class = most frequent token bucket (bucket = token % C);
    * ``rare``    - class = least frequent token bucket;
    * ``last``    - class = bucket of the last token;
    * ``first``   - class = identity of the "marker" token (tokens >= vocab - 8) mod C;
    * ``parity``  - class = (sum of tokens in the first half) mod C;
    * ``span``    - class = bucket of the token after the marker.

    ``task_token`` (if given) is written to position 0 so that several tasks can
    share one model (multi-task composition).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab - 8, (n, length), generator=g)
    C = num_classes
    if task_token is not None:
        x[:, 0] = task_token
    if task == "count":
        y = F.one_hot(x % C, C).sum(1).argmax(1)
    elif task == "rare":
        counts = F.one_hot(x[:, 1:] % C, C).sum(1).float()
        counts += torch.rand(counts.shape, generator=g) * 0.1
        y = counts.argmin(1)
    elif task == "last":
        y = x[:, -1] % C
    elif task == "parity":
        y = x[:, : length // 2].sum(1) % C
    elif task in ("first", "span"):
        pos = torch.randint(1, length - 2, (n,), generator=g)
        marker = torch.randint(vocab - 8, vocab, (n,), generator=g)
        x[torch.arange(n), pos] = marker
        if task == "first":
            y = (marker - (vocab - 8)) % C
        else:
            y = x[torch.arange(n), pos + 1] % C
    else:
        raise ValueError(task)
    return x, y


def accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / max(total, 1)
