"""Attribute heads. A head maps backbone tokens (B, T, D) to logits (B, A).

tokens[:, 0] is the CLS token, tokens[:, :n_prefix] are CLS + register tokens,
tokens[:, n_prefix:] are the patch tokens.
"""
import torch
import torch.nn as nn

from ..registry import Registry

HEADS = Registry("head")


@HEADS.register("linear")
class LinearHead(nn.Sequential):
    """[CLS ; mean(patches)] -> LayerNorm -> Dropout -> Linear. One shared feature for all attributes.

    Subclasses nn.Sequential so parameter names stay head.0.* / head.2.* (checkpoints of submission v3).
    """

    def __init__(self, dim, num_attrs, drop=0.1):
        super().__init__(nn.LayerNorm(2 * dim), nn.Dropout(drop), nn.Linear(2 * dim, num_attrs))

    def forward(self, tokens, n_prefix):
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], 1)
        return super().forward(feat)


@HEADS.register("query")
class QueryHead(nn.Module):
    """One learned query per attribute cross-attends to the tokens (ML-Decoder / Query2Label style).

    Each attribute can look at its own region (head for hats and glasses, legs for trousers), instead of
    sharing one pooled vector. Output of query a goes through its own linear classifier (group FC).
    """

    def __init__(self, dim, num_attrs, n_layers=1, n_heads=8, drop=0.1, mlp_ratio=2.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "q_norm": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True),
                "mlp_norm": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(int(dim * mlp_ratio), dim)),
            }))
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))

    def forward(self, tokens, n_prefix):
        kv = self.kv_norm(torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1))  # CLS + patches, no registers
        q = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            h = layer["q_norm"](q)
            q = q + layer["attn"](h, kv, kv, need_weights=False)[0]
            q = q + layer["mlp"](layer["mlp_norm"](q))
        q = self.drop(self.out_norm(q))
        return (q * self.fc_weight).sum(-1) + self.fc_bias
