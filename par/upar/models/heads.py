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


class MoEFFN(nn.Module):
    """Sparse mixture-of-experts MLP: a router sends each vector to its top-k of E expert MLPs (gates renormalised over
    the k chosen). Train mode computes every expert densely with zero gates for the unchosen (compile-friendly, same
    result); eval mode only runs the chosen experts (CPU inference). `last_aux` holds the Switch load-balancing loss
    E * sum_e f_e * P_e (f_e: share of routing slots to expert e, P_e: mean router probability), 1 when balanced."""

    def __init__(self, dim, n_experts=8, topk=2, hidden_mult=2.0, drop=0.1):
        super().__init__()
        h = int(dim * hidden_mult)
        self.E, self.k = n_experts, topk
        self.router = nn.Linear(dim, n_experts, bias=False)
        nn.init.zeros_(self.router.weight)  # uniform routing at start
        self.w1 = nn.Parameter(torch.randn(n_experts, dim, h) * dim ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n_experts, h))
        self.w2 = nn.Parameter(torch.randn(n_experts, h, dim) * h ** -0.5 * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_experts, dim))
        self.drop = nn.Dropout(drop)
        self.last_aux = None

    def gates(self, x):
        probs = self.router(x).float().softmax(-1)                       # (N, E)
        topv, topi = probs.topk(self.k, -1)
        g = torch.zeros_like(probs).scatter(-1, topi, topv / topv.sum(-1, keepdim=True))
        f = torch.zeros_like(probs).scatter(-1, topi, 1.0).mean(0) / self.k
        self.last_aux = self.E * (f * probs.mean(0)).sum()
        return g.to(x.dtype), topi

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        g, topi = self.gates(x)
        if self.training:
            hid = self.drop(torch.nn.functional.gelu(torch.einsum("nd,edh->enh", x, self.w1) + self.b1[:, None]))
            y = torch.einsum("enh,ehd->end", hid, self.w2) + self.b2[:, None]   # (E, N, D)
            out = torch.einsum("ne,end->nd", g, y)
        else:
            out = torch.zeros_like(x)
            for e in range(self.E):
                rows = (topi == e).any(-1).nonzero(as_tuple=True)[0]
                if rows.numel():
                    hid = torch.nn.functional.gelu(x[rows] @ self.w1[e] + self.b1[e])
                    out.index_add_(0, rows, g[rows, e:e + 1] * (hid @ self.w2[e] + self.b2[e]))
        return out.reshape(shape)


# vertical centre (0 = top, 1 = bottom of the crop) of each attribute's region; None = whole body (gender, age)
ATTR_ROW = {0: None, 1: None, 2: None, 3: None, 4: .12, 5: .15, 6: .10, 7: .35, **{k: .35 for k in range(8, 20)},
            20: .65, **{k: .65 for k in range(21, 33)}, 33: .65, 34: .65, 35: .35, 36: .50, 37: .12, 38: .12, 39: .08}


def body_prior(grid, num_attrs, sigma=0.18):
    """(A, H*W) additive attention bias: -(y - c_k)^2 / (2 sigma^2) on the patch row y, 0 for whole-body attributes."""
    H, W = grid
    y = (torch.arange(H).float() + 0.5) / H
    out = torch.zeros(num_attrs, H, W)
    for k in range(num_attrs):
        c = ATTR_ROW.get(k)
        if c is not None:
            out[k] = (-(y - c) ** 2 / (2 * sigma ** 2))[:, None].expand(H, W)
    return out.reshape(num_attrs, H * W)


class EfficientCrossAttention(nn.Module):
    """Cross-attention of the attribute queries over the tokens without the softmax over tokens.

    linear:  out_i = phi(q_i)^T (sum_j phi(k_j) v_j^T) / (phi(q_i)^T sum_j phi(k_j)), phi = elu + 1 (Katharopoulos et
             al. 2020). Cost linear in the number of tokens; weights are positive but less peaked than a softmax.
    sigmoid: out_i = sum_j sigmoid(q_i.k_j / sqrt(d) - log n) v_j (Ramapuram et al. 2024). Every token is gated
             independently, so a query can take several regions at full weight instead of sharing one unit of mass.
    Same call convention as nn.MultiheadAttention(batch_first=True) for the parts QueryHead uses."""

    def __init__(self, dim, n_heads, kind="linear", drop=0.0):
        super().__init__()
        self.h, self.kind, self.d = n_heads, kind, dim // n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.drop = nn.Dropout(drop)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        B, A, D = query.shape
        N = key.shape[1]
        q = self.q(query).view(B, A, self.h, self.d).transpose(1, 2)          # (B, h, A, d)
        k = self.k(key).view(B, N, self.h, self.d).transpose(1, 2)            # (B, h, N, d)
        v = self.v(value).view(B, N, self.h, self.d).transpose(1, 2)
        if self.kind == "linear":
            q, k = nn.functional.elu(q) + 1, nn.functional.elu(k) + 1
            kv = k.transpose(-2, -1) @ v                                       # (B, h, d, d)
            z = 1 / (q @ k.sum(2, keepdim=True).transpose(-2, -1) + 1e-6)      # (B, h, A, 1)
            out = (q @ kv) * z
        else:
            logits = q @ k.transpose(-2, -1) / self.d ** 0.5 - torch.log(torch.tensor(float(N)))
            if attn_mask is not None:
                logits = logits + attn_mask.view(B, self.h, A, N)
            out = self.drop(torch.sigmoid(logits)) @ v
        return self.o(out.transpose(1, 2).reshape(B, A, D)), None


@HEADS.register("query")
class QueryHead(nn.Module):
    supports_q = True  # forward(..., return_q=True) also returns the attribute embeddings (training-only decoders)

    """One learned query per attribute cross-attends to the tokens (ML-Decoder / Query2Label style).

    Each attribute can look at its own region (head for hats and glasses, legs for trousers), instead of
    sharing one pooled vector. Output of query a goes through its own linear classifier (group FC).
    """

    def __init__(self, dim, num_attrs, n_layers=1, n_heads=8, drop=0.1, mlp_ratio=2.0, self_attn=False,
                 n_domains=1, avg_domains=None, moe_experts=0, moe_topk=2, moe_where="tokens",
                 spatial_prior=False, fg_bias=False, grid=None, attn_type="softmax"):
        super().__init__()
        # attention selectivity: a learnable body-part prior per attribute query and / or a data-dependent foreground
        # bias (CLS-patch similarity of the backbone) added to the attention logits over the patch tokens
        self.n_heads = n_heads
        self.prior = nn.Parameter(body_prior(grid, num_attrs)) if spatial_prior else None
        self.fg_w = nn.Parameter(torch.ones(n_heads)) if fg_bias else None
        self.fg_mode = fg_bias if isinstance(fg_bias, str) else "cls"   # fg_bias=true / "cls" or "pca" (spotlight)
        # mixture of experts: "tokens" routes every patch token to its experts before the attribute queries read them
        # (feature-type specific processing); "queries" replaces the query MLP (attribute-specific processing)
        self.tok_moe = None
        if moe_experts and "tokens" in moe_where:
            self.tok_moe_norm = nn.LayerNorm(dim)
            self.tok_moe = MoEFFN(dim, moe_experts, moe_topk, mlp_ratio, drop)
        q_moe = bool(moe_experts) and "queries" in moe_where
        self.self_attn, self.n_domains = self_attn, n_domains
        self.register_buffer("avg_domains", torch.tensor(avg_domains if avg_domains is not None else range(n_domains)),
                             persistent=False)
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            extra = {"sa_norm": nn.LayerNorm(dim),  # attributes exchange information (Query2Label decoder)
                     "sa": nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)} if self_attn else {}
            self.layers.append(nn.ModuleDict({
                **extra,
                "q_norm": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True) if attn_type == "softmax"
                else EfficientCrossAttention(dim, n_heads, attn_type, drop),
                "mlp_norm": nn.LayerNorm(dim),
                "mlp": MoEFFN(dim, moe_experts, moe_topk, mlp_ratio, drop) if q_moe else
                nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                              nn.Linear(int(dim * mlp_ratio), dim)),
            }))
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        if n_domains == 1:  # parameter names of the plain query head
            self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
            self.fc_bias = nn.Parameter(torch.zeros(num_attrs))
        else:  # one classifier per source dataset: each keeps its own labelling convention
            self.fc_weight = nn.Parameter(torch.randn(n_domains, num_attrs, dim) * 0.02)
            self.fc_bias = nn.Parameter(torch.zeros(n_domains, num_attrs))

    def forward(self, tokens, n_prefix, ds=None, return_q=False):
        logits, q = self._logits(tokens, n_prefix, ds)
        return (logits, q) if return_q else logits

    def _logits(self, tokens, n_prefix, ds=None):
        t = torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1)  # CLS + patches, no registers
        if self.tok_moe is not None:
            t = t + self.tok_moe(self.tok_moe_norm(t))
        kv = self.kv_norm(t)
        B, A = tokens.shape[0], self.queries.shape[0]
        mask = None
        if self.prior is not None or self.fg_w is not None:
            S = kv.shape[1]                                            # CLS + patches
            mask = kv.new_zeros(B, self.n_heads, A, S)
            if self.prior is not None:
                mask[..., 1:] = mask[..., 1:] + self.prior.to(kv.dtype)
            if self.fg_w is not None:
                c = nn.functional.normalize(t[:, :1].float(), dim=-1)
                pt = nn.functional.normalize(t[:, 1:].float(), dim=-1)
                sim = (pt @ c.transpose(1, 2)).squeeze(-1)             # (B, N) CLS-patch cosine similarity
                if self.fg_mode == "pca":
                    # PCA spotlight: projection of each patch on the image's first principal component (5 power
                    # iterations on the centred patch tokens). The first component separates person from background
                    # in DINO features; its sign is fixed so that it agrees with the CLS similarity.
                    X = t[:, 1:].float().detach()
                    X = X - X.mean(1, keepdim=True)
                    v = X.mean(1)[..., None] + 1e-3
                    v = torch.randn_like(v) if not torch.isfinite(v).all() else v
                    for _ in range(5):
                        v = nn.functional.normalize(X.transpose(1, 2) @ (X @ v), dim=1)
                    proj = (X @ v).squeeze(-1)                         # (B, N)
                    sgn = torch.sign(((proj - proj.mean(1, keepdim=True)) * (sim - sim.mean(1, keepdim=True))).sum(1, keepdim=True))
                    sim = proj * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                z = (sim - sim.mean(1, keepdim=True)) / (sim.std(1, keepdim=True) + 1e-6)
                mask[..., 1:] = mask[..., 1:] + (self.fg_w[None, :, None, None] * z[:, None, None, :]).to(kv.dtype)
            mask = mask.reshape(B * self.n_heads, A, S)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            if self.self_attn:
                h = layer["sa_norm"](q)
                q = q + layer["sa"](h, h, h, need_weights=False)[0]
            h = layer["q_norm"](q)
            q = q + layer["attn"](h, kv, kv, need_weights=False, attn_mask=mask)[0]
            q = q + layer["mlp"](layer["mlp_norm"](q))
        q = self.drop(self.out_norm(q))
        if self.n_domains == 1:
            return (q * self.fc_weight).sum(-1) + self.fc_bias, q
        if ds is None:  # unseen domain: average the conventions of the datasets trained on
            w, b = self.fc_weight[self.avg_domains].mean(0), self.fc_bias[self.avg_domains].mean(0)
            return (q * w).sum(-1) + b, q
        return (q * self.fc_weight[ds]).sum(-1) + self.fc_bias[ds], q


@HEADS.register("dual")
class DualHead(nn.Module):
    """Query head and linear head on the same tokens, logits blended: a two-model ensemble for one backbone pass.
    The linear head keeps the ranking of a global feature, the query head adds per-attribute spatial evidence."""

    def __init__(self, dim, num_attrs, alpha=0.5, **query_kw):
        super().__init__()
        self.alpha = alpha
        self.query = QueryHead(dim, num_attrs, **query_kw)
        self.linear = LinearHead(dim, num_attrs)

    def forward(self, tokens, n_prefix):
        return self.alpha * self.query(tokens, n_prefix) + (1 - self.alpha) * self.linear(tokens, n_prefix)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, c, k=7, drop=0.1):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = nn.GroupNorm(1, c)  # = LayerNorm over (C, H, W), batch-size independent
        self.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Dropout(drop), nn.Conv2d(2 * c, c, 1))

    def forward(self, x):
        return x + self.mlp(self.norm(self.dw(x)))


@HEADS.register("conv")
class ConvHead(nn.Module):
    """CNN head on the ViT patch grid: patch tokens -> (B, D, H, W) -> 1x1 reduce -> ConvNeXt blocks -> CSRA pooling.

    CSRA (Zhu & Wu, ICCV 2021): each attribute gets a spatial score map; its logit is the mean score plus lam times
    an attention-weighted score, so small attributes (hat, glasses, shoes) come from their strongest region instead
    of being diluted by global averaging. The CLS token is added to every cell as global context."""

    def __init__(self, dim, num_attrs, grid=(16, 8), width=256, n_blocks=2, lam=0.3, temp=1.0, drop=0.1):
        super().__init__()
        self.grid, self.lam, self.temp = tuple(grid), lam, temp
        self.in_norm = nn.LayerNorm(dim)
        self.reduce = nn.Conv2d(dim, width, 1)
        self.blocks = nn.Sequential(*[_ConvNeXtBlock(width, drop=drop) for _ in range(n_blocks)])
        self.out_norm = nn.GroupNorm(1, width)
        self.drop = nn.Dropout2d(drop)
        self.score = nn.Conv2d(width, num_attrs, 1)

    def forward(self, tokens, n_prefix):
        t = self.in_norm(tokens)
        cls, patches = t[:, :1], t[:, n_prefix:]
        H, W = self.grid
        B, N, D = patches.shape
        assert N == H * W, f"{N} patch tokens do not fit the grid {self.grid}"
        x = (patches + cls).transpose(1, 2).reshape(B, D, H, W)
        x = self.drop(self.out_norm(self.blocks(self.reduce(x))))
        s = self.score(x).flatten(2)                     # (B, A, H*W) per-attribute score maps
        att = (s * self.temp).softmax(-1)
        return s.mean(-1) + self.lam * (att * s).sum(-1)


# attributes from global to local, top to bottom: gender, age, hair, hat, glasses, upper length / colour,
# lower length / type / colour, backpack, bag. Groups are read one after the other by the sequential heads.
SEQ_GROUPS = [[3], [0, 1, 2], [4, 5, 6], [39], [37, 38], [7], list(range(8, 20)), [20], [33, 34],
              list(range(21, 33)), [35], [36]]


class SelectiveSSM(nn.Module):
    """Mamba (S6) block over a sequence, causal: h_t = exp(dt_t * A) h_{t-1} + dt_t B_t u_t, y_t = C_t h_t + D u_t,
    with dt, B, C computed from the input (selective: each step decides how much of the past to keep), gated by
    SiLU(z), short causal depthwise conv in front (Gu & Dao 2023). A linear recurrence, so it can also be evaluated
    with a parallel scan; the sequences here are short (<= 40 steps) and use the plain loop."""

    def __init__(self, dim, d_inner=512, d_state=16, d_conv=4):
        super().__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state)  # dt (shared rank-1), B, C
        self.dt_proj = nn.Linear(1, d_inner)
        nn.init.constant_(self.dt_proj.bias, -2.0)  # dt ~ softplus(-2) = 0.13 at start
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, dim)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)  # starts as identity (residual)

    def forward(self, x):  # (B, L, dim) -> (B, L, dim), causal along L
        Bsz, L, _ = x.shape
        u, z = self.in_proj(self.norm(x)).chunk(2, -1)
        u = nn.functional.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_r, Bm, Cm = self.x_proj(u).split([1, self.d_state, self.d_state], -1)
        dt = nn.functional.softplus(self.dt_proj(dt_r)).float()                 # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                                       # (d_inner, N)
        h = x.new_zeros(Bsz, self.d_inner, self.d_state, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * u[:, t].float())[..., None] * Bm[:, t, None].float()
            ys.append((h * Cm[:, t, None].float()).sum(-1))
        y = (torch.stack(ys, 1) + u.float() * self.D).to(x.dtype) * nn.functional.silu(z)
        return x + self.out_proj(y)


@HEADS.register("seq")
class SequentialHead(nn.Module):
    """Attribute queries read the image, then information flows through the attributes in SEQ_GROUPS order.

    mode="mamba": all queries cross-attend in parallel, then a selective SSM scans the 40 attribute features in
      order (parallel reading, sequential reasoning).
    mode="seqattn": groups read the image one after another; a selective gated state carries what the previous
      groups found and is added to the next group's queries before they attend, so where a group looks depends
      on what was found before (age, then hair, ...). Within a group the queries work in parallel.
    Only hidden states are passed on, never predicted labels (no exposure bias, no hard co-occurrence rules)."""

    def __init__(self, dim, num_attrs, mode="mamba", n_heads=8, drop=0.1, mlp_ratio=2.0, d_state=16, d_inner=512):
        super().__init__()
        assert sorted(sum(SEQ_GROUPS, [])) == list(range(num_attrs))
        self.mode = mode
        self.order = [a for g in SEQ_GROUPS for a in g]
        self.register_buffer("inv_order", torch.argsort(torch.tensor(self.order)), persistent=False)
        self.queries = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=drop, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        if mode == "mamba":
            self.ssm = SelectiveSSM(dim, d_inner, d_state)
        elif mode == "seqattn":
            self.state_in = nn.Linear(dim, dim)        # group summary -> candidate state
            self.state_gate = nn.Linear(2 * dim, dim)  # selective forget gate from (state, summary)
            self.state_out = nn.Linear(dim, dim)       # state -> query shift for the next group
            nn.init.zeros_(self.state_out.weight); nn.init.zeros_(self.state_out.bias)
        else:
            raise ValueError(mode)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.fc_weight = nn.Parameter(torch.randn(num_attrs, dim) * 0.02)
        self.fc_bias = nn.Parameter(torch.zeros(num_attrs))

    def read(self, q, kv):
        q = q + self.attn(self.q_norm(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.mlp_norm(q))

    def forward(self, tokens, n_prefix):
        kv = self.kv_norm(torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1))
        B = tokens.shape[0]
        if self.mode == "mamba":
            q = self.read(self.queries[self.order].unsqueeze(0).expand(B, -1, -1), kv)
            q = self.ssm(q)[:, self.inv_order]                      # back to attribute order
        else:
            state = tokens.new_zeros(B, self.queries.shape[1])
            outs = [None] * len(SEQ_GROUPS)
            for gi, g in enumerate(SEQ_GROUPS):
                qg = self.queries[g].unsqueeze(0).expand(B, -1, -1) + self.state_out(state)[:, None]
                qg = self.read(qg, kv)
                summary = qg.mean(1)
                gate = torch.sigmoid(self.state_gate(torch.cat([state, summary], -1)))
                state = gate * state + (1 - gate) * torch.tanh(self.state_in(summary))
                outs[gi] = qg
            q = torch.cat(outs, 1)[:, self.inv_order]
        q = self.drop(self.out_norm(q))
        return (q * self.fc_weight).sum(-1) + self.fc_bias
