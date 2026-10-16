"""Matrix-walkthrough figures of the query head, from figs/qh_mats.npz (qh_mats.py, real v25 H+, example image 30465).

../../../Upar-Challenge-Track-2/.venv/bin/python make_qh_mat_figs.py
qh_mat_in.png   the two input matrices: tokens after LayerNorm (99 x 1280) and the 40 queries after LayerNorm (40 x 1280)
qh_mat_qk.png   head 2: Q (40 x 160) x K^T (160 x 99) = scores (40 x 99) -> softmax -> weights (40 x 99)
qh_mat_av.png   head 2: weights (40 x 99) x V (99 x 160) = O (40 x 160); then features F (40 x 1280) * W (40 x 1280)
                summed per row -> 40 logits
Rows of the hat query and the column of the top-of-head patch are outlined. Colours: orange positive, blue negative.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path(__file__).parent / "figs"
INK, PAPER, MUTE, ORANGE, BLUE = "#14202E", "#F5F3EE", "#6B7684", "#B85A1E", "#2F7FB5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "text.color": INK, "figure.facecolor": PAPER,
                     "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
DIV = LinearSegmentedColormap.from_list("div", ["#1F5F8F", "#8FC1E3", "#FBFAF7", "#F0A868", "#9C4A15"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#FBFAF7", "#F0A868", "#9C4A15", "#4A2208"])
d = np.load(OUT / "qh_mats.npz")
attrs = list(d["attrs"]); K = attrs.index("Accessory-Hat"); HEAD = 1; P = 1 + 3     # row of patch 3 (CLS is row 0)


def mat(ax, m, title, dims, cmap=DIV, sym=True, row=None, col=None, vmax=None):
    m = m.astype(np.float32)
    v = vmax or np.percentile(np.abs(m), 99)
    ax.imshow(m, cmap=cmap, aspect="auto", vmin=-v if sym else 0, vmax=v, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTE); s.set_linewidth(1)
    if row is not None:
        ax.add_patch(Rectangle((-0.5, row - 0.5), m.shape[1], 1, fill=False, edgecolor=INK, lw=2.5))
    if col is not None:
        ax.add_patch(Rectangle((col - 0.5, -0.5), 1, m.shape[0], fill=False, edgecolor=INK, lw=2.5, ls="--"))
    ax.set_title(title, fontsize=21, color=INK, pad=10)
    ax.text(0.5, -0.06, dims, transform=ax.transAxes, ha="center", va="top", fontsize=19, color=MUTE)


def op(fig, x, y, s, size=34):
    fig.text(x, y, s, ha="center", va="center", fontsize=size, color=INK)


def strip(fig, rect, v, label, note, cmap=DIV, sym=True, vmax=None):
    """One row of a matrix, enlarged, with its label on the left and a note on the right."""
    ax = fig.add_axes(rect); v = np.asarray(v, np.float32)[None]
    m = vmax or np.percentile(np.abs(v), 99)
    ax.imshow(v, cmap=cmap, aspect="auto", vmin=-m if sym else 0, vmax=m, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(2)
    fig.text(rect[0] - 0.006, rect[1] + rect[3] / 2, label, ha="right", va="center", fontsize=17, color=INK)
    fig.text(rect[0], rect[1] - 0.02, note, ha="left", va="top", fontsize=17, color=ORANGE, fontweight="bold")
    return ax


vi = lambda x, k=1: f"{x:.{k}f}".replace(".", ",").replace("-", "−")  # noqa: E731

# ---- inputs ----
fig = plt.figure(figsize=(15, 5.6))
a1 = fig.add_axes([0.03, 0.14, 0.40, 0.72]); a2 = fig.add_axes([0.56, 0.36, 0.38, 0.50])
mat(a1, d["KV"], "Token của ảnh (KV)", "99 hàng: CLS + 98 mảnh  ×  1280 cột", row=P)
mat(a2, d["Qn"], "40 query (đã chuẩn hoá)", "40 hàng: thuộc tính  ×  1280 cột", row=K)
a1.text(1.01, 1 - (P + 0.5) / 99, "← mảnh đỉnh đầu", transform=a1.transAxes, ha="left", va="center", fontsize=17)
a2.text(1.01, 1 - (K + 0.5) / 40, "← mũ", transform=a2.transAxes, ha="left", va="center", fontsize=17)
fig.savefig(OUT / "qh_mat_in.png", dpi=100); plt.close(fig)

# ---- Q x K^T = S -> softmax -> A (head 2) ----
Qh, Kh, S, A = d["Qh"][HEAD], d["Kh"][HEAD], d["S"][HEAD], d["A"][HEAD]
fig = plt.figure(figsize=(15.5, 6.6))
y0, h = 0.36, 0.52
ax = fig.add_axes([0.02, y0, 0.15, h]); mat(ax, Qh, "Q", "40 × 160", row=K)
op(fig, 0.195, y0 + h / 2, "×")
ax = fig.add_axes([0.225, 0.22, 0.11, 0.66]); mat(ax, Kh.T, "Kᵀ", "160 × 99", col=P)
op(fig, 0.36, y0 + h / 2, "=")
ax = fig.add_axes([0.39, y0, 0.22, h]); mat(ax, S, "Điểm S", "40 × 99", row=K, col=P)
op(fig, 0.655, y0 + h / 2 + 0.04, "softmax", 20); op(fig, 0.655, y0 + h / 2 - 0.04, "theo hàng →", 20)
ax = fig.add_axes([0.73, y0, 0.21, h]); mat(ax, A, "Tỉ lệ A", "40 × 99", cmap=SEQ, sym=False, row=K, col=P, vmax=1.0)
strip(fig, [0.39, 0.08, 0.22, 0.06], S[K], "hàng mũ", f"đỉnh đầu: {vi(float(S[K, P]))}")
strip(fig, [0.73, 0.08, 0.21, 0.06], A[K], "", f"đỉnh đầu: {vi(float(A[K, P]) * 100)}%, cả hàng = 1", cmap=SEQ, sym=False, vmax=1.0)
fig.savefig(OUT / "qh_mat_qk.png", dpi=100); plt.close(fig)
print("hat row: S[hat, top patch]", round(float(S[K, P]), 2), "A", round(float(A[K, P]), 4))

# ---- A x V = O ; F * W -> logits ----
Vh, Oh = d["Vh"][HEAD], d["O"][:, HEAD * 160:(HEAD + 1) * 160]
F, W, logits = d["F"], d["W"], d["logits"]
fig = plt.figure(figsize=(15.5, 8.0))
yt, ht = 0.62, 0.28
ax = fig.add_axes([0.02, yt, 0.16, ht]); mat(ax, A, "Tỉ lệ A", "40 × 99", cmap=SEQ, sym=False, row=K, col=P, vmax=1.0)
op(fig, 0.205, yt + ht / 2, "×")
ax = fig.add_axes([0.235, yt - 0.04, 0.11, ht + 0.08]); mat(ax, Vh, "V", "99 × 160", row=P)
op(fig, 0.37, yt + ht / 2, "=")
ax = fig.add_axes([0.395, yt, 0.17, ht]); mat(ax, Oh, "O của head 2", "40 × 160", row=K)
fig.text(0.6, yt + ht / 2 + 0.05, "ghép 8 head, cộng query,", fontsize=19, va="center")
fig.text(0.6, yt + ht / 2 - 0.01, "MLP, chuẩn hoá", fontsize=19, va="center")
fig.text(0.6, yt + ht / 2 - 0.07, "→ đặc trưng F bên dưới", fontsize=19, va="center")
yb, hb = 0.24, 0.24
ax = fig.add_axes([0.02, yb, 0.29, hb]); mat(ax, F, "Đặc trưng F", "40 × 1280", row=K)
op(fig, 0.335, yb + hb / 2, "⊙")
ax = fig.add_axes([0.36, yb, 0.29, hb]); mat(ax, W, "Hướng phân loại W", "40 × 1280", row=K)
fig.text(0.665, yb + hb / 2 + 0.03, "cộng theo hàng", fontsize=18, va="center")
fig.text(0.665, yb + hb / 2 - 0.03, "+ hệ số chệch →", fontsize=18, va="center")
ax = fig.add_axes([0.80, yb, 0.022, hb]); mat(ax, logits[:, None], "40 logit", "", row=K, vmax=6)
fig.text(0.83, yb + hb * (1 - (K + 0.5) / 40), "← mũ", fontsize=18, va="center")
prod = (F[K].astype(np.float32) * W[K].astype(np.float32))
strip(fig, [0.17, 0.07, 0.48, 0.05], prod, "hàng mũ của F ⊙ W", f"1280 tích, cộng lại {vi(float(prod.sum()), 2)}, cộng hệ số chệch {vi(float(d['bias'][K]), 2)}"
      f" = logit {vi(float(logits[K]), 2)} → {vi(100 / (1 + np.exp(-float(logits[K]))))}% có mũ")
fig.savefig(OUT / "qh_mat_av.png", dpi=100); plt.close(fig)
print("O hat row = A[hat] @ V:", float(np.abs(A[K].astype(np.float32) @ Vh.astype(np.float32) - Oh[K].astype(np.float32)).max()), "prod sum", float(prod.sum()))
