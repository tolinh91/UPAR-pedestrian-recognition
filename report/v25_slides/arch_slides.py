"""Architecture slides of the T1 deck (v29; v25 version kept in arch_slides_v25.py) in the style of the DeepSeek-V3/V4 block figures: rounded modules, token
stacks drawn as rows of circles, residual adds, dashed repeat containers and dashed training-only loss arrows.

python arch_slides.py SLIDES_DIR [--preview PNG_DIR]
Writes arch.html (whole model), block.html (one transformer block), qharch.html (query head), textarch.html (text
branch) and the two design slides qhdesign.html / textdesign.html (compact diagram with numbered choices + reasons) in the Slides-type section format. Boxes and labels are pinned <div>/<p> elements so that they use the deck
fonts; shapes and arrows are one inline <svg> per slide (no text inside it). All sizes are the real ones of the v25
members (printed from the built models: DINOv3 ViT-H+/16 and CLIP ViT-L/14 at 224x112, query head n_layers 1, 8 heads).
--preview wraps each section in a 1920x1080 page and screenshots it with headless Firefox (layout check only).
"""
import math
import subprocess
import sys
from pathlib import Path

INK, LINE, MUTE = "#14202E", "#2B3A4E", "#6B7684"
YEL, GREY, WHITE, PAPER = "#F8E6BC", "#E1E6EB", "#FFFFFF", "#F5F3EE"
BLUE, DBLUE, ORANGE, DORANGE, LORANGE = "#2F7FB5", "#23679A", "#B85A1E", "#9C4A15", "#F0A868"
SX, SY, SW, SH = 128, 256, 1664, 672          # the svg covers this part of the 1920x1080 canvas
PHOTO, HAT_ATTN = "/_blob/ffb0b66c4c3254ffa22a81858932eac7", "/_blob/59ec47e0dff6ca0eef30fe6a98f18e7f"
TOTAL = 32
QCOLS = ["#F0A868", "#8FC1E3", "#C9B6E4", "#9FD3B5", "#F2C4CE"]   # five of the 40 attribute queries


class Slide:
    def __init__(self, sid, part, part_name, page, title, notes):
        self.sid, self.part, self.part_name, self.page, self.title, self.notes = sid, part, part_name, page, title, notes
        self.svg, self.el = [], []

    # ---- svg shapes (canvas coordinates) ----
    def _p(self, x, y):
        return f"{x - SX:.1f} {y - SY:.1f}"

    def stack(self, cx, yb, n=5, layers=3, fills=None):
        """Token stack: a row of n circles; extra layers drawn up-left behind it (several tokens / vectors).
        Returns the visual bounding box (x0, y0, x1, y1)."""
        w, h = n * 22 + 8, 26
        x0, y0 = cx - w / 2, yb - h
        for L in range(layers - 1, -1, -1):
            rx, ry = x0 - 5 * L, y0 - 5 * L
            self.svg.append(f'<rect x="{rx - SX:.1f}" y="{ry - SY:.1f}" width="{w}" height="{h}" rx="8" fill="{WHITE}" '
                            f'stroke="{LINE}" stroke-width="2"/>')
            for i in range(n):
                f = fills[i] if fills and L == 0 and fills[i] else WHITE
                self.svg.append(f'<circle cx="{rx + 15 + 22 * i - SX:.1f}" cy="{ry + 13 - SY:.1f}" r="8" fill="{f}" '
                                f'stroke="{LINE}" stroke-width="1.8"/>')
        return x0 - 5 * (layers - 1), y0 - 5 * (layers - 1), x0 + w, yb

    def plus(self, cx, cy, r=16):
        self.svg.append(f'<circle cx="{cx - SX}" cy="{cy - SY}" r="{r}" fill="{WHITE}" stroke="{LINE}" stroke-width="2.5"/>')
        self.svg.append(f'<path d="M {self._p(cx - r, cy)} L {self._p(cx + r, cy)} M {self._p(cx, cy - r)} '
                        f'L {self._p(cx, cy + r)}" stroke="{LINE}" stroke-width="2.5" fill="none"/>')

    def arrow(self, pts, dashed=False, head=True, color=LINE, sw=3):
        pts = [list(p) for p in pts]
        if head:  # the marker tip sits 4 stroke widths past the path end: stop short so the tip touches the target
            (xa, ya), (xb, yb) = pts[-2], pts[-1]
            L = math.hypot(xb - xa, yb - ya); k = (L - 4 * sw) / L
            pts[-1] = [xa + (xb - xa) * k, ya + (yb - ya) * k]
        d = "M " + " L ".join(self._p(x, y) for x, y in pts)
        extra = ' stroke-dasharray="10 7"' if dashed else ""
        extra += ' marker-end="url(#h)"' if head else ""
        self.svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{extra}/>')

    def frame(self, x0, y0, x1, y1, color):
        self.svg.append(f'<rect x="{x0 - SX}" y="{y0 - SY}" width="{x1 - x0}" height="{y1 - y0}" rx="10" fill="none" '
                        f'stroke="{color}" stroke-width="2.5" stroke-dasharray="9 7"/>')

    # ---- html elements (deck fonts) ----
    def box(self, x0, y0, x1, y1, text, sub=None, kind="main", lock=False):
        fill = {"main": YEL, "op": GREY, "input": WHITE}[kind]
        pad = ";padding-left:40px" if lock else ""
        body = text + (f'<br><span style="color:{MUTE}">{sub}</span>' if sub else "")
        self.el.append(f'<div style="position:absolute;left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px;'
                       f'background:{fill};border:2px solid {LINE};border-radius:12px;display:flex;flex-direction:column;'
                       f'justify-content:center;align-items:center{pad}"><p style="font-size:24px;line-height:1.25;'
                       f'text-align:center;color:{INK}">{body}</p></div>')
        if lock:
            self.el.append(f'<x-icon name="Lock" style="position:absolute;left:{x0 + 12}px;top:{(y0 + y1) // 2 - 13}px;'
                           f'width:26px;height:26px;color:{MUTE}"></x-icon>')

    def rect(self, x0, y0, x1, y1, fill, rx=8):
        self.svg.append(f'<rect x="{x0 - SX}" y="{y0 - SY}" width="{x1 - x0}" height="{y1 - y0}" rx="{rx}" fill="{fill}" '
                        f'stroke="{LINE}" stroke-width="2"/>')

    def dot(self, cx, cy, r, fill):
        self.svg.append(f'<circle cx="{cx - SX}" cy="{cy - SY}" r="{r}" fill="{fill}" stroke="{LINE}" stroke-width="2"/>')

    def cross(self, x0, y0, x1, y1):
        self.svg.append(f'<path d="M {self._p(x0, y0)} L {self._p(x1, y1)} M {self._p(x0, y1)} L {self._p(x1, y0)}" '
                        f'stroke="{ORANGE}" stroke-width="4" stroke-linecap="round" fill="none"/>')

    def badge(self, x0, y0, n):
        self.el.append(f'<p style="position:absolute;left:{x0}px;top:{y0}px;width:40px;height:40px;border-radius:50%;'
                       f'background:{ORANGE};color:{WHITE};font-size:24px;font-weight:700;line-height:40px;text-align:center">'
                       f'{n}</p>')

    def cards(self, x0, y0, w, items):
        """Numbered design choices: bold choice + one or two lines of reason, stacked in one pinned column."""
        rows = "".join(
            f'<div style="background:{WHITE};border:1px solid #DCE1E6;border-radius:14px;padding:10px 16px;display:flex;'
            f'flex-direction:row;gap:16px;align-items:start"><p style="flex:none;width:40px;height:40px;border-radius:50%;'
            f'background:{ORANGE};color:{WHITE};font-size:24px;font-weight:700;line-height:40px;text-align:center">{i}</p>'
            f'<div style="flex:1;display:flex;flex-direction:column;gap:2px"><p style="font-size:24px;font-weight:700;'
            f'line-height:1.3;color:{INK}">{t}</p><p style="font-size:24px;line-height:1.3;color:#3B4756">{r}</p></div></div>'
            for i, (t, r) in enumerate(items, 1))
        self.el.append(f'<div style="position:absolute;left:{x0}px;top:{y0}px;width:{w}px;display:flex;flex-direction:column;'
                       f'gap:10px">{rows}</div>')

    def label(self, x0, y0, w, text, align="left", color=INK, bold=False, italic=False, mono=False):
        st = (f"position:absolute;left:{x0}px;top:{y0}px;width:{w}px;font-size:24px;line-height:1.25;text-align:{align};"
              f"color:{color}")
        st += ";font-weight:700" if bold else ""
        st += ";font-style:italic" if italic else ""
        st += ";font-family:'JetBrains Mono', monospace" if mono else ""
        self.el.append(f'<p style="{st}">{text}</p>')

    def image(self, x0, y0, w, h, src, alt):
        self.el.append(f'<img src="{src}" alt="{alt}" style="position:absolute;left:{x0}px;top:{y0}px;width:{w}px;'
                       f'height:{h}px;object-fit:cover;border-radius:6px">')

    def legend(self, items):
        """Up to four keys in two columns right of the title: main, op, lock, dash."""
        slots = [(1110, 146), (1110, 192), (1460, 146), (1460, 192)]
        for (x, y), key in zip(slots, items):
            if key in ("main", "op"):
                self.el.append(f'<div style="position:absolute;left:{x}px;top:{y + 3}px;width:40px;height:26px;'
                               f'background:{YEL if key == "main" else GREY};border:2px solid {LINE};border-radius:6px"></div>')
            elif key == "lock":
                self.el.append(f'<x-icon name="Lock" style="position:absolute;left:{x + 7}px;top:{y + 2}px;width:26px;'
                               f'height:26px;color:{MUTE}"></x-icon>')
            else:
                self.el.append(f'<svg aria-label="mũi tên nét đứt" width="44" height="20" viewBox="0 0 44 20" '
                               f'style="position:absolute;left:{x}px;top:{y + 6}px;width:44px;height:20px">'
                               f'<path d="M 2 10 L 30 10" stroke="{LINE}" stroke-width="3" stroke-dasharray="7 5" fill="none"/>'
                               f'<path d="M 30 3 L 42 10 L 30 17 Z" fill="{LINE}"/></svg>')
            text = {"main": "Khối chính", "op": "Phép tính phụ", "lock": "Giữ cố định", "dash": "Chỉ khi huấn luyện"}[key]
            self.label(x + 54, y, 290, text, color="#3B4756")

    def html(self):
        svg = (f'<svg aria-label="{self.title}" width="{SW}" height="{SH}" viewBox="0 0 {SW} {SH}" '
               f'style="position:absolute;left:{SX}px;top:{SY}px;width:{SW}px;height:{SH}px"><defs>'
               f'<marker id="h" orient="auto" markerWidth="4" markerHeight="4" refX="0" refY="2" overflow="visible">'
               f'<path d="M0 0 L4 2 L0 4 Z" fill="{LINE}" stroke="none"/></marker></defs>' + "".join(self.svg) + "</svg>")
        return "\n".join([
            f'<section id="{self.sid}" data-transition="fade" style="background:{PAPER};color:{INK};font-family:\'Be Vietnam Pro\', '
            f'Arial, sans-serif;padding:128px 128px 160px;display:flex;flex-direction:column;gap:20px">',
            f'<p style="font-size:24px;letter-spacing:3px;text-transform:uppercase;font-weight:600;color:{DORANGE}">'
            f'Phần {self.part} · {self.part_name}</p>',
            f'<h2 style="font-size:60px;font-weight:700;line-height:1.1;color:{INK}">{self.title}</h2>',
            svg, *self.el,
            f'<p style="position:absolute;left:128px;bottom:64px;width:900px;font-size:24px;color:{MUTE}">Kiến trúc T1 v29 · '
            f'Phần {self.part}</p>',
            f'<p style="position:absolute;right:128px;bottom:64px;width:160px;text-align:right;font-size:24px;color:{MUTE}">'
            f'{self.page:02d} / {TOTAL}</p>',
            f"<aside>{self.notes}</aside>", "</section>", ""])


def arch():
    s = Slide("arch", 2, "Kiến trúc tổng thể", 9, "Kiến trúc tổng thể của v29", (
        "Sơ đồ đọc từ dưới lên. Ảnh 224×112 đi vào hai thành viên độc lập. Ở mỗi thành viên, patch embedding chia ảnh "
        "thành các ô vuông và biến mỗi ô thành một token. H+ dùng ô 16×16 pixel, tạo 98 patch token, cộng thêm một "
        "token CLS và bốn token register. CLIP dùng ô 14×14 pixel, tạo 128 patch token và một token CLS. Các token đi "
        "qua chồng Transformer block, 32 block ở H+ và 24 block ở CLIP. Khi fine-tune, patch embedding và 8 block đầu "
        "được giữ cố định, các block còn lại được cập nhật. Query head nhận token CLS và các patch token, bỏ bốn token "
        "register, và cho ra 40 logit. Riêng CLIP có thêm nhánh văn bản, chỉ đọc token CLS. Logit của nhánh văn bản "
        "được cộng vào logit của query head. Hai thành viên được huấn luyện riêng, mỗi thành viên có hàm mất mát riêng, "
        "vẽ bằng mũi tên nét đứt. Hàm mất mát là BCE có điều chỉnh theo tần suất nhãn. Khi huấn luyện, logit được dịch "
        "theo tần suất của từng thuộc tính, nhờ đó khi suy luận ngưỡng 0.5 cho quyết định cân bằng mà không cần bước "
        "hậu xử lý nào. Khi suy luận, "
        "hai vector 40 logit được lấy trung bình và đi qua hàm sigmoid. Thuộc tính có xác suất lớn hơn 0.5 được dự "
        "đoán là có."))
    xH, xC, xT = 560, 1120, 1520
    H0, H1, C0, C1, T0, T1 = xH - 190, xH + 190, xC - 190, xC + 190, xT - 130, xT + 130
    s.legend(["main", "op", "lock", "dash"])
    # fusion row and output
    s.box(H0, 264, C1, 312, "Trung bình hai vector 40 logit", kind="op")
    s.box(1370, 264, 1540, 312, "Sigmoid", kind="op")
    s.box(1600, 264, 1792, 312, "Có nếu > 0.5", kind="op")
    s.arrow([(C1, 288), (1370, 288)]); s.arrow([(1540, 288), (1600, 288)])
    # per member: logits, head, tokens, blocks, tokens, patch embedding
    for x, c0, c1, col, dcol, name, sub, blk, patch, s1, s2 in (
            (xH, H0, H1, BLUE, DBLUE, "DINOv3 ViT-H+", ["32 block", "1280 chiều"], "Block 9–32 · fine-tune",
             "Patch Embedding 16×16", ["CLS + 4 register", "+ 98 patch"], "CLS + 98 patch"),
            (xC, C0, C1, ORANGE, DORANGE, "CLIP ViT-L/14", ["24 block", "1024 chiều"],
             "Block 9–24 · fine-tune", "Patch Embedding 14×14", ["CLS + 128 patch"], "CLS + 128 patch")):
        z = s.stack(x, 372, fills=[LORANGE if x == xC else "#8FC1E3"] * 5)
        s.arrow([(x, z[1]), (x, 312)])
        s.box(c0, 460, c1, 508, "Query Head · 40 truy vấn")
        t2 = s.stack(x, 568, fills=[col] + [None] * 4)
        s.arrow([(x, t2[1]), (x, 508)])
        s.frame(c0 - 20, 594, c1 + 20, 736, col)
        s.box(c0, 606, c1, 654, blk)
        s.box(c0, 676, c1, 724, "Block 1–8 · cố định", lock=True)
        s.arrow([(x, 676), (x, 654)]); s.arrow([(x, 606), (x, 568)])
        t1 = s.stack(x, 794, fills=[col] + [None] * 4)
        s.arrow([(x, t1[1]), (x, 724)])
        s.box(c0, 820, c1, 868, patch, lock=True)
        s.arrow([(x, 820), (x, 794)])
        if x == xH:
            s.arrow([(x, 460), (x, 372)])
            s.label(128, 612, 212, name, align="right", color=dcol, bold=True)
            for i, t in enumerate(sub):
                s.label(128, 646 + 32 * i, 212, t, align="right", color="#3B4756")
            s.label(140, 538, t2[0] - 8 - 140, s2, align="right", color="#3B4756")
            s.label(140, 746, t1[0] - 8 - 140, s1[0] + "<br>" + s1[1], align="right", color="#3B4756")
            s.label(z[2] + 12, 344, 200, "40 logit", color="#3B4756")
            s.arrow([(z[0] - 4, 359), (392, 359)], dashed=True)
            s.label(140, 344, 240, "Loss điều chỉnh", align="right", italic=True)
        else:
            s.plus(x, 420)
            s.arrow([(x, 460), (x, 436)]); s.arrow([(x, 404), (x, 372)])
            s.label(1350, 612, 442, name, color=dcol, bold=True)
            for i, t in enumerate(sub):
                s.label(1350, 646 + 32 * i, 442, t, color="#3B4756")
            s.label(760, 538, t2[0] - 8 - 760, s2, align="right", color="#3B4756")
            s.label(t1[2] + 14, 762, 300, s1[0], color="#3B4756")
            s.label(z[0] - 8 - 200, 344, 200, "40 logit", align="right", color="#3B4756")
            s.arrow([(z[2] + 4, 359), (1300, 359)], dashed=True)
            s.label(1312, 344, 240, "Loss điều chỉnh", italic=True)
            # text branch: reads only the CLS token, its 40 logits are added to the query head's
            s.box(T0, 460, T1, 508, "Nhánh văn bản")
            s.arrow([(t2[2] + 2, 552), (xT, 552), (xT, 508)])
            s.label(1186, 512, 196, "chỉ token CLS", color=DORANGE)
            s.arrow([(xT, 460), (xT, 420), (x + 16, 420)])
            s.label(1170, 384, 300, "40 logit văn bản", color="#3B4756")
    # the input image between the two patch embeddings
    s.image(818, 800, 44, 88, PHOTO, "Ảnh đầu vào 224×112")
    s.arrow([(818, 844), (H1, 844)]); s.arrow([(862, 844), (C0, 844)])
    s.label(740, 892, 200, "Ảnh 224×112", align="center", color="#3B4756")
    return s


def block():
    s = Slide("block", 3, "Hai thành viên và query head", 14, "Một Transformer block", (
        "Hai backbone được ghép từ cùng một loại block, xếp chồng 32 lần ở H+ và 24 lần ở CLIP. Đường thẳng bên trái là "
        "dòng residual (tạm dịch: đường tắt), mang token đi thẳng từ dưới lên. Mỗi block có hai lớp con. Ở lớp con thứ "
        "nhất, token được chuẩn hoá bằng LayerNorm rồi đưa vào multi-head self-attention, nơi mỗi token tổng hợp thông "
        "tin từ mọi token khác của ảnh. Kết quả được cộng trở lại dòng residual tại dấu cộng. Ở lớp con thứ hai, token "
        "được chuẩn hoá lần nữa và đi qua MLP, một mạng hai lớp xử lý riêng từng token, rồi lại được cộng vào dòng "
        "residual. Ở H+, đầu ra của mỗi lớp con được nhân với một vector LayerScale học được trước khi cộng, CLIP không "
        "có bước này. Bảng bên phải liệt kê các điểm khác nhau giữa hai backbone. Khác biệt lớn nhất là cách mã hoá vị "
        "trí. H+ xoay vector query và key theo vị trí của patch ngay trong attention (RoPE), còn CLIP cộng một bảng vị "
        "trí học được vào token một lần ở đầu vào."))
    xr, xm, xs = 350, 600, 880
    M0, M1, S0, S1 = xm - 110, xm + 110, xs - 150, xs + 150
    s.legend(["main", "op"])
    s.frame(272, 456, 1046, 916, BLUE)
    s.label(670, 418, 376, "×32 ở H+ · ×24 ở CLIP", align="right", color=DBLUE, bold=True)
    tin = s.stack(xr, 900)
    s.label(128, 872, 272 - 8 - 128, "Token vào", align="right", color="#3B4756")
    # sub-layer 1: LayerNorm -> self-attention -> LayerScale -> add
    s.box(M0, 863, M1, 911, "LayerNorm", kind="op")
    s.arrow([(tin[2] + 2, 887), (M0, 887)])
    n1 = s.stack(xs, 900)
    s.arrow([(M1, 887), (n1[0] - 2, 887)])
    s.box(S0, 760, S1, 836, "Multi-Head<br>Self-Attention")
    s.arrow([(xs, n1[1]), (xs, 836)])
    a1 = s.stack(xs, 734)
    s.arrow([(xs, 760), (xs, 734)])
    s.box(M0, 697, M1, 745, "LayerScale (H+)", kind="op")
    s.arrow([(a1[0] - 2, 721), (M1, 721)])
    s.plus(xr, 721)
    s.arrow([(M0, 721), (xr + 16, 721)])
    s.arrow([(xr, tin[1]), (xr, 737)])
    # sub-layer 2: LayerNorm -> MLP -> LayerScale -> add
    m = s.stack(xr, 660)
    s.arrow([(xr, 705), (xr, 660)])
    s.box(M0, 623, M1, 671, "LayerNorm", kind="op")
    s.arrow([(m[2] + 2, 647), (M0, 647)])
    n2 = s.stack(xs, 660)
    s.arrow([(M1, 647), (n2[0] - 2, 647)])
    s.box(S0, 540, S1, 596, "MLP")
    s.arrow([(xs, n2[1]), (xs, 596)])
    a2 = s.stack(xs, 510)
    s.arrow([(xs, 540), (xs, 510)])
    s.box(M0, 473, M1, 521, "LayerScale (H+)", kind="op")
    s.arrow([(a2[0] - 2, 497), (M1, 497)])
    s.plus(xr, 497)
    s.arrow([(M0, 497), (xr + 16, 497)])
    s.arrow([(xr, m[1]), (xr, 513)])
    out = s.stack(xr, 440)
    s.arrow([(xr, 481), (xr, 440)])
    s.arrow([(xr, out[1]), (xr, 300)])
    s.label(xr + 14, 288, 360, "Sang block tiếp theo", color="#3B4756")
    # the two backbones side by side
    rows = [("", "DINOv3 ViT-H+", "CLIP ViT-L/14"), ("Số block", "32", "24"), ("Số chiều token", "1280", "1024"),
            ("Số head attention", "20", "16"), ("Mã hoá vị trí", "RoPE, trong attention", "Bảng vị trí, cộng ở đầu vào"),
            ("MLP", "SwiGLU, 5120 chiều ẩn", "GELU, 4096 chiều ẩn"), ("LayerScale", "Có", "Không"),
            ("Token đặc biệt", "CLS + 4 register", "CLS"), ("Tham số", "840 triệu", "304 triệu")]
    cells = []
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        w = ['style="width:34%"', 'style="width:33%"', 'style="width:33%"'] if i == 0 else ["", "", ""]
        cells.append("<tr>" + "".join(f"<{tag} {w[j]}>{v}</{tag}>".replace(" >", ">") for j, v in enumerate(r)) + "</tr>")
    s.el.append(f'<div style="position:absolute;left:1070px;top:300px;width:722px;background:{WHITE};border:1px solid #DCE1E6;'
                f'border-radius:16px;padding:20px 24px"><table style="font-size:24px;color:{INK}">' + "".join(cells) +
                "</table></div>")
    return s


def qharch():
    s = Slide("qharch", 3, "Hai thành viên và query head", 17, "Cấu trúc của query head", (
        "Sơ đồ đọc từ dưới lên. Bên trái là 40 truy vấn, mỗi truy vấn là một vector học được dành cho một thuộc tính, "
        "năm vòng tròn màu tượng trưng cho năm trong 40 truy vấn. Bên phải là token của backbone, gồm token CLS và các "
        "patch token. Trong multi-head cross-attention, mỗi truy vấn đóng vai trò query, còn token của ảnh đóng vai trò "
        "key và value. Mỗi truy vấn chấm điểm từng token, chuẩn hoá điểm thành trọng số rồi lấy tổng có trọng số của "
        "các token. Ảnh nhỏ bên phải là trọng số thật của truy vấn mũ trên ảnh ví dụ. Kết quả được cộng vào truy vấn ban "
        "đầu, đi qua một MLP có lớp ẩn rộng gấp đôi, và được cộng thêm một lần nữa. Cuối cùng, mỗi truy vấn được nhân "
        "với vector phân loại riêng của thuộc tính tương ứng, cộng một hệ số chệch, để cho ra một logit. Toàn bộ head "
        "chỉ có một lớp như vậy, với 8 head attention, khoảng 13 triệu tham số ở H+ và 8,5 triệu tham số ở CLIP. Khi "
        "huấn luyện, 40 logit được so với nhãn bằng hàm mất mát BCE."))
    xq, xm, xs, xt = 380, 680, 1020, 1440
    M0, M1, S0, S1, T0, T1 = xm - 120, xm + 120, xs - 200, xs + 200, xt - 140, xt + 140
    s.legend(["main", "op", "dash"])
    qin = s.stack(xq, 900, fills=QCOLS)
    s.label(128, 856, qin[0] - 10 - 128, "40 truy vấn<br>học được", align="right", color="#3B4756")
    tok = s.stack(xt, 900, fills=[ORANGE] + [None] * 4)
    s.label(tok[2] + 14, 856, 1792 - tok[2] - 14, "CLS + patch<br>từ backbone", color="#3B4756")
    # cross-attention: queries give Q, image tokens give K and V
    s.box(M0, 863, M1, 911, "LayerNorm", kind="op")
    s.arrow([(qin[2] + 2, 887), (M0, 887)])
    qn = s.stack(xs, 900, fills=QCOLS)
    s.arrow([(M1, 887), (qn[0] - 2, 887)])
    s.box(T0, 780, T1, 828, "LayerNorm", kind="op")
    s.arrow([(xt, tok[1]), (xt, 828)])
    s.box(S0, 690, S1, 766, "Multi-Head Cross-Attention", "8 head")
    s.arrow([(xs, qn[1]), (xs, 766)])
    s.label(xs + 12, 806, 60, "Q", bold=True, color=DORANGE)
    s.arrow([(xt, 780), (xt, 728), (S1, 728)])
    s.label(1256, 690, 120, "K, V", bold=True, color=DORANGE)
    s.image(1640, 600, 60, 120, HAT_ATTN, "Trọng số attention thật của truy vấn mũ")
    s.label(1600, 730, 192, "Attention thật<br>của truy vấn mũ", align="center", color="#3B4756")
    a = s.stack(xs, 660, fills=QCOLS)
    s.arrow([(xs, 690), (xs, 660)])
    s.plus(xq, 647)
    s.arrow([(a[0] - 2, 647), (xq + 16, 647)])
    s.arrow([(xq, qin[1]), (xq, 663)])
    # MLP sub-layer
    q1 = s.stack(xq, 596, fills=QCOLS)
    s.arrow([(xq, 631), (xq, 596)])
    s.box(M0, 559, M1, 607, "LayerNorm", kind="op")
    s.arrow([(q1[2] + 2, 583), (M0, 583)])
    s.box(S0 + 20, 559, S1 - 20, 607, "MLP, lớp ẩn 2× chiều")
    s.arrow([(M1, 583), (S0 + 20, 583)])
    b = s.stack(xs, 520, fills=QCOLS)
    s.arrow([(xs, 559), (xs, 520)])
    s.plus(xq, 507)
    s.arrow([(b[0] - 2, 507), (xq + 16, 507)])
    s.arrow([(xq, q1[1]), (xq, 523)])
    # one classifier per attribute
    q2 = s.stack(xq, 456, fills=QCOLS)
    s.arrow([(xq, 491), (xq, 456)])
    s.box(M0, 419, M1, 467, "LayerNorm", kind="op")
    s.arrow([(q2[2] + 2, 443), (M0, 443)])
    s.box(S0, 405, S1, 481, "40 bộ phân loại riêng", "mỗi thuộc tính một vector")
    s.arrow([(M1, 443), (S0, 443)])
    z = s.stack(xs, 372, fills=["#8FC1E3"] * 5)
    s.arrow([(xs, 405), (xs, 372)])
    s.label(z[0] - 10 - 200, 344, 200, "40 logit", align="right", color="#3B4756")
    s.arrow([(z[2] + 4, 359), (1250, 359)], dashed=True)
    s.label(1262, 344, 240, "BCE loss", italic=True)
    return s


def textarch():
    s = Slide("textarch", 4, "Bộ phân loại khởi tạo từ câu mô tả", 22, "Cấu trúc của nhánh văn bản", (
        "Sơ đồ đọc từ dưới lên, gồm hai đường gặp nhau ở phép tích vô hướng. Đường bên trái xử lý ảnh. Token CLS 1024 "
        "chiều đi qua phép chiếu có sẵn của CLIP, là phép chiếu CLIP dùng để đưa ảnh vào không gian chung với câu chữ, "
        "rồi được chuẩn hoá về độ dài 1. Đường bên phải xử lý câu chữ và chỉ chạy một lần trước khi huấn luyện. Mỗi "
        "thuộc tính có một câu mô tả, ghép vào 4 mẫu câu, ví dụ “a blurry photo of a woman.” cho thuộc tính nữ. Text "
        "encoder của CLIP biến mỗi câu thành vector 768 chiều. Lấy trung bình 4 mẫu câu và trừ đi vector của câu trung "
        "tính “a person”, ta được hướng gốc của thuộc tính. Hướng gốc được giữ cố định. Trong lúc huấn luyện chỉ có một "
        "phần chỉnh nhỏ được học, nhân 0.1 trước khi cộng vào hướng gốc, nên hướng phân loại chỉ xoay trung bình 4,7 độ. "
        "Tích vô hướng giữa vector ảnh và 40 hướng phân loại, nhân 100, cộng hệ số chệch, cho ra 40 logit văn bản. Hệ số "
        "chệch được khởi tạo từ tần suất của thuộc tính trong dữ liệu huấn luyện. Phần học thêm của nhánh này chỉ có "
        "khoảng 31 nghìn tham số."))
    xi, xc, xt = 560, 950, 1340
    I0, I1, T0, T1 = xi - 200, xi + 200, xt - 220, xt + 220
    s.legend(["main", "op", "lock"])
    # image side
    cls = s.stack(xi, 900, layers=1, fills=[ORANGE] + [None] * 4)
    s.label(128, 856, cls[0] - 12 - 128, "Token CLS<br>1024 chiều", align="right", color="#3B4756")
    s.box(I0, 770, I1, 846, "Phép chiếu của CLIP", "1024 → 768 chiều", lock=True)
    s.arrow([(xi, cls[1]), (xi, 846)])
    s.box(I0, 690, I1, 738, "Chuẩn hoá độ dài", kind="op")
    s.arrow([(xi, 770), (xi, 738)])
    e = s.stack(xi, 660, layers=1, fills=[LORANGE] * 5)
    s.arrow([(xi, 690), (xi, 660)])
    s.label(128, 618, e[0] - 12 - 128, "Vector ảnh<br>768 chiều", align="right", color="#3B4756")
    # text side (computed once before training)
    s.box(xt - 250, 846, xt + 250, 916, "40 câu mô tả × 4 mẫu câu", "“a blurry photo of a woman.”", kind="input")
    s.box(T0, 750, T1, 818, "CLIP Text Encoder", "chạy một lần trước huấn luyện", lock=True)
    s.arrow([(xt, 846), (xt, 818)])
    s.box(T0, 668, T1, 716, "Trung bình 4 mẫu câu", kind="op")
    s.arrow([(xt, 750), (xt, 716)])
    s.box(T0, 588, T1, 636, "Trừ vector câu “a person”", kind="op")
    s.arrow([(xt, 668), (xt, 636)])
    w0 = s.stack(xt, 560)
    s.arrow([(xt, 588), (xt, 560)])
    s.label(w0[2] + 14, 530, 1792 - w0[2] - 14, "Hướng gốc, cố định", color="#3B4756")
    s.plus(xt, 480)
    s.arrow([(xt, w0[1]), (xt, 496)])
    d = s.stack(1650, 493, fills=[LORANGE] * 5)
    s.box(1450, 456, 1530, 504, "× 0.1", kind="op")
    s.arrow([(d[0] - 2, 480), (1530, 480)]); s.arrow([(1450, 480), (xt + 16, 480)])
    s.label(1454, 412, 338, "Phần chỉnh, học được", color=DORANGE)
    w = s.stack(xt, 430)
    s.arrow([(xt, 464), (xt, 430)])
    s.label(960, 402, w[0] - 12 - 960, "40 hướng phân loại", align="right", color="#3B4756")
    # meeting point: dot product, scale, bias
    s.box(xc - 140, 326, xc + 140, 374, "Tích vô hướng × 100", kind="op")
    s.arrow([(xi, e[1]), (xi, 350), (xc - 140, 350)])
    s.arrow([(xt, w[1]), (xt, 350), (xc + 140, 350)])
    s.plus(xc, 288)
    s.arrow([(xc, 326), (xc, 304)])
    bb = s.stack(710, 301, fills=[LORANGE] * 5)
    s.arrow([(bb[2] + 2, 288), (xc - 16, 288)])
    s.label(128, 272, bb[0] - 12 - 128, "Hệ số chệch, khởi tạo từ tần suất", align="right", color=DORANGE)
    z = s.stack(1140, 301, fills=["#8FC1E3"] * 5)
    s.arrow([(xc + 16, 288), (z[0] - 2, 288)])
    s.label(z[2] + 14, 272, 360, "40 logit văn bản", color="#3B4756")
    return s


def qhdesign():
    s = Slide("qhdesign", 3, "Hai thành viên và query head", 18, "Thiết kế của query head", (
        "Slide tóm tắt năm lựa chọn thiết kế của query head, đánh số trên sơ đồ thu gọn bên trái. Lựa chọn thứ nhất là "
        "mỗi thuộc tính có một truy vấn riêng, thay vì 40 bộ phân loại dùng chung một vector của cả ảnh. Nhờ vậy truy "
        "vấn mũ chỉ cần đọc vùng đầu và truy vấn quần chỉ cần đọc nửa dưới. Trên bốn ảnh ví dụ, truy vấn mũ đặt 77% "
        "attention vào ba hàng patch trên cùng, trong khi cách lấy trung bình đều chỉ đặt 21%. Lựa chọn thứ hai là bỏ bốn "
        "token register của H+ khỏi tập token mà truy vấn đọc. Register là các token phụ mà DINOv3 dùng để chứa thông "
        "tin chung của cả ảnh, chúng không ứng với vị trí nào trong ảnh. Token CLS vẫn được giữ làm nguồn thông tin toàn "
        "ảnh. Lựa chọn thứ ba là chỉ dùng một lớp cross-attention với 8 head. Các truy vấn đọc token của backbone nhưng "
        "không sửa token đó và không trao đổi với nhau. Toàn bộ head thêm khoảng 13 triệu tham số, bằng 1,6% backbone H+. "
        "Lựa chọn thứ tư là mỗi truy vấn có vector phân loại của riêng thuộc tính đó, minh hoạ bằng năm ô màu tương ứng "
        "với năm truy vấn. Lựa chọn thứ năm thuộc về huấn luyện. Head được khởi tạo ngẫu nhiên nên dùng learning rate "
        "gấp 10 lần block cuối của backbone, còn backbone đã học trước chỉ được chỉnh nhẹ."))
    centers = [258, 354, 450, 546, 642]
    s.frame(196, 396, 880, 800, BLUE)
    s.label(712, 412, 160, "Query head", color=DBLUE, bold=True)
    for cx, col in zip(centers, QCOLS):           # one classifier and one logit per query
        s.dot(cx, 356, 14, col)
        s.rect(cx - 38, 474, cx + 38, 526, col)
        s.arrow([(cx, 474), (cx, 370)]); s.arrow([(cx, 600), (cx, 526)])
    s.label(670, 342, 130, "40 logit", color="#3B4756")
    s.box(220, 600, 680, 652, "MLP")
    s.box(220, 700, 680, 772, "Cross-Attention · 8 head")
    s.arrow([(450, 700), (450, 652)])
    q = s.stack(290, 880, fills=QCOLS)
    s.arrow([(290, q[1]), (290, 772)])
    s.label(210, 886, 160, "40 truy vấn", align="center", color="#3B4756")
    t = s.stack(600, 880, fills=[ORANGE] + [None] * 4)
    s.arrow([(600, t[1]), (600, 772)])
    s.label(510, 886, 180, "CLS + patch", align="center", color="#3B4756")
    r = s.stack(770, 880, n=4, layers=1, fills=["#C3CAD2"] * 4)
    s.cross(r[0] - 4, r[1] - 6, r[2] + 4, r[3] + 6)
    s.label(700, 886, 140, "4 register", align="center", color="#3B4756")
    s.label(302, 806, 40, "Q", bold=True, color=DORANGE)
    s.label(612, 806, 80, "K, V", bold=True, color=DORANGE)
    for x, y, n in ((150, 846, 1), (832, 846, 2), (692, 716, 3), (692, 480, 4), (836, 352, 5)):
        s.badge(x, y, n)
    s.cards(920, 270, 872, [
        ("Mỗi thuộc tính một truy vấn riêng", "Truy vấn mũ đặt 77% attention vào vùng đầu, lấy trung bình đều chỉ đặt 21%."),
        ("Bỏ 4 token register của H+", "Register giữ thông tin chung, không gắn với vùng ảnh nào. CLS giữ thông tin toàn ảnh."),
        ("Một lớp cross-attention, 8 head", "Đủ để mỗi truy vấn chọn vùng ảnh. Cả head chỉ thêm 13 triệu tham số, 1,6% của H+."),
        ("Bộ phân loại riêng cho từng thuộc tính", "Đặc trưng của mũ không dùng chung trọng số đầu ra với đặc trưng của quần."),
        ("Learning rate của head gấp 10 lần backbone", "Head khởi tạo ngẫu nhiên cần học nhanh, backbone đã học trước chỉ cần chỉnh nhẹ.")])
    return s


def textdesign():
    s = Slide("textdesign", 4, "Bộ phân loại khởi tạo từ câu mô tả", 23, "Thiết kế của nhánh văn bản", (
        "Slide tóm tắt năm lựa chọn thiết kế của nhánh văn bản, đánh số trên sơ đồ thu gọn bên trái. Lựa chọn thứ nhất là "
        "chỉ dùng token CLS và phép chiếu gốc của CLIP. Tích của hai thành phần này chính là vector ảnh mà CLIP dùng khi "
        "học ghép ảnh với câu trên 400 triệu cặp, nên so sánh trực tiếp được với vector câu. Phép chiếu và text encoder "
        "được giữ cố định để hai vector luôn nằm trong cùng không gian đó. Lựa chọn thứ hai là cách viết câu. Mỗi thuộc "
        "tính được ghép vào bốn mẫu câu gần với ảnh camera giám sát, như “a low resolution surveillance photo of …” và "
        "“a blurry photo of …”, rồi lấy trung bình. Vector của câu trung tính “a person” được trừ đi để bỏ phần chung "
        "của mọi ảnh người. Lựa chọn thứ ba là phần chỉnh học được được nhân 0.1 trước khi cộng vào hướng gốc. Vì vậy "
        "hướng phân loại học chậm hơn khoảng 10 lần so với phần còn lại của head, và sau huấn luyện chỉ xoay trung bình "
        "4,7 độ. Thuộc tính hiếm, vốn có ít ảnh để học, vẫn giữ nghĩa của câu mô tả. Lựa chọn thứ tư là hệ số 100 và hệ "
        "số chệch. Hệ số 100 là nhiệt độ mà CLIP học được khi huấn luyện trước, giữ logit ở đúng thang. Hệ số chệch "
        "được khởi tạo từ tần suất của thuộc tính, nên thuộc tính hiếm xuất phát ở mức nghiêng về “không”. Lựa chọn thứ "
        "năm là cách gộp. Logit văn bản được cộng vào logit của query head và cả hai được học bằng cùng một hàm mất mát. "
        "Nhánh văn bản cho điểm xuất phát theo câu chữ, query head bổ sung chi tiết theo vùng ảnh. Phần học thêm của "
        "nhánh văn bản chỉ có khoảng 31 nghìn tham số."))
    s.legend(["main", "op", "lock"])
    # image side
    c = s.stack(300, 880, layers=1, fills=[ORANGE] + [None] * 4)
    s.label(220, 886, 160, "Token CLS", align="center", color="#3B4756")
    s.box(180, 770, 420, 822, "Phép chiếu CLIP", lock=True)
    s.arrow([(300, c[1]), (300, 822)])
    e = s.stack(300, 730, layers=1, fills=[LORANGE] * 5)
    s.arrow([(300, 770), (300, 730)])
    # text side
    s.box(490, 852, 790, 912, "40 câu × 4 mẫu câu", kind="input")
    s.box(490, 770, 790, 822, "Text Encoder", lock=True)
    s.arrow([(640, 852), (640, 822)])
    s.box(490, 690, 790, 738, "Trừ câu “a person”", kind="op")
    s.arrow([(640, 770), (640, 738)])
    w0 = s.stack(640, 660)
    s.arrow([(640, 690), (640, 660)])
    s.label(712, 630, 150, "Hướng gốc", color="#3B4756")
    s.plus(640, 590)
    s.arrow([(640, w0[1]), (640, 606)])
    d = s.stack(846, 603, n=3, fills=[LORANGE] * 3)
    s.box(696, 566, 776, 614, "× 0.1", kind="op")
    s.arrow([(d[0] - 2, 590), (776, 590)]); s.arrow([(696, 590), (656, 590)])
    s.label(760, 524, 140, "Phần chỉnh", color=DORANGE)
    w = s.stack(640, 540)
    s.arrow([(640, 574), (640, 540)])
    # dot product x 100, bias, text logits, sum with the query head
    s.box(330, 436, 610, 484, "Tích vô hướng × 100", kind="op")
    s.arrow([(300, e[1]), (300, 460), (330, 460)])
    s.arrow([(640, w[1]), (640, 460), (610, 460)])
    s.plus(470, 390)
    s.arrow([(470, 436), (470, 406)])
    b = s.stack(250, 403, n=3, layers=1, fills=[LORANGE] * 3)
    s.arrow([(b[2] + 2, 390), (454, 390)])
    s.label(150, 340, 200, "Hệ số chệch", color=DORANGE)
    z = s.stack(470, 340, fills=["#8FC1E3"] * 5)
    s.arrow([(470, 374), (470, 340)])
    s.plus(640, 327)
    s.arrow([(z[2] + 2, 327), (624, 327)])
    qz = s.stack(800, 340, n=3, fills=["#8FC1E3"] * 3)
    s.arrow([(qz[0] - 2, 327), (656, 327)])
    s.label(740, 346, 150, "query head", color="#3B4756")
    s.arrow([(640, 311), (640, 278)])
    s.label(656, 258, 240, "40 logit của CLIP", color="#3B4756")
    for x, y, n in ((132, 776, 1), (800, 694, 2), (712, 518, 3), (500, 370, 4), (556, 262, 5)):
        s.badge(x, y, n)
    s.cards(920, 270, 872, [
        ("Dùng token CLS và phép chiếu gốc của CLIP", "Tạo đúng vector ảnh CLIP dùng khi học ghép ảnh với câu, nên so trực tiếp được với câu."),
        ("Câu kiểu ảnh giám sát, trừ câu “a person”", "Mẫu câu kiểu “a blurry photo of…” gần với ảnh camera. Phép trừ bỏ phần chung của ảnh người."),
        ("Phần chỉnh học được nhân 0.1", "Hướng phân loại học chậm hơn 10 lần, chỉ xoay trung bình 4,7°, nên vẫn bám nghĩa của câu."),
        ("Nhân 100, hệ số chệch khởi tạo từ tần suất", "100 là nhiệt độ của CLIP, giữ logit đúng thang. Thuộc tính hiếm xuất phát nghiêng về “không”."),
        ("Cộng vào logit của query head", "Học chung một loss. Câu chữ cho điểm xuất phát, query head thêm chi tiết theo vùng ảnh.")])
    return s


def main(out, preview=None):
    out = Path(out)
    slides = [arch(), block(), qharch(), qhdesign(), textarch(), textdesign()]
    for s in slides:
        html = s.html()
        n_svg = len(html.split("<svg", 1)[1].split("</svg>", 1)[0])
        (out / f"{s.sid}.html").write_text(html)
        print(f"{s.sid}: {html.count('<') // 2} tags approx, svg {n_svg / 1024:.1f} KB")
    if preview:
        pv = Path(preview); pv.mkdir(parents=True, exist_ok=True)
        head = ('<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
                'family=Be+Vietnam+Pro:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap"><style>'
                'body{margin:0} section{position:relative;width:1920px;height:1080px;box-sizing:border-box} '
                'p,h2{margin:0} table{border-collapse:collapse} td,th{padding:6px 10px;border-bottom:1px solid #DCE1E6;'
                'text-align:left} aside{display:none} section div{box-sizing:border-box}</style>')
        for s in slides:
            f = pv / f"{s.sid}.html"
            f.write_text(head + (out / f"{s.sid}.html").read_text())
            # snap Firefox only sees the home folder; a separate profile avoids the user's running instance
            subprocess.run(["firefox", "--headless", "--new-instance", "--profile", str(pv / "prof"), "--window-size=1920,1080",
                            "--screenshot", str(pv / f"{s.sid}.png"), f.as_uri()], check=True, capture_output=True, timeout=180)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--preview" else None)
