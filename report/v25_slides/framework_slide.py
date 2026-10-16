"""Framework slide of the T1 v29 deck in the style of the AM-RADIO overview figure: input image, image encoders drawn as
hexagons, trainable heads (purple), frozen parts (blue), training losses (pink), snowflake = frozen, flame = trainable,
results of the example image on the right.

python framework_slide.py SLIDES_DIR PAGE TOTAL   -> SLIDES_DIR/framework.html
Boxes and text are pinned <div>/<p> (deck fonts); hexagons, arrows and icons are two inline <svg> (no text inside).
Numbers on the result cards are the deck's own (slides 08, 13 and 20): backpack attention 89% on the upper body,
backpack sentence score +2,5 on this image and -1,6 on an image without a backpack, 9 predicted attributes, 7 correct.
"""
import sys
from pathlib import Path

INK, MUTE, PAPER = "#14202E", "#5A6573", "#F5F3EE"
GREEN_F, GREEN_S = "#DCEBD5", "#5E9E4F"
PURPLE_F, PURPLE_S = "#E9E1F3", "#8E7CC3"
BLUE_F, BLUE_S = "#D9E7F5", "#6C9BD2"
PINK_F, PINK_S = "#F9DCD8", "#D9776B"
LINE = "#2B3A4E"
PHOTO = "/_blob/ffb0b66c4c3254ffa22a81858932eac7"        # the deck's example image (backpack)
ATT_BACKPACK = "/_blob/9736324aa948f13de6ca0c73d5ae666c"  # attention of the backpack query (slide 08)

out_dir, PAGE, TOTAL = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
el, bg, fg = [], [], []      # html elements, svg drawn under the text (hexagons), svg drawn on top (arrows, icons)


def box(x0, y0, x1, y1, lines, fill, stroke, dashed=False, italic=False, size=24):
    """Rounded module; first line bold."""
    ps = "".join(f'<p style="font-size:{size if i else size + 2}px;line-height:1.25;text-align:center;'
                 f'{"font-weight:700;" if i == 0 and not italic else ""}{"font-style:italic;" if italic else ""}'
                 f'color:{INK if i == 0 else "#3B4756"}">{t}</p>' for i, t in enumerate(lines))
    el.append(f'<div style="position:absolute;left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px;background:{fill};'
              f'border:{3 if not dashed else 2}px {"dashed" if dashed else "solid"} {stroke};border-radius:14px;display:flex;'
              f'flex-direction:column;justify-content:center;align-items:center;gap:2px">{ps}</div>')


def label(x0, y0, w, text, size=24, color=INK, bold=False, align="left", italic=False):
    el.append(f'<p style="position:absolute;left:{x0}px;top:{y0}px;width:{w}px;font-size:{size}px;line-height:1.3;'
              f'text-align:{align};color:{color}{";font-weight:700" if bold else ""}{";font-style:italic" if italic else ""}">'
              f'{text}</p>')


def hexagon(x0, y0, x1, y1):
    a, cy = 40, (y0 + y1) / 2
    pts = f"{x0},{cy} {x0 + a},{y0} {x1 - a},{y0} {x1},{cy} {x1 - a},{y1} {x0 + a},{y1}"
    bg.append(f'<polygon points="{pts}" fill="{GREEN_F}" stroke="{GREEN_S}" stroke-width="7" stroke-linejoin="round"/>')


def arrow(pts, dashed=False, color=LINE, sw=3):
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    L = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5; k = (L - 4 * sw) / L
    pts = pts[:-1] + [(xa + (xb - xa) * k, ya + (yb - ya) * k)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    fg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{dash} marker-end="url(#h)"/>')


def plus(cx, cy, r=16):
    fg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{LINE}" stroke-width="2.5"/>'
              f'<path d="M {cx - r} {cy} L {cx + r} {cy} M {cx} {cy - r} L {cx} {cy + r}" stroke="{LINE}" stroke-width="2.5"/>')


def flame(cx, cy, r=15):
    """Trainable: a two-tone flame."""
    def drop(x, y, s):
        return (f"M {x} {y - s} C {x + 0.8 * s} {y - 0.2 * s} {x + 0.75 * s} {y + 0.9 * s} {x} {y + s} "
                f"C {x - 0.75 * s} {y + 0.9 * s} {x - 0.8 * s} {y - 0.2 * s} {x} {y - s} Z")
    fg.append(f'<path d="{drop(cx, cy, r)}" fill="#E8772E" stroke="#B4501A" stroke-width="1.5"/>'
              f'<path d="{drop(cx, cy + 0.35 * r, 0.55 * r)}" fill="#F6C344"/>')


def snow(cx, cy, r=14):
    """Frozen: a six-armed snowflake."""
    import math
    d = []
    for k in range(3):
        a = math.pi / 3 * k
        dx, dy = r * math.cos(a), r * math.sin(a)
        d.append(f"M {cx - dx:.1f} {cy - dy:.1f} L {cx + dx:.1f} {cy + dy:.1f}")
        for sgn in (1, -1):                       # a small V near each arm end
            ex, ey = cx + sgn * 0.62 * dx, cy + sgn * 0.62 * dy
            for b in (math.pi / 4, -math.pi / 4):
                bx, by = math.cos(a + b) * 0.3 * r * sgn, math.sin(a + b) * 0.3 * r * sgn
                d.append(f"M {ex:.1f} {ey:.1f} L {ex + bx:.1f} {ey + by:.1f}")
    fg.append(f'<path d="{" ".join(d)}" stroke="#1F5F8F" stroke-width="2.6" stroke-linecap="round" fill="none"/>')


# ---- input ----
label(128, 330, 220, "Ảnh đầu vào", bold=True)
el.append(f'<img src="{PHOTO}" alt="Ảnh ví dụ: người đeo balo xanh ngọc, áo trắng ngắn tay" style="position:absolute;left:128px;'
          f'top:380px;width:160px;height:320px;object-fit:cover;border-radius:12px">')
arrow([(288, 540), (308, 540), (308, 365), (330, 365)])
arrow([(308, 540), (308, 615), (330, 615)])

# ---- image encoders (hexagons) ----
for y0, name in ((290, "DINOv3 ViT-H+"), (540, "CLIP ViT-L/14")):
    hexagon(330, y0, 650, y0 + 150)
    label(370, y0 + 22, 240, name, size=26, bold=True, align="center")
    label(370, y0 + 60, 240, "bộ mã hóa ảnh", align="center", color="#3B4756")
    label(420, y0 + 95, 190, "8 block đầu", align="left", color="#3B4756")
    snow(400, y0 + 111, 12)
    flame(632, y0 + 14)

# ---- DINOv3 member ----
box(700, 330, 970, 400, ["Query head", "40 query"], PURPLE_F, PURPLE_S)
flame(958, 330)
arrow([(650, 365), (700, 365)])

# ---- CLIP member: query head + text branch fed by the frozen text encoder ----
box(700, 545, 970, 615, ["Query head", "40 query"], PURPLE_F, PURPLE_S)
flame(958, 545)
box(700, 650, 970, 720, ["Nhánh văn bản", "phần chỉnh nhỏ"], PURPLE_F, PURPLE_S)
flame(958, 650)
arrow([(650, 615), (675, 615), (675, 580), (700, 580)])
arrow([(675, 615), (675, 685), (700, 685)])
label(632, 696, 66, "CLS", size=24, color=MUTE, align="right")
box(700, 770, 970, 840, ["Bộ mã hóa văn bản", "chạy một lần"], BLUE_F, BLUE_S)
snow(958, 772)
arrow([(835, 770), (835, 720)])
box(380, 765, 650, 845, ["“a person carrying", "a backpack”"], "#FFFFFF", "#8E99A6", italic=True)
arrow([(650, 805), (700, 805)])
plus(1010, 632)
arrow([(970, 580), (1010, 580), (1010, 616)])
arrow([(970, 685), (1010, 685), (1010, 648)])

# ---- training: one loss per member against the labels (dashed = training only) ----
for ly, lo in ((235, 330), (500, 597)):
    box(1055, ly, 1325, ly + 60, ["Nhãn 40 thuộc tính"], "#FFFFFF", "#8E99A6")
    arrow([(1190, ly + 60), (1190, lo)])
    box(1055, lo, 1325, lo + 70, ["Loss điều chỉnh", "theo tần suất nhãn"], PINK_F, PINK_S)
arrow([(970, 365), (1055, 365)], dashed=True)
arrow([(1026, 632), (1055, 632)], dashed=True)

# ---- results on the example image ----
label(1390, 196, 402, "Kết quả trên ảnh ví dụ", size=24, color=MUTE, bold=True)
el.append(f'<div style="position:absolute;left:1390px;top:235px;width:402px;height:225px;background:#FFFFFF;border:1px solid #DCE1E6;'
          f'border-radius:14px"></div>')
el.append(f'<img src="{ATT_BACKPACK}" alt="Attention của query balo trên ảnh ví dụ, tập trung ở thân trên" style="position:absolute;'
          f'left:1406px;top:250px;width:96px;height:192px;object-fit:cover;border-radius:8px">')
label(1518, 262, 260, "Vùng query balo chú ý", bold=True)
label(1518, 330, 260, "89% trọng số ở thân trên", color="#3B4756")
el.append(f'<div style="position:absolute;left:1390px;top:500px;width:402px;height:220px;background:#FFFFFF;border:1px solid #DCE1E6;'
          f'border-radius:14px"></div>')
label(1410, 518, 362, "Điểm của câu mô tả balo", bold=True)
label(1410, 560, 362, "“a person carrying a backpack”", italic=True, color="#3B4756")
label(1410, 612, 250, "ảnh này", color="#3B4756"); label(1650, 612, 122, "+2,5", bold=True, color="#9C4A15", align="right")
label(1410, 655, 250, "ảnh không balo", color="#3B4756"); label(1650, 655, 122, "−1,6", bold=True, color="#23679A", align="right")
el.append(f'<div style="position:absolute;left:1390px;top:765px;width:402px;height:150px;background:#14202E;border-radius:14px"></div>')
label(1410, 782, 362, "Dự đoán sau khi kết hợp", bold=True, color="#F5F3EE")
label(1410, 826, 362, "9 thuộc tính “có”, 7 đúng", color="#C5D0DB")
label(1410, 862, 362, "trung bình logit hai mô hình", color="#C5D0DB", size=24)

# ---- legend ----
el.append(f'<div style="position:absolute;left:128px;top:770px;width:226px;height:146px;background:#FFFFFF;border:1px solid #DCE1E6;'
          f'border-radius:14px"></div>')
snow(156, 801, 13); label(180, 786, 170, "Cố định")
flame(156, 843, 14); label(180, 828, 170, "Học được")
fg.append(f'<path d="M 140 885 L 176 885" stroke="{LINE}" stroke-width="3" stroke-dasharray="8 6"/>')
label(186, 868, 168, "Khi huấn luyện")

svg_bg = (f'<svg aria-label="Hai khối lục giác của bộ mã hóa ảnh" width="1920" height="1080" viewBox="0 0 1920 1080" '
          f'style="position:absolute;left:0px;top:0px;width:1920px;height:1080px">' + "".join(bg) + "</svg>")
svg_fg = (f'<svg aria-label="Mũi tên luồng dữ liệu, biểu tượng bông tuyết (cố định) và ngọn lửa (học được)" width="1920" '
          f'height="1080" viewBox="0 0 1920 1080" style="position:absolute;left:0px;top:0px;width:1920px;height:1080px"><defs>'
          f'<marker id="h" orient="auto" markerWidth="4" markerHeight="4" refX="0" refY="2" overflow="visible">'
          f'<path d="M0 0 L4 2 L0 4 Z" fill="{LINE}"/></marker></defs>' + "".join(fg) + "</svg>")
notes = (
    "Lời nói: Hình này gom toàn bộ v29 vào một khung, theo kiểu các bài báo về mô hình nền. Bên trái là ảnh đầu vào. "
    "Ảnh đi vào hai bộ mã hóa ảnh, vẽ bằng hình lục giác: DINOv3 ở trên và CLIP ở dưới. Biểu tượng ngọn lửa nghĩa là "
    "phần đó được học trong lúc huấn luyện, bông tuyết nghĩa là phần đó giữ cố định. Ở cả hai bộ mã hóa, 8 block đầu "
    "giữ cố định, các block sau được học tiếp. Sau bộ mã hóa là query head màu tím, học được, cho 40 logit. Riêng CLIP "
    "có thêm nhánh văn bản. Câu mô tả như “a person carrying a backpack” đi qua bộ mã hóa văn bản của CLIP, phần này cố "
    "định và chỉ chạy một lần trước huấn luyện. Nhánh văn bản lấy token CLS của ảnh, so với vector của câu, và chỉ học "
    "một phần chỉnh nhỏ. Logit của query head và của nhánh văn bản được cộng lại. Mũi tên nét đứt là phần chỉ dùng khi "
    "huấn luyện: logit của mỗi mô hình được so với 40 nhãn thật bằng loss điều chỉnh theo tần suất nhãn, mỗi mô hình một "
    "loss riêng. Cột bên phải là kết quả thật trên ảnh ví dụ. Query balo đặt 89% trọng số ở thân trên. Câu mô tả balo cho "
    "+2,5 trên ảnh này và −1,6 trên ảnh không có balo. Sau khi lấy trung bình logit của hai mô hình, có 9 thuộc tính được "
    "dự đoán là có, 7 trong số đó đúng. Khi suy luận, mô hình chỉ dùng các mũi tên liền và bước lấy trung bình ở slide trước.\n\n"
    "Câu hỏi kiểm tra: Bộ mã hóa văn bản có được học khi huấn luyện không? (Đáp án: không, nó giữ cố định và chỉ chạy "
    "một lần.)\n\nChuyển: Thành phần cần nói kỹ nhất là loss, vì nó quyết định mô hình xử lý thuộc tính hiếm ra sao.")
html = "\n".join([
    f'<section id="framework" data-transition="fade" style="background:{PAPER};color:{INK};font-family:\'Be Vietnam Pro\', '
    f'Arial, sans-serif;padding:128px 128px 160px;display:flex;flex-direction:column;gap:40px">',
    svg_bg,
    f'<h2 style="font-size:64px;font-weight:700;line-height:1.1;color:{INK}">Khung kiến trúc của v29</h2>',
    *el, svg_fg,
    f'<p style="position:absolute;left:128px;bottom:64px;width:1000px;font-size:24px;color:{MUTE}">Nhận dạng thuộc tính người · v29</p>',
    f'<p style="position:absolute;right:128px;bottom:64px;width:160px;text-align:right;font-size:24px;color:{MUTE}">{PAGE:02d} / {TOTAL}</p>',
    f"<aside>{notes}</aside>", "</section>", ""])
(out_dir / "framework.html").write_text(html, encoding="utf-8")
print("framework.html", len(html), "chars,", html.count("<"), "tags approx")
