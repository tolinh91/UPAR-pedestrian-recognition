"""Architecture slide of the T2 v12 deck (DeepSeek-style block figure), drawn with the T1 deck's Slide helpers.

python t2_arch.py SLIDES_DIR      -> SLIDES_DIR/arch.html
Read bottom-up: one gallery image -> three members (student ViT-L/16 176x96, CLIP ViT-L/14 with the text branch after
WiSE-FT, DINOv3 ViT-H+/16) -> mean of the three 40-logit vectors -> kNN smoothing over the gallery, using the H+ mean
patch token -> log-likelihood of each query -> ranked list. Sizes are the package's (assets/config.json of v12).
"""
import sys
from pathlib import Path

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-1/report/v25_slides")
import arch_slides as K  # noqa: E402
from arch_slides import BLUE, DBLUE, DORANGE, LORANGE, ORANGE, Slide  # noqa: E402

PHOTO = "/_blob/bc7b8cfbd68b623a8a06f9b89951d714"   # PETA gallery image (cover_1), uploaded to the T2 deck
K.TOTAL = 22


def arch():
    s = Slide("arch", 2, "Kiến trúc", 9, "Kiến trúc tổng thể của v12", (
        "Sơ đồ đọc từ dưới lên. Mỗi ảnh trong kho ảnh đi qua ba thành viên độc lập. Thành viên thứ nhất là student "
        "DINOv3 ViT-L/16, nhận ảnh 176×96 và chia thành 66 patch token. Thành viên thứ hai là CLIP ViT-L/14, nhận ảnh "
        "224×112, có thêm nhánh văn bản và được trộn trọng số theo WiSE-FT sau huấn luyện. Thành viên thứ ba là DINOv3 "
        "ViT-H+/16, nhận ảnh 224×112 và tạo 98 patch token. Ở cả ba thành viên, patch embedding và 8 block đầu được giữ "
        "cố định khi fine-tune, query head đọc các token và cho ra 40 logit. Ba vector 40 logit được lấy trung bình. "
        "Cùng lần chạy đó, trung bình các patch token của H+ được chuẩn hoá thành một vector mô tả ngoại hình của ảnh. "
        "Vector này dùng để tìm 50 ảnh giống nhất trong kho, và logit của mỗi ảnh được trộn 20% với trung bình logit của "
        "50 ảnh đó. Cuối cùng, với mỗi truy vấn 40 thuộc tính, mỗi ảnh được chấm bằng log-likelihood của đúng truy vấn đó "
        "dưới xác suất của mô hình, và kho ảnh được xếp theo điểm này. Xác suất được dùng trực tiếp, không có bước hiệu "
        "chỉnh hay dịch ngưỡng nào."))
    s.legend(["main", "op", "lock"])
    cols = ((390, "student", BLUE, DBLUE, "DINOv3 ViT-L/16", "student · 24 block<br>176×96 · 66 patch", "Patch 16×16",
             "Query head · 40 truy vấn"),
            (750, "clip", ORANGE, DORANGE, "CLIP ViT-L/14", "24 block · 128 patch<br>văn bản + WiSE-FT", "Patch 14×14",
             "Query head + văn bản"),
            (1110, "hplus", BLUE, DBLUE, "DINOv3 ViT-H+/16", "32 block · 98 patch<br>224×112", "Patch 16×16",
             "Query head · 40 truy vấn"))
    s.box(230, 384, 1270, 432, "Trung bình ba vector 40 logit", kind="op")
    for x, key, col, dcol, name, sub, patch, qh in cols:
        c0, c1 = x - 160, x + 160
        z = s.stack(x, 500, fills=[LORANGE if key == "clip" else "#8FC1E3"] * 5)
        s.arrow([(x, z[1]), (x, 432)])
        s.box(c0, 520, c1, 568, qh)
        s.arrow([(x, 520), (x, 500)])
        t2 = s.stack(x, 618, fills=[col] + [None] * 4)
        s.arrow([(x, t2[1]), (x, 568)])
        s.box(c0, 640, c1, 760, f'<b><span style="color:{dcol}">{name}</span></b>', sub=sub)
        s.arrow([(x, 640), (x, 618)])
        t1 = s.stack(x, 806, fills=[col] + [None] * 4)
        s.arrow([(x, t1[1]), (x, 760)])
        s.box(c0, 820, c1, 868, patch, lock=True)
        s.arrow([(x, 820), (x, 806)])
        if key == "student":
            s.label(z[2] + 12, 470, 120, "40 logit", color="#3B4756")
        if key == "hplus":
            s.arrow([(t2[2] + 2, 605), (1330, 605)])
    # right panel: appearance embedding -> kNN smoothing -> log-likelihood of the query -> ranked list
    s.box(1330, 569, 1550, 641, "Trung bình<br>patch token", kind="op")
    s.box(1330, 372, 1550, 444, "Làm mượt kNN", sub="k 50 · trộn 20%", kind="op")
    s.arrow([(1440, 569), (1440, 444)])
    s.arrow([(1270, 408), (1330, 408)])
    s.box(1330, 256, 1550, 324, "Log-likelihood<br>với truy vấn")
    s.arrow([(1440, 372), (1440, 324)])
    s.box(1600, 256, 1792, 324, "Truy vấn<br>40 thuộc tính", kind="input")
    s.arrow([(1600, 290), (1550, 290)])
    s.box(950, 256, 1270, 324, "Danh sách ảnh<br>xếp hạng", kind="op")
    s.arrow([(1330, 290), (1270, 290)])
    # one gallery image feeds the three members
    s.image(128, 780, 44, 88, PHOTO, "Ảnh trong kho ảnh")
    s.label(128, 718, 100, "Ảnh<br>trong kho", color="#3B4756")
    s.arrow([(172, 844), (230, 844)])
    s.arrow([(150, 868), (150, 900), (750, 900), (750, 868)])
    s.arrow([(150, 900), (1110, 900), (1110, 868)])
    return s


out = Path(sys.argv[1])
h = arch().html().replace("Kiến trúc T1 v29 · ", "Kiến trúc T2 v12 · ")
(out / "arch.html").write_text(h, encoding="utf-8")
print("arch.html", len(h), "chars")
