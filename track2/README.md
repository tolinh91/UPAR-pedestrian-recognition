# UPAR Challenge 2027 – Track 2: Attribute-based Person Retrieval

Một truy vấn là một vector 40 thuộc tính (0/1). Chương trình phải xếp hạng toàn bộ gallery ảnh người đi bộ sao cho
những người khớp truy vấn đứng đầu. Metric xếp hạng là mADM (mean average degree of match, tạm dịch: mức khớp trung
bình), giống mAP nhưng cộng điểm cho những ảnh khớp gần đúng.

Cách làm hiện tại: dùng lại model nhận dạng thuộc tính của Track 1 (`../Upar-Challenge-Track-1`) để tính xác suất 40
thuộc tính cho mỗi ảnh, rồi chấm mỗi cặp truy vấn và ảnh bằng log-likelihood khớp đúng toàn bộ 40 thuộc tính.

| Thư mục / file | Nội dung |
|---|---|
| `par/upar/retrieval.py` | Hàm chấm điểm (L1, log-likelihood) và các metric chính thức (mADM, mAP, mINP, Rank-k) |
| `par/eval_retrieval.py` | Đánh giá truy vấn từ logits của một run Track 1, mỗi dataset val là một gallery |
| `submission_template/run.py` | Bài nộp: `rank_gallery()` trả về ma trận `similarities` |
| `par/fake_ingestion_t2.py` | Chương trình ingestion giả để test bài nộp trong docker |

`UPAR-Challenge-2027`, `cache` và `.venv` là symlink sang thư mục Track 1.
