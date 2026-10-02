# Phản biện kế hoạch thí nghiệm UPAR 2027 (Track 1 và Track 2), 2026-10-02

Tài liệu này phản biện `report/experiment_plan.md`. Mọi con số mới trong tài liệu được tính lại hôm nay từ các file logits có sẵn trong `Upar-Challenge-Track-1/runs/` bằng chính `upar/retrieval.py` và `calibrate.py`. Không có code nào trong repo bị sửa. Script phân tích nằm ở scratchpad của phiên làm việc, không thuộc repo. Các số Track 2 dưới đây là mADM (mean average degree of match, tạm dịch: mức khớp trung bình) trên gallery PETA val của proxy giữ lại PETA, gồm 6963 ảnh và 1763 query, trừ khi ghi khác.

## 0. Năm phê bình quan trọng nhất

1. **Track 2 đang bị chọn model bằng tiêu chí của Track 1, và tiêu chí đó loại nhầm biến thể tốt nhất.** Run `q_gce` (query head cộng group cross-entropy, tạm dịch: entropy chéo trên từng nhóm thuộc tính loại trừ nhau) bị loại ở Track 1 vì Avg@0.6 thấp hơn mốc 0.004. Trên Track 2, nó đạt mADM 0.6424 so với 0.6261 của query head thường. Chênh lệch +0.0164 có sai số chuẩn ghép cặp theo query là 0.0024, gấp khoảng 7 lần sai số, và gấp 23 lần chênh lệch giữa hai lần chạy lặp (0.0007). Đây là biến thể ViT-B tốt nhất cho Track 2 trong 15 run có logits. mADM bám theo NLL (negative log-likelihood, tạm dịch: log-likelihood âm) của cả vector nhãn trên ảnh held-out, không bám theo AUC. Ví dụ ViT-L 160×80 có AUC thấp nhất (0.896) nhưng mADM 0.6335, cao hơn mọi ViT-B query.
2. **Recipe được chấm trên proxy không phải recipe được nộp.** Proxy train 5 epoch với cosine schedule theo 5 epoch và chấm `last.pt`. Bài nộp ViT-L 176 train 10 epoch (bản train+val 8 epoch) và lấy `best.pt` ở epoch 6, chọn theo Challenge_Avg trên val cùng miền. Trên PETA held-out, mọi query head đạt đỉnh ở epoch 2 rồi giảm (ViT-L 176 raw 0.7979 ở epoch 2 xuống 0.7956 ở epoch 4), trong khi val cùng miền tăng đến epoch 6. Run 3 epoch của ViT-B query cho mADM 0.6350 so với 0.6261 của run 5 epoch. Chưa có số proxy nào cho chính model đang được nộp.
3. **Hướng ViT-L 224 (cắt bớt block, token merging, dense distillation) dựa trên một chênh lệch chỉ xuất hiện sau offset.** Ở ViT-L query, 224 so với 176 có AUC 0.9101 so với 0.9102 và raw Avg 0.7969 so với 0.7956. Ở 192 thì raw và AUC đều thấp hơn 176. Toàn bộ khoảng +0.0066 nằm trong Avg@0.6, đúng loại thay đổi qua ngưỡng mà kế hoạch tự nói là proxy không đáng tin. Khoảng 6 đến 8 run GPU trong danh sách ứng viên phụ thuộc vào một tiền đề chưa được kiểm chứng.
4. **Phép đo nhiễu quá yếu nên các bài học rút ra từ leaderboard chưa có cơ sở.** Hai lần chạy y hệt của ViT-B query lệch nhau 0.0005 ở Avg@0.6 nhưng lệch 0.0016 ở raw và 0.0033 ở AUC. Khoảng AUC của toàn bộ biến thể ViT-B (0.898 đến 0.908) chỉ rộng gấp 3 lần mức lệch này. Trên tập chấm ẩn, train so với train+val (cùng recipe) lệch 0.0019. Hai lần proxy được cho là "đoán sai" có độ lệch −0.0025 và −0.0009, cùng cỡ với mức lệch giữa hai lần train. Bài học "proxy không đáng tin cho thay đổi kiểu ngưỡng" vì vậy chưa được xác nhận và cũng chưa bị bác bỏ.
5. **Hai rủi ro vận hành có thể làm hỏng bài nộp Track 2 trước khi model kịp có ý nghĩa.** Thứ nhất, bài nộp trả về `{"similarities": ...}`, còn code mẫu chính thức trả về `{"distances": ...}` (nhỏ hơn là khớp hơn). `fake_ingestion_t2.py` do đội tự viết nên kiểm tra theo giả định của đội, không theo chương trình chấm thật. Nếu chương trình chấm chỉ đọc `distances`, bài nộp lỗi. Nếu nó đọc ma trận như khoảng cách, thứ hạng bị đảo ngược. Thứ hai, hằng số quy đổi thời gian CPU đang mâu thuẫn. Memory ghi ViT-L 176 khoảng 78 ms/ảnh và 224 khoảng 118 ms/ảnh, còn kế hoạch ghi 98 và 149 ms/ảnh. Hằng số này quyết định ViT-L 224 có cần token merging hay không, và có còn chỗ cho ensemble hay không.

---

## 1. Lỗi phương pháp trong giao thức đánh giá

### 1.1. Proxy và bài nộp dùng hai recipe khác nhau

Mục đích của proxy là dự đoán điểm của đúng model sẽ được nộp. Hiện proxy và bài nộp khác nhau ở ba điểm.

| Yếu tố | Proxy (`score_run.py`) | Bài nộp ViT-L 176 (v9, Track 2 v1) |
|---|---|---|
| Số epoch, độ dài cosine | 5 | 10 (train), 8 (train+val) |
| Checkpoint | `last.pt` (EMA) | `best.pt` ở epoch 6 theo Avg trên val cùng miền (v9, Track 2 v1), `last.pt` (v10) |
| Dữ liệu train | 2 trên 3 dataset | cả 3 dataset |

Đường cong theo epoch trên PETA held-out cho thấy hai recipe này không tương đương.

| Run (PETA held-out, raw Avg theo epoch 0 đến 4) | Đỉnh |
|---|---|
| ViT-B query lần 1: 0.7143, 0.7886, **0.7919**, 0.7886, 0.7864 | epoch 2 |
| ViT-B query lần 2: 0.7118, 0.7863, **0.7904**, 0.7886, 0.7880 | epoch 2 |
| ViT-L 176 query: 0.7334, 0.7938, **0.7979**, 0.7959, 0.7956 | epoch 2 |
| ViT-L 224 query: 0.7318, 0.7967, **0.7997**, 0.7993, 0.7969 | epoch 2 |
| ViT-B linear: 0.6644, 0.7536, 0.7718, 0.7774, **0.7785** | epoch 4 |

Trên val cùng miền, ViT-L 176 query tăng đến epoch 5 và 6 (0.8797) rồi mới đi ngang. Vì vậy chọn `best.pt` theo val cùng miền sẽ chọn epoch muộn, đúng vùng mà điểm ngoài miền đang giảm. Với Track 2, run 3 epoch (`ft_PETA_dinov3_q_e3`) đạt mADM 0.6350 và NLL/ảnh 7.10, so với 0.6261 của run 5 epoch. Ở đây NLL lệch hướng với mADM, nên đây là một điểm dữ liệu cần lặp lại chứ chưa phải kết luận.

Hệ quả là chưa có con số proxy nào cho recipe 10 epoch cộng `best.pt`. Cách sửa rẻ nhất là nộp đúng recipe được chấm. Cụ thể là train 5 epoch trên toàn bộ dữ liệu và lấy `last.pt`. Cách này còn giảm một nửa chi phí GPU. Thí nghiệm R6 và R7 ở mục 4 đo trực tiếp ảnh hưởng của số epoch.

### 1.2. Nhiễu và seed

- Kế hoạch ước lượng nhiễu từ một cặp run. Một cặp chỉ cho một bậc tự do, nên ước lượng độ lệch chuẩn từ nó có thể sai gấp 2 đến 3 lần.
- Nhiễu khác nhau theo thống kê. Với cặp run lặp của ViT-B query, Avg@0.6 lệch 0.0005, raw Avg lệch 0.0016, AUC lệch 0.0033, mADM lệch 0.0007 và NLL/ảnh lệch 0.022. Ngưỡng chấp nhận 0.003 chỉ phù hợp với Avg@0.6. Ngưỡng này không đủ cho AUC, trong khi báo cáo lại khuyên dùng AUC làm tiêu chí.
- Nhiễu do lấy mẫu query trên PETA Track 2 có sai số chuẩn 0.0052 cho một model. Khi so ghép cặp hai model trên cùng các query, sai số chuẩn giảm còn 0.0021 đến 0.0029. Mọi so sánh Track 2 phải là so sánh ghép cặp trên cùng query.
- `--seed 0` có nghĩa là không đặt seed. Mọi run cũ đều không có seed và chạy với phiên bản code cũ hơn. Hai run seed 1 và seed 2 trong đợt đang chạy chỉ nên được ghép với nhau, không ghép với run cũ, vì thay đổi code (compile, data loader) là một nguồn lệch khác.
- Trên leaderboard, chênh lệch giữa train và train+val là −0.0019. Đây là ước lượng duy nhất của nhiễu giữa hai lần train trên tập chấm thật. Mọi chênh lệch leaderboard dưới khoảng 0.003 nên được coi là hoà.

### 1.3. Chọn trên chính tập held-out

- Hệ số offset 0.6 được chọn bằng cách nhìn đồ thị trên PETA (báo cáo v5, Hình metric tradeoff), rồi lại dùng Avg@0.6 trên PETA để so các biến thể. Tiêu chí như vậy bị lặp vòng.
- Khoảng 35 biến thể được so trên cùng một gallery PETA val. Khi chọn biến thể có điểm cao nhất trong 35 lần đo có nhiễu độ lệch chuẩn σ, điểm của biến thể thắng bị thổi lên khoảng 2σ. Với σ của Avg@0.6 cỡ 0.001, mức thổi khoảng 0.002.
- `teacher_PETA_dinov3L/val_logits.npz` và `ft_PETA_dinov3L_160x80/val_logits.npz` là `best.pt` chọn trên PETA. Với head linear, đường cong tăng đều nên `best.pt` trùng `last.pt` và không có thiên lệch. Nhưng `offset_proxy.py` mặc định vẫn là `best.pt`. Cần ép `last.pt` ở mọi nơi để tránh nhầm.

### 1.4. Mỗi track cần một thống kê khác

Track 1 chấm ở ngưỡng cố định 0.5, nên điểm phụ thuộc cả khả năng phân biệt (AUC) lẫn vị trí ngưỡng. Track 2 xếp hạng bằng log-likelihood của cả vector 40 thuộc tính. Thứ hạng đó phụ thuộc vào xác suất có hiệu chỉnh tốt trên miền mới hay không, và AUC từng thuộc tính không đo được điều này. Bảng dưới tính lại cho mọi run có logits PETA held-out.

| Run (PETA held-out) | AUC | NLL/ảnh | mADM (loglik) | mAP |
|---|---|---|---|---|
| ViT-L 256 linear | 0.9126 | 6.569 | **0.6697** | 0.3233 |
| ViT-B query + group CE | 0.9048 | 6.771 | 0.6424 | 0.2996 |
| ViT-B query, 3 epoch | 0.9041 | 7.102 | 0.6350 | 0.2840 |
| ViT-L 160×80 linear (8 epoch) | 0.8960 | 6.823 | 0.6335 | 0.2877 |
| ViT-B linear, logit KD từ ViT-L | 0.9028 | 6.861 | 0.6334 | 0.2868 |
| ViT-B linear | 0.9076 | 6.993 | 0.6313 | 0.2800 |
| ViT-B query + white balance | 0.9001 | 6.985 | 0.6294 | 0.2836 |
| PE-Core linear | 0.8950 | 6.916 | 0.6276 | 0.2766 |
| ViT-B query 2 lớp | 0.9020 | 7.013 | 0.6270 | 0.2771 |
| ViT-B query (lần 1 / lần 2) | 0.9007 / 0.9040 | 6.976 / 6.998 | 0.6261 / 0.6254 | 0.2773 / 0.2768 |
| ViT-B dual head | 0.9041 | 7.037 | 0.6260 | 0.2760 |
| ViT-B query, 4 tầng token | 0.8988 | 6.997 | 0.6242 | 0.2749 |
| ViT-B query, cân bằng dataset | 0.9049 | 7.008 | 0.6214 | 0.2740 |
| ViT-B logit adjustment | 0.9019 | 31.86 | 0.5396 | 0.1682 |

Trừ run 3 epoch, thứ tự theo NLL/ảnh gần như trùng thứ tự theo mADM, còn AUC thì không. Đề xuất cụ thể là Track 2 dùng mADM ghép cặp làm tiêu chí chính và NLL/ảnh trên ảnh held-out làm tiêu chí phụ. NLL là thống kê trơn, không có ngưỡng, và đo đúng đại lượng mà hàm xếp hạng loglik sử dụng. `score_run.py` nên ghi thêm NLL/ảnh vào `scores.json`.

### 1.5. Bằng chứng cho ViT-L 224 nằm hoàn toàn trong offset

| ViT-L query, PETA | AUC | raw Avg | Avg@0.6 | Avg@1.0 |
|---|---|---|---|---|
| 176×96 | 0.9102 | 0.7956 | 0.8005 | 0.7980 |
| 192×96 | 0.9078 | 0.7935 | 0.8014 | 0.7999 |
| 224×112 | 0.9101 | 0.7969 | 0.8071 | 0.8053 |

Độ phân giải từ 176 đến 224 không đổi AUC và raw Avg vượt mức nhiễu. Thứ tự 192 < 176 < 224 không đơn điệu, phù hợp với nhiễu. Theo đúng nguyên tắc báo cáo v5 tự đặt ra (chỉ tin thay đổi làm AUC tăng), hướng 224 chưa đủ cơ sở. Điều duy nhất có thể cứu hướng này là Track 2. Nếu 224 làm NLL/ảnh và mADM tăng rõ, nó đáng làm cho Track 2. Hiện chưa có logits PETA của q_L224 ở máy local, nên cần chạy lại (R8).

Nhận xét liên quan là head linear có AUC cao hơn query head ở cả ViT-B (0.9076 so với 0.9007 đến 0.9040) lẫn ViT-L (256 linear 0.9126). Leaderboard cũng xếp ViT-B linear (0.7775) trên ViT-B query (0.7750), và Track 2 xếp linear trên query (+0.0052, sai số chuẩn ghép cặp 0.0029). Chỉ Avg@0.6 trên PETA ủng hộ query head. Mọi bài nộp hiện tại đều dùng query head. Run ViT-L 176 linear đang chạy là run có giá trị quyết định cao nhất trong đợt này.

### 1.6. Gallery val của Track 2 và miền test duy nhất

- Cách chấm từng dataset thành một gallery riêng là đúng, vì test là một miền duy nhất.
- Ba gallery có cấu trúc rất khác nhau. Ở PETA, 61% query chỉ có một ảnh khớp. Ở PA100k tỉ lệ này là 42%, ở Market là 15%, và Market có trung vị 10 ảnh khớp cho mỗi query. mADM của cùng một model ở ba gallery lệch nhau tới 0.08 (Market 0.758, PA100k 0.724, PETA 0.805 với ViT-L 176 cùng miền). Vì vậy khi gộp các holdout, ta phải lấy trung bình của chênh lệch ghép cặp so với mốc trong từng gallery, không lấy trung bình mADM thô.
- Val cùng miền chỉ hợp để kiểm tra code và so các hàm chấm điểm có chênh lệch lớn như loglik so với L1. Nó không được dùng để chỉnh bất kỳ tham số nào của hàm chấm điểm. Bằng chứng là temperature tối ưu theo NLL trên val của miền nguồn là T = 0.80 đến 0.85 (làm xác suất sắc hơn), và khi áp lên PETA held-out nó làm mADM giảm 0.004 đến 0.009. Ngược lại, trên miền mới, T = 1.25 làm mADM tăng ở 4 trên 5 model.
- Holdout PA100k nghĩa là chỉ còn 18.7 nghìn ảnh train, ít hơn khoảng 5 lần so với recipe nộp. Kết quả ở fold này đo hành vi khi thiếu dữ liệu và thiên về các thay đổi kiểu regularization. Nên giảm trọng số của fold này, hoặc chỉ dùng nó để phát hiện thay đổi gây hại.
- Holdout Market có 6 thuộc tính màu không bao giờ được gán nhãn (Brown, Orange, Pink của áo và Orange, Purple, Red của quần). Trên gallery này, loglik phạt những ảnh mà model đoán có màu đó, dù nhãn thật luôn bằng 0. Bỏ 6 thuộc tính này khỏi hàm chấm điểm làm mADM tăng 0.004 và mAP tăng 0.008 (ViT-L 176, val cùng miền). Fold này vì vậy đo cả độ bền với quy ước gán nhãn, điều mà fold PETA không đo được.

### 1.7. Kiểm tra `upar/retrieval.py` so với định nghĩa

Phần cài đặt khớp với định nghĩa ở các điểm sau.

- `dom = eq / K` với `eq` là số thuộc tính bằng nhau trên đủ 40 thuộc tính, gồm cả thuộc tính mà dataset không gán nhãn. Điều này đúng định nghĩa.
- `ndom = clip((dom − mean)/(max − mean), 0)`, với mean và max lấy trên toàn gallery của query đó, kể cả chính các ảnh khớp. Điều này đúng định nghĩa.
- `ADM = mean_k cumsum(ndom theo thứ hạng)[r_k − 1] / r_k`. Điều này đúng định nghĩa. mAP, mINP = R / r_R và Rank-k cũng đúng.
- Ảnh khớp được xác định bằng `g_qid == q_id`. Tôi đã kiểm tra trên cả train và val rằng mỗi qid ứng với đúng một vector duy nhất (3462 vector khác nhau cho 3462 query ở val) và 100% ảnh có vector nhãn bằng vector query của nó. Vì vậy so qid tương đương với điều kiện cả 40 thuộc tính bằng nhau.

Có bốn điểm cần lưu ý.

1. Khi điểm bằng nhau, `argsort(..., kind="stable")` giữ thứ tự gallery. Code mẫu chính thức ghi "Ties keep the gallery order" cho `distances`. Điều này chỉ quan trọng khi cơ chế dự phòng ngân sách thời gian gán cùng một xác suất trung bình cho các ảnh cuối, tạo ra cả khối điểm bằng nhau.
2. Query không có ảnh khớp bị bỏ qua. Theo cách tạo query (mỗi vector có trong gallery là một query) thì trường hợp này không xảy ra trên test.
3. Khi `max == mean`, code trả ndom = 0, còn code chính thức có thể trả NaN. Trường hợp này không xảy ra trong thực tế.
4. Tôi không có code chấm chính thức. Nếu trang Codabench công bố scoring program thì nên tải về và so từng dòng. Nó cũng sẽ trả lời câu hỏi về `similarities` và `distances` ở mục 0.

Kích thước ma trận cũng là một rủi ro. `rank_gallery` tạo ma trận Q×G kiểu float64 cùng ít nhất một ma trận tạm cùng cỡ. Với khoảng 3500 query và 29 nghìn ảnh, mỗi ma trận chiếm khoảng 0.8 GB, nên đỉnh bộ nhớ khoảng 2 đến 2.5 GB. Nên tính theo từng khối và giữ float32.

### 1.8. Hằng số thời gian CPU

Kế hoạch ghi ViT-L 176×96 khoảng 98 ms/ảnh (48 phút) và 224×112 khoảng 149 ms/ảnh. Memory `upar-eval-lessons` ghi lần lượt 78 và 118 ms/ảnh, dựa trên log server của C-RADIO-B (49.5 ms/ảnh, hệ số 1.6 so với docker trên laptop). Ghi chú về bài nộp v9 lại dùng hệ số 120/98 = 1.22. Hai hệ số lệch nhau 30%. Bài v9 (0.7857) đã chạy trên server, và `run.py` ghi `ms/img` sau mỗi 1280 ảnh. Đọc log ingestion của v9 sẽ cho con số thật mà không tốn lượt nộp. Mọi quyết định về ToMe, cắt block và ensemble phải chờ con số này.

### 1.9. Luật thi và tập query

Query của test được tạo theo cùng quy tắc với val, nên tập query chính là tập các vector nhãn khác nhau của gallery test. Mọi thống kê tính chéo nhiều query đều là thống kê trên nhãn test. Ví dụ là thuộc tính nào bằng 0 ở mọi query, tần suất đa màu, hay giả định "nhãn của mỗi ảnh phải là một trong các query". Luật chỉ cấm dùng ảnh test, nhưng dùng tập query theo cách này đi ngược tinh thần của luật. Trên val, hai mẹo dạng này cho mADM tăng 0.002 đến 0.005 và mAP tăng 0.01 đến 0.035 (mục 3.2). Không được dùng nếu ban tổ chức chưa xác nhận bằng văn bản. Chấm từng query một cách độc lập thì luôn hợp lệ.

---

## 2. Thí nghiệm đang chạy và ứng viên

Ký hiệu trong bảng: GIỮ, SỬA, BỎ. Mức nhiễu tham chiếu là sai số chuẩn ghép cặp của mADM, khoảng 0.0025, và độ lệch Avg@0.6 giữa hai run lặp, khoảng 0.0005 đến 0.0016.

| # | Thí nghiệm | Câu hỏi | Có đổi được quyết định không | Hiệu ứng kỳ vọng so với nhiễu | Kết luận |
|---|---|---|---|---|---|
| 1 | ViT-B query/linear, PETA, seed 1 và 2 | Nhiễu giữa các seed | Có. Nó đặt ngưỡng chấp nhận cho mọi so sánh | Không áp dụng | GIỮ. Chỉ ghép seed 1 với seed 2 cùng đợt, không ghép với run cũ. Dùng thêm làm nguyên liệu cho model soup (mục 3, R-S1) mà không cần train thêm |
| 2 | ViT-L 176 query và linear, PETA | Head nào cho ViT-L | Có, cho cả hai track. Mọi bằng chứng ngoài Avg@0.6 ủng hộ linear | Ở ViT-B, linear hơn query +0.005 mADM và +0.004 đến +0.007 AUC | GIỮ. Run query đồng thời là bản lặp thứ hai của q_L176, cho ước lượng nhiễu ở ViT-L |
| 3 | ViT-B query/linear, holdout Market và PA100k | Proxy nhiều miền | Có, nếu dùng chênh lệch ghép cặp trong từng fold | Không áp dụng | SỬA. Giữ cả hai vì đã chạy, nhưng fold PA100k chỉ có 18.7 nghìn ảnh train nên giảm trọng số. Trên Market, báo cáo Track 2 cả khi có và không có 6 thuộc tính màu không được gán nhãn |
| 4 | Query head với linear attention (elu+1) và sigmoid attention | Chuẩn hoá softmax trong head có quan trọng không | Không. Head chiếm khoảng 0.03% FLOPs nên không có lợi về CPU. 30 biến thể head trước đều nằm trong ±0.003 | Kỳ vọng khoảng 0, nằm trong nhiễu | BỎ mọi thí nghiệm nối tiếp. Nếu run chưa bắt đầu thì huỷ |
| C1 | Cắt ViT-L 224 còn 16 hoặc 18 block | Có giữ được lợi ích của 224 khi đủ thời gian không | Chỉ khi lợi ích của 224 là thật, mà mục 1.5 cho thấy chưa | Có thể âm, vì cắt 25 đến 33% độ sâu làm mất đúng phần đặc trưng sâu mang lại lợi ích của ViT-L | BỎ. Chờ R8 trước. Nếu 224 thắng thì ToMe là cách giảm chi phí ít tổn hại hơn |
| C2 | Dense distillation từ ViT-L 224 sang ViT-B 256 | ViT-B có học được đặc trưng của ViT-L không | Yếu. Ở cùng ngân sách CPU, đối thủ là chính ViT-L 176. Logit KD trước đó chỉ tăng 0.0017 Avg và 0.002 mADM | Khả năng đóng được 30 đến 50% khoảng AUC giữa B và L, khoảng 0.002 đến 0.003. Mức này vẫn chưa vượt ViT-L 176 | BỎ khỏi đợt này. Chỉ làm lại nếu số đo thời gian cho thấy ViT-L 176 không vừa ngân sách, hoặc cần một model ViT-B rẻ để ensemble |
| C3 | Track 2 với loss không có DeepMAR (BCE thuần) | Xác suất có hiệu chỉnh tốt hơn có giúp loglik không | Đã trả lời được mà không cần GPU | Mô phỏng sau train (đảo ngược đúng công thức của smoothing và DeepMAR) cho thấy bỏ DeepMAR làm mADM giảm 0.003 và mAP tăng 0.006. Bỏ smoothing làm mADM giảm 0.027 | BỎ. Smoothing 0.05 đang giới hạn mức phạt cho mỗi thuộc tính sai, và mADM cần đúng tính chất đó |
| C4 | Xác suất theo nhóm (softmax trong nhóm loại trừ nhau) lúc chấm điểm | Ràng buộc cấu trúc nhóm có giúp xếp hạng không | Đã trả lời mà không cần GPU | mADM giảm 0.005 đến 0.012 trên 5 model PETA và 3 gallery cùng miền. mAP và R1 chỉ tăng ở gallery cùng miền | BỎ ở bước chấm điểm. Thay bằng group CE lúc train (R1, R2), cách đã cho +0.016 |
| C5 | Temperature scaling | Làm mềm hay làm sắc xác suất | Có, không tốn GPU | Với T = 1.25, mADM tăng 0.002 đến 0.005 ở 4 trên 5 model query và giảm 0.002 ở linear. T fit trên miền nguồn làm mADM giảm 0.004 đến 0.009 | SỬA. Chọn T bằng trung bình chênh lệch ghép cặp trên các fold LODO, không fit trên val miền nguồn |
| C6 | Trộn loglik với kỳ vọng degree of match | Có thêm điểm cho ảnh khớp gần không | Có, không tốn GPU | Với β = 0.5, mADM tăng 0.003 (một model). Với β = 1 (chỉ kỳ vọng ndom), mADM giảm 0.045 | GIỮ ở dạng kiểm tra không tốn GPU trên mọi logits LODO |
| C7 | ViT-L 176 trên holdout Market và PA100k | Kết quả ViT-L có lặp lại ở miền khác không | Có, nhưng chỉ cho cấu hình cuối | Không áp dụng | SỬA. Chỉ chạy holdout Market, chỉ cho cấu hình thắng ở R1, R2 và run 2 |
| C8 | Thêm seed cho ứng viên cuối | Nhiễu của ứng viên cuối | Có | Không áp dụng | GIỮ. Các seed này đồng thời là nguyên liệu của soup và ensemble |

---

## 3. Thí nghiệm còn thiếu, xếp theo giá trị kỳ vọng cho điểm test ẩn

Chi phí GPU tính trên 2×RTX 5090 theo số của kế hoạch. ViT-B proxy mất khoảng 9 phút, ViT-L 176 khoảng 10 phút, ViT-L 224 khoảng 12 phút cho 5 epoch. Run train toàn bộ dữ liệu tính thêm khoảng 10%.

### 3.1. Track 2

**T2-A. Group CE trên ViT-L.** Run `q_gce` cho thấy group CE nâng mADM của ViT-B thêm 0.0164 và hạ NLL/ảnh từ 6.98 xuống 6.77, trong khi AUC chỉ đổi 0.001. Lý do toán học như sau. Hàm chấm loglik giả sử 40 thuộc tính độc lập và lấy tổng $\sum_k \log P(q_k\mid x)$. Trong một nhóm loại trừ nhau như 12 màu áo, mô hình sigmoid độc lập phân bố xác suất lên cả những cấu hình không bao giờ xảy ra (không màu nào, hoặc hai màu). Group CE thêm vào loss số hạng $-\log \mathrm{softmax}(z_g)_c$. Số hạng này ép các logit trong nhóm tách nhau theo đúng likelihood phân loại. Do đó tích các sigmoid gần với phân phối phân loại thật hơn, và tổng log-xác suất xếp ảnh khớp đúng lên trên ảnh lệch một màu. Điều này giải thích vì sao lợi ích xuất hiện ở NLL và mADM mà không ở AUC. Kết quả bác bỏ giả thuyết là ViT-L 176 cộng group CE không vượt ViT-L 176 thường quá 0.005 mADM ghép cặp trên PETA và Market. Chi phí là 10 phút mỗi run. Lưu ý rằng `args.json` của `q_gce` không còn ở máy local, nên cần tìm lại trọng số `--group-ce` trong lệnh đã chạy. Nếu không tìm được, chạy 1.0.

**T2-B. Số epoch và recipe.** Xem mục 1.1. Lý do là fine-tune lâu hơn làm model khớp thói quen gán nhãn của miền nguồn, nên log-likelihood trên miền mới xấu đi trước khi AUC kịp đổi. Ta so 3 epoch, 5 epoch và 10 epoch cộng `best.pt` cho ViT-L 176 trên PETA, đọc cả Avg raw, AUC, NLL và mADM. Kết quả bác bỏ là bản 10 epoch không kém bản 5 epoch ở cả NLL lẫn mADM. Chi phí khoảng 6 và 20 phút.

**T2-C. Ensemble và model soup.** Trung bình logits cho các mức tăng sau trên PETA.

| Tổ hợp | AUC | mADM | Track 1 Avg@0.6 |
|---|---|---|---|
| ViT-B query | 0.9007 | 0.6261 | 0.7961 |
| ViT-B query, 2 lần chạy | 0.9044 | 0.6315 | 0.7965 |
| ViT-B query + linear | 0.9099 | 0.6389 | 0.7956 |
| ViT-B group CE + linear | 0.9108 | 0.6465 | 0.7945 |
| ViT-L 160 linear | 0.8960 | 0.6335 | |
| ViT-L 160 linear + ViT-B group CE | 0.9090 | 0.6601 | |
| ViT-L 256 linear + ViT-B group CE | 0.9173 | 0.6783 | 0.8028 |

Lý do toán học là trung bình logits của các model có lỗi ít tương quan làm giảm phương sai của log-odds. Lợi ích đó vào thẳng NLL và AUC. Ensemble còn làm xác suất mềm hơn, đúng hướng mà mADM ưa (mục 3.2). Ensemble tốn thời gian CPU, nên có hai biến thể.

- *Model soup* (trung bình trọng số của các run fine-tune từ cùng một checkpoint pretrain, Wortsman et al. 2022) không tốn thêm thời gian CPU. Với lr 5e-5 và layer decay 0.75, các run gần như chắc chắn nằm cùng một vùng tối ưu, nên trung bình trọng số hợp lệ. Kết quả bác bỏ là soup của seed 1 và seed 2 không hơn trung bình hai seed. Kiểm tra này không cần train thêm vì seed 1 và seed 2 đang chạy.
- *Ensemble logits* dưới ngân sách CPU chỉ khả thi khi biết hằng số thời gian (mục 1.8) và kích thước gallery test của Track 2. Nếu gallery Track 2 nhỏ hơn nhiều so với 29 248 ảnh, ngân sách cho mỗi ảnh tăng theo tỉ lệ, và ViT-L 256 hoặc ensemble hai model có thể vừa. Bài nộp Track 2 đầu tiên ghi `queries` và `gallery images` vào log, nên nộp nó sớm chính là một thí nghiệm.

**T2-D. Hàm chấm điểm suy ra từ mADM.** Phần này không tốn GPU. Trước hết cần hiểu mADM thưởng gì. Trên PETA, mean dom thật của mỗi query khoảng 0.82, với khoảng phân vị 10 đến 90 là 0.78 đến 0.85. Do đó

$$\mathrm{ndom}(j\ \text{thuộc tính sai}) = \frac{1 - j/40 - 0.82}{1 - 0.82} \approx 0.86,\ 0.72,\ 0.58,\ 0.44\ \text{với}\ j = 1,2,3,4,$$

và chỉ về 0 khi j khoảng 7. Mỗi ảnh lệch 1 hoặc 2 thuộc tính đứng trước ảnh khớp đúng vẫn nhận 70 đến 86% điểm của một ảnh khớp đúng. Kỳ vọng của ADM xấp xỉ

$$\mathbb{E}[\mathrm{ADM}] \approx \frac{1}{R}\sum_i \frac{P(\text{khớp}_i)}{i}\sum_{j\le i}\mathbb{E}[\mathrm{ndom}_j].$$

Trong biểu thức này, mỗi ảnh đóng góp theo hai cách. Xác suất khớp đúng $P(\text{khớp}_i)$ quyết định vị trí nào được tính điểm. Kỳ vọng ndom quyết định ảnh đó cộng bao nhiêu điểm cho các ảnh khớp đứng sau. Loglik chỉ xếp theo thành phần thứ nhất, còn L1 xấp xỉ thành phần thứ hai. Dữ liệu cho thấy thành phần thứ nhất chiếm ưu thế (loglik 0.626 so với L1 0.579), nhưng thành phần thứ hai giải thích vì sao xác suất mềm hơn lại giúp mADM trong khi làm hại mAP. Các phép đo hôm nay như sau.

| Hàm chấm điểm (ViT-B query, PETA) | mADM | mAP |
|---|---|---|
| loglik (đang dùng) | 0.6261 | 0.2773 |
| loglik, T = 1.25 | 0.6309 | 0.2717 |
| loglik + 0.5 × log(Σ_j e_j ndom_j) (Poisson-binomial) | 0.6290 | 0.2743 |
| loglik sau khi đảo ngược smoothing | 0.5990 | 0.2826 |
| noise floor ε = 0.05 cho mọi thuộc tính | 0.6251 | 0.2631 |
| xác suất theo nhóm (softmax trong nhóm) | 0.6139 | 0.2744 |
| chuẩn hoá theo ảnh trên tập query, α = 0.25 (cần xác nhận luật) | 0.6273 | 0.2877 |

Cách tính số hạng thứ ba như sau. Gọi $m_k = q_k(1-p_k) + (1-q_k)p_k$ là xác suất thuộc tính k sai và $r_k = m_k/(1-m_k)$. Khi đó $P(\text{đúng } j \text{ thuộc tính sai}) = P_0\, e_j(r)$, với $e_j$ là đa thức đối xứng sơ cấp bậc j. Các tổng luỹ thừa $\sum_k r_k^m$ tính được bằng nhân ma trận, nên $e_1,\dots,e_4$ có được bằng đẳng thức Newton với khoảng 8 phép nhân ma trận Q×40×G. Trên CPU, chi phí này không đáng kể so với backbone.

Kết luận là các tinh chỉnh ở bước chấm điểm chỉ mang lại tối đa khoảng 0.003 đến 0.005 mADM. Mức này cùng cỡ với nhiễu ghép cặp, nhỏ hơn khoảng 3 lần so với group CE và nhỏ hơn khoảng 10 lần so với đổi backbone. Chỉ áp dụng một tham số (T hoặc β) khi nó tăng mADM trên cả ba fold LODO.

Về các thuộc tính mà một dataset không bao giờ gán nhãn, query luôn đặt chúng bằng 0, và từng query riêng lẻ không cho biết thuộc tính nào thuộc loại này. Cách hợp lệ duy nhất là làm hàm chấm bền với chúng bằng cách giới hạn mức phạt cho mỗi thuộc tính. Smoothing 0.05 hiện đã làm việc này. Bỏ hẳn các thuộc tính đó (+0.004 mADM trên Market) đòi hỏi thống kê trên toàn tập query, nên thuộc nhóm cần xác nhận luật ở mục 1.9.

**T2-E. Loss riêng cho Track 2.** BCE là một proper scoring rule (tạm dịch: quy tắc chấm điểm đúng đắn) cho từng thuộc tính. Nó tối đa hoá log-xác suất của chính vector nhãn của mỗi ảnh, nhưng không trực tiếp tối ưu so sánh giữa các ảnh dưới cùng một query, mà xếp hạng cần đúng so sánh này. Đề xuất cụ thể là thêm một số hạng InfoNCE trong batch. Ta gọi $s_{ij} = \sum_k [y_{jk}\log\sigma(z_{ik}) + (1-y_{jk})\log(1-\sigma(z_{ik}))]$ là loglik của ảnh i dưới vector nhãn của ảnh j. Loss là $-\log \sum_{i: y_i=y_j} e^{s_{ij}/\tau} / \sum_i e^{s_{ij}/\tau}$ cho mỗi j, với trọng số 0.1 đến 0.3 và τ = 1. Để khớp với cách mADM cho điểm ảnh khớp gần, mẫu âm có thể nhân trọng số $(1 - \mathrm{ndom}_{ij})$. Kết quả bác bỏ là mADM ghép cặp trên PETA không tăng quá 0.005. Chi phí là 9 phút trên ViT-B. Đây là ưu tiên thấp vì NLL đã tương quan rất sát với mADM, nên lợi ích chính có lẽ đã đến từ T2-A.

**T2-F. Độ phân giải và ToMe.** Chỉ làm sau R8, xem mục 3.3.

### 3.2. Track 1

**T1-A. Đồng bộ recipe (5 epoch, `last.pt`).** Lý do và kết quả bác bỏ giống T2-B. Bước này không cần run riêng vì dùng chung R6 và R7. Nếu R7 (10 epoch) kém R5 epoch, bài nộp Track 1 tiếp theo dùng 5 epoch.

**T1-B. Model soup và WiSE-FT.** `wise.py` đã có sẵn nhưng chưa từng được chạy trên proxy (không có trong `experiments.json`). WiSE-FT nội suy trọng số $\theta(\alpha) = (1-\alpha)\theta_{\text{pretrain}} + \alpha\theta_{\text{ft}}$. Trên các benchmark chuyển miền, nó thường giữ được độ bền của model pretrain mà ít mất độ chính xác trong miền. Cả soup lẫn WiSE-FT đều không tốn thêm thời gian CPU lúc chạy bài. Chi phí chỉ là vài lượt suy luận trên PETA và Market. Kết quả bác bỏ là không giá trị α nào trong {0.6, 0.7, 0.8, 0.9} làm tăng cả raw Avg lẫn AUC so với α = 1 trên hai fold. Một lưu ý là head được giữ nguyên trong khi backbone bị nội suy, nên α nhỏ có thể làm lệch phân phối đặc trưng mà head nhận.

**T1-C. Chọn head.** Đây là run số 2 đang chạy. Nếu ViT-L linear có AUC và raw Avg không thấp hơn query trên PETA và Market, Track 1 nên chuyển sang linear, vì leaderboard ViT-B đã xếp linear trên query.

**T1-D. Ensemble trong ngân sách CPU.** Hai model ViT-B cho AUC 0.9099 đến 0.9108, bằng ViT-L 176 (0.9102) với thời gian CPU tương đương (2 × 49.5 ms). Tuy nhiên Avg@0.6 của các ensemble này phẳng (0.7945 đến 0.7965). Theo nguyên tắc AUC của báo cáo v5, ensemble đáng làm. Theo Avg@0.6 thì không. Đây là một phép thử trực tiếp cho chính nguyên tắc đó, nhưng nó tốn một lượt nộp và không thể vượt ViT-L 176. Tôi đề xuất không nộp ensemble ViT-B. Thay vào đó, sau khi có hằng số thời gian thật, thử ViT-L 176 cộng một model rẻ (ViT-B 160×80 hoặc ViT-S) nếu tổng thời gian không quá khoảng 105 ms/ảnh.

**T1-E. Hệ số offset chọn bằng trung bình ba fold.** Fit offset trên hai miền nguồn, áp lên miền còn lại, và chọn hệ số tối đa hoá trung bình ba fold thay vì chỉ PETA. Phép tính này chỉ tốn CPU và dùng logits của các run nhiều miền đang chạy. Nó trả lời đúng câu hỏi "hệ số nào chuyển được sang miền mới", và câu hỏi này hiện chỉ có một điểm dữ liệu.

### 3.3. Efficient attention, token merging và nơi chúng có tác dụng

Chi phí mỗi block của ViT-L (d = 1024) với N token gồm hai phần. Phần tuyến tính (QKV, projection và MLP) tốn khoảng $24Nd^2$ FLOPs. Phần attention ($QK^\top$ và $AV$) tốn khoảng $4N^2d$ FLOPs. Tỉ lệ giữa hai phần là $N/(6d)$.

| Kích thước ảnh | N (patch + 5 token tiền tố) | Phần attention / phần tuyến tính |
|---|---|---|
| 176×96 | 71 | 1.2% |
| 224×112 | 103 | 1.7% |
| 256×128 | 133 | 2.2% |

Tỉ lệ thời gian đo được giữa 224 và 176 (149/98 = 1.52) gần với tỉ lệ số token (103/71 = 1.45). Điều này xác nhận chi phí tỉ lệ tuyến tính theo N. Có ba hệ quả.

- Linear attention trong backbone chỉ có thể tiết kiệm tối đa khoảng 2% thời gian, đổi lại phải bỏ attention softmax đã pretrain. Không đáng làm.
- Linear attention trong head (40 query nhân N token) chiếm khoảng 0.03% FLOPs. Run số 4 vì vậy chỉ thử một inductive bias (tạm dịch: thiên kiến cấu trúc), không thử tốc độ.
- Cách duy nhất làm giảm chi phí là giảm N. ToMe (token merging, tạm dịch: gộp token) gộp r cặp token giống nhau nhất ở mỗi block. Chi phí tỉ lệ với $\sum_l N_l$, và với 24 block thì $\bar N \approx N - 11.5r$. Muốn ViT-L 224 đạt khoảng 100 ms/ảnh, ta cần tỉ lệ 0.67 nếu 224 tốn 149 ms (r ≈ 3, còn khoảng 31 token ở block cuối), hoặc 0.85 nếu 224 tốn 118 ms (r = 1). Do đó cấu hình ToMe phụ thuộc hoàn toàn vào hằng số ở mục 1.8. Query head phải đọc các token đã gộp với bias $\log(\text{kích thước token})$ (proportional attention, tạm dịch: attention tỉ lệ kích thước), nếu không các vùng nền đã gộp sẽ bị đánh giá quá thấp hoặc quá cao. Một kiểm tra rẻ là áp ToMe khi suy luận lên checkpoint 224 của R8, không train lại, để đo mức sụt giảm trước khi trả tiền cho R9.

---

## 4. Hàng đợi ưu tiên (tối đa 20 run) và quy tắc nộp bài

### 4.0. Bước không tốn GPU, làm trước

- Z1. Kiểm tra khoá đầu ra của Track 2 (`similarities` hay `distances`) trên trang Codabench hoặc scoring program. Nếu chưa chắc, trả `{"distances": -sim}` theo đúng code mẫu. Sau đó nộp `t2_vitl176_loglik_v1` để có điểm mốc. Đọc log để lấy số query, số ảnh gallery và ms/ảnh.
- Z2. Đọc log ingestion của bài Track 1 v9 để lấy ms/ảnh thật của ViT-L 176 trên server, rồi thống nhất một hệ số quy đổi duy nhất.
- Z3. Thêm NLL/ảnh vào `scores.json`. Chấm mọi logits LODO với T ∈ {1, 1.25, 1.5} và β ∈ {0, 0.25, 0.5}.
- Z4. Soup seed 1 với seed 2 (ViT-B query và linear, PETA), và WiSE-FT α ∈ {0.6, 0.7, 0.8, 0.9} trên ViT-L 176 PETA. Chỉ cần suy luận.
- Z5. Chọn hệ số offset bằng trung bình ba fold (T1-E).

### 4.1. Track 2 (11 run)

| Thứ tự | Run | Cấu hình | GPU | Điều kiện chạy |
|---|---|---|---|---|
| R1 | ViT-L 176 query + group CE | holdout PETA, 5 ep, seed 1, `--group-ce` như q_gce (hoặc 1.0) | 10' | luôn chạy |
| R2 | ViT-L 176 linear + group CE | như R1 | 10' | luôn chạy |
| R3 | ViT-B query + group CE | holdout Market, 5 ep | 8' | luôn chạy (lặp lại hiệu ứng group CE ở miền thứ hai) |
| R4 | ViT-B query + group CE | holdout PETA, seed 2 | 9' | luôn chạy (đo nhiễu của hiệu ứng +0.016) |
| R5 | ViT-L 176, head thắng ở R1/R2/run 2 | holdout Market | 9' | sau R1, R2 |
| R8 | ViT-L 224, head thắng, có/không group CE theo R1 | holdout PETA, seed 1 | 12' | sau R1, R2. Chạy kèm ToMe khi suy luận (không train) trên checkpoint |
| R9 | ViT-L 224 + ToMe trong lúc train, r theo Z2 | holdout PETA | 12' | chỉ khi R8 hơn ViT-L 176 cùng cấu hình ≥ 0.005 mADM và NLL thấp hơn |
| R10 | ViT-B query + group CE + InfoNCE loglik (trọng số 0.2) | holdout PETA | 9' | ưu tiên thấp, chạy khi GPU rảnh |
| R11 | ViT-B 224×112 + group CE (thành viên ensemble) | holdout PETA | 8' | chỉ khi Z1/Z2 cho thấy còn ≥ 35 ms/ảnh ngoài ViT-L 176 |
| R12 | Ứng viên Track 2 cuối, toàn bộ train | 5 ep, `last.pt`, seed 1 | 11' | sau R1 đến R5 |
| R13 | Như R12, seed 2 | 5 ep, `last.pt` | 11' | để làm soup với R12 |

### 4.2. Dùng chung cho cả hai track (2 run)

| Thứ tự | Run | Cấu hình | GPU | Câu hỏi |
|---|---|---|---|---|
| R6 | ViT-L 176 query | holdout PETA, 3 ep, `last.pt` | 6' | số epoch tối ưu ngoài miền |
| R7 | ViT-L 176 query | holdout PETA, 10 ep, ghi held-out mỗi epoch, chấm cả `last.pt` lẫn checkpoint mà `best.pt` cùng miền sẽ chọn | 20' | recipe đang nộp có làm mất điểm không |

### 4.3. Track 1 (4 run)

| Thứ tự | Run | Cấu hình | GPU | Điều kiện chạy |
|---|---|---|---|---|
| R14 | ViT-L 176 head thắng (run 2), không group CE | holdout Market | 9' | sau run 2 |
| R15 | ViT-L 176 head thắng | holdout PETA, seed 2 | 10' | đo nhiễu ở ViT-L, làm nguyên liệu soup |
| R16 | Ứng viên Track 1 cuối, toàn bộ train | 5 ep (hoặc theo R6/R7), `last.pt`, seed 1 | 11' | sau R6, R7, R14 |
| R17 | Như R16, seed 2 | | 11' | soup với R16 |

Tổng cộng 17 run, khoảng 180 phút GPU, chia ra khoảng 1.5 giờ trên hai card. Nếu R9 hoặc R11 không đủ điều kiện, tổng giảm còn khoảng 160 phút.

### 4.4. Quy tắc quyết định bài nộp

**Track 1.**
1. Ứng viên chỉ thay mốc (ViT-L 176 query, 0.7857) khi, trên trung bình chênh lệch ghép cặp của PETA và Market (mỗi fold ít nhất hai seed hoặc một soup hai seed), raw Avg và AUC cùng tăng ít nhất 0.003, và Avg sau offset (hệ số chọn theo Z5) không giảm.
2. Thời gian ước lượng trên server, theo hằng số của Z2, không quá 52 phút. Gói nộp phải cho ra xác suất khớp logits lúc train trên một mẫu val.
3. Chênh lệch leaderboard dưới 0.003 được coi là hoà. Khi hoà, giữ cấu hình rẻ hơn và đơn giản hơn. Leaderboard chỉ giữ bài gần nhất, nên lượt nộp cuối của mỗi ngày luôn là bài tốt nhất đã biết.

**Track 2.**
1. Tiêu chí chính là trung bình chênh lệch mADM ghép cặp trên các gallery holdout PETA và Market. Tiêu chí phụ là NLL/ảnh. Fold PA100k chỉ dùng để phủ quyết khi nó giảm quá 0.005.
2. Ứng viên thay mốc khi chênh lệch trung bình ≥ +0.005 và không fold nào giảm quá 0.003.
3. Tham số của hàm chấm (T, β) được chọn trên LODO, không trên val cùng miền. Một tham số chỉ được bật khi nó tăng mADM trên mọi fold.
4. Không dùng thống kê tính chéo nhiều query (mục 1.9) khi chưa có xác nhận bằng văn bản từ ban tổ chức.
5. Track 2 là một cuộc thi riêng, nên model của Track 2 không cần trùng model Track 1. Cấu hình Track 2 dự kiến, nếu R1 đến R5 xác nhận, là ViT-L (head thắng) cộng group CE, 5 epoch, soup của hai seed, chấm bằng loglik với T chọn trên LODO.

---

## Phụ lục. Nguồn của các số mới

- Logits: `Upar-Challenge-Track-1/runs/*/held_val_logits_last.npz` (query heads, `last.pt`), `ft_PETA_dinov3/holdout_logits.npz`, `teacher_PETA_dinov3L/val_logits.npz` (linear, `best.pt` = epoch cuối), `L176q_train/val_logits.npz` (cùng miền).
- Metric: `upar.retrieval.retrieval_metrics`. Offset: `calibrate.fit_offsets`, tái tạo đúng các số Avg@0.6 trong `experiments.json` (ví dụ 0.7961, 0.7956, 0.7904).
- Sai số chuẩn ghép cặp: bootstrap 1000 lần trên 1763 query PETA.
- Mô phỏng bỏ DeepMAR và smoothing sau train dùng nghiệm đúng của loss có trọng số và nhãn làm mượt, $q^* = s/2 + (1-s)\,\sigma(\mathrm{logit}\,\pi + (1-2p))$, với p là prior của thuộc tính trên miền nguồn.
