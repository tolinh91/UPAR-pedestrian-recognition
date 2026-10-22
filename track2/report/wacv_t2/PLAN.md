# Track 2 paper: writing plan and terminology

Rules: skill `upar-paper-rules` (all of B, C, D, E, G). Vietnamese copy `main_vi.tex` first; English `main.tex` only after
the user approves the Vietnamese copy. Numbers come only from `analysis/*.json` and `analysis/knn_logs_2026-10-11.txt`.

## What is worth a paper (frank assessment, 2026-10-18)

| Part | New? | Evidence | Verdict |
|---|---|---|---|
| Rank by log-likelihood of the exact query instead of L1 | idea known (DAP, PRP); not used by the organizers' sample baseline | both folds, six metrics; all 49 development models | strong, simple |
| Recalibration after training does not transfer | answers the open question of the 2024 report | both folds; 48 of 49 development models | strong, the analysis core |
| kNN smoothing of gallery logits over DINOv3 neighbours | DBA applied to attribute logits; query stays binary | held-out PETA + source-dataset val galleries; feature ablation; probability-neighbour ablation on both folds | medium: Market held-out fold missing (fold checkpoints lost) |
| DINOv3 + CLIP ensemble, text score, WiSE-FT, student | known components (B17: not contributions) | both folds (student: PETA only) | system detail |
| Hidden test 0.6783 | | three submissions | no 2027 baseline score available |

## One-sentence contribution

For attribute-based person retrieval on unseen cameras, the outputs of fine-tuned foundation encoders should be ranked by
the likelihood of the exact query and smoothed over appearance neighbours in the gallery, while recalibration fitted after
training on the source datasets lowers retrieval quality although the miscalibration it targets is real.

## Narrative spine

| # | Step | Question | Evidence |
|---|------|----------|----------|
| 1 | Problem | Why is retrieval on unseen cameras hard? | task, domain shift |
| 2 | Gap | Earlier editions never beat the baseline; organizers blame over/underconfidence | reports 2023, 2024 |
| 3 | Q1 | Given attribute probabilities, which score ranks the gallery well? | Tab. LL vs L1, Fig. example |
| 4 | Q2 | Is miscalibration the obstacle, can it be fixed after training? | Tab. calibration, Fig. scale/bias |
| 5 | Recognizers | Since post-hoc fixes fail, better probabilities: two foundation encoders | Tab. encoders |
| 6 | Q3 | Can the gallery's appearance structure help? | Tab. kNN |
| 7 | Transfer | Hidden test | Tab. hidden test |
| 8 | Limits | | Limitations |

## Section outline

1. Introduction (no numbers, no section pointers): task and shift; foundation encoders and the usual downstream recipe;
   earlier editions and the organizers' hypothesis; three questions; system; evaluation framework; three contributions.
2. Related work: attribute-based person retrieval (incl. the challenge), calibration under shift, re-ranking with the
   gallery, foundation encoders and ensembles.
3. Method: task and mADM (Eq. 1, 2), recognizers (Fig. 1), likelihood score (Eq. 3), kNN smoothing (Eq. 4).
4. Experiments: protocol; scoring; calibration; encoders; kNN smoothing; hidden test.
5. Limitations. 6. Conclusion.

## Terminology (one name per concept)

| Use (EN) | Dùng (VI) | Meaning | Do not use |
|---|---|---|---|
| query | truy vấn | 40-dim 0/1 attribute vector | description, attribute query (that is the query head's learned vector) |
| gallery | gallery | the images ranked for a query | database, kho ảnh |
| exact match | ảnh khớp đúng | gallery image whose 40 labels equal the query | correct match, relevant image |
| likelihood score, s_LL | điểm likelihood | log-probability of the exact query, Eq. 3 | log-likelihood ranking (ok once), LL |
| L1 distance | khoảng cách L1 | sum_k abs(q_k - p_k), the sample baseline's ranking | L1 score |
| recognizer | — | avoid; say "the DINOv3 model", "the CLIP model" | member, recognizer |
| DINOv3 model / CLIP model | mô hình DINOv3 / mô hình CLIP | encoder + query head fine-tuned alone | H+, member |
| DINOv3 + CLIP ensemble | ensemble DINOv3 + CLIP | logit mean of the two models | pair, separate ensemble |
| student | student | DINOv3 ViT-L/16 distilled from DINOv3-L and CLIP-L teachers | stuL |
| encoder | encoder | pretrained ViT image encoder | backbone |
| query head | query head | 40 attribute queries + cross-attention + per-attribute classifiers | head (alone) |
| recalibration | hiệu chỉnh xác suất | per-attribute Platt map sigma(a_k z + b_k) after training | calibration (for the act) |
| oracle recalibration | hiệu chỉnh oracle | map fitted on held-out labels | |
| source recalibration | hiệu chỉnh trên dữ liệu nguồn | map fitted on validation images of the two source datasets | |
| calibration term | thành phần calibration | NLL removed by the map fitted on the same images (Kull and Flach) | calibration error |
| scale a_k, bias b_k | hệ số scale, bias | the two parameters of the map | slope, offset |
| kNN smoothing | kNN smoothing | Eq. 4 | re-ranking (only when quoting the rules), DBA (only in related work) |
| appearance feature | đặc trưng ngoại hình | L2-normalised mean of the DINOv3 model's patch tokens | embedding |
| neighbour | láng giềng | one of the k most similar other gallery images | |
| source datasets / held-out dataset / fold | dataset nguồn / held-out dataset / fold giữ ngoài X | as in Track 1 (VI copy approved there) | unseen domain, dataset giữ lại |
| development models | các mô hình của giai đoạn phát triển | the 49 earlier models with saved held-out logits | runs |
| points | điểm | metric x 100 | |
