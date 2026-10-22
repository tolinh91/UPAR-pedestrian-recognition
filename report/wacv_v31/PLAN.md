# Track 1 paper (v31): writing plan and terminology

Guidance: `~/.claude/skills/ml-paper-writing` (Orchestra Research, MIT): one-sentence contribution, Farquhar 5-sentence
abstract, Gopen & Swan (old before new, one paragraph = one point), consistent terminology (Steinhardt), every
experiment states the claim it tests, mandatory Limitations section.

## One-sentence contribution

Two foundation encoders with different pretraining (DINOv3, CLIP) transfer better to unseen pedestrian datasets when
they are fine-tuned together with a joint attribute query than when separately fine-tuned encoders are averaged,
and the combination cannot be learned after training on frozen encoders.

## Narrative spine (each step answers the question raised by the previous one)

| # | Step | Question it answers | Evidence |
|---|------|---------------------|----------|
| 1 | Problem | Why is cross-domain PAR hard? | UPAR setting |
| 2 | Observation | Why two encoders, and why different ones? | Tab. 1: 2nd copy +0.04..0.10, different encoder +0.4..0.6 |
| 3 | Gap | What is wrong with averaging? | fixed equal mix, no co-adaptation |
| 4 | Failed fix | Can the mix be learned afterwards? | Tab. 2: no fusion head beats the average; they gain on source images |
| 5 | Method | Learn the mix during training | Fig. 1, Sec. 3 |
| 6 | Main result | Does joint training beat the separate ensemble? | Tab. 3 |
| 7 | Robustness | Is the small gain real? | Tab. 4: 4/4 runs win on both folds |
| 8 | Mechanism | How does the joint query head help? | Fig. 2 (per attribute), Fig. 3 (routing) |
| 9 | Transfer | Does it hold on the hidden test? | Tab. 5 |
| 10 | Limits | What is not shown? | Limitations |

## Section outline

1. Introduction: problem, observation (separate ensemble defined), gap + failed fix, approach (+ C2T-Net positioning), 3 falsifiable contributions.
2. Related work: PAR, earlier UPAR editions, combining encoders, imbalanced attributes.
3. Method: encoders, single-encoder query heads, joint query head, training and decision rule (Eq. 1, 2).
4. Experiments (order mirrors the introduction): 4.1 protocol, 4.2 why two different encoders (Tab. 1), 4.3 can the combination be learned after training (Tab. 2), 4.4 joint model vs separate ensemble (Tab. 3), 4.5 does the gain repeat (Tab. 4), 4.6 how the joint query head uses the two encoders (Fig. 2, 3), 4.7 hidden test (Tab. 5).
5. Limitations. 6. Conclusion.

## Terminology (one name per concept)

| Use | Meaning | Do not use |
|-----|---------|------------|
| encoder | pretrained ViT image encoder | backbone, network |
| DINOv3 | DINOv3 ViT-H+/16; full name only where the encoders are introduced | H+, DINO, DINOv3-H, DINOv2 (for cited evidence that used DINOv2, say "self-supervised encoder") |
| CLIP | OpenAI CLIP ViT-L/14 image encoder | CLIP-L |
| encoder pass | one forward pass of one encoder | |
| query head | module: 40 attribute queries, cross-attention, per-attribute classifier | head (alone), member head |
| DINOv3 query head, CLIP query head | query head reading the tokens of one encoder | member, member head, H+ query head |
| single-encoder query heads | the DINOv3 and CLIP query heads together | members, DINOv3 and CLIP heads |
| joint query head | query head reading the tokens of both encoders | joint head, fusion head |
| attribute query | learned query vector of one attribute | |
| attention head | one of the 8 heads of multi-head cross-attention | head (alone) |
| attribute vector | output of a query head for one attribute, before its classifier | feature |
| separate ensemble | DINOv3 and CLIP, each fine-tuned alone with its own query head, logits averaged | pair, two-encoder average, averaged separate encoders |
| UniQuery | the model: DINOv3 and CLIP fine-tuned in one network with three query heads, logits of the three averaged (name agreed 2026-10-21, teammate draft) | joint model, dual model, v31 |
| fusion head | small module trained on top of the separate ensemble (its fine-tuned encoders kept fixed) | joint head |
| frozen | encoder never fine-tuned: pretrained weights kept (Tab. 4, BRAVE); also the first eight blocks | frozen for the fixed fine-tuned encoders under fusion heads |
| source datasets / held-out dataset | the two training datasets / the third dataset of a fold | unseen domain (in Sec. 4) |
| fold | one leave-one-dataset-out split, named by its held-out dataset | |
| run | one training of one configuration with one seed | |
| logit-adjusted loss, shift s_a, tau | Eq. 1 | prior offset, post-hoc offset |
| Challenge_Avg, points | harmonic mean of mA and instance F1; points = x100 | score, accuracy |
| balanced accuracy | (TPR + TNR) / 2 of one attribute; mA is its mean | |

## Additions (2026-10-17)

- New Sec. 4.5 "Must the encoders be fine-tuned, and which ones?" with Tab. 4 (frozen H+ and CLIP; C-RADIOv4-H and SigLIP2-L swaps), placed between the main comparison and the repetition check. Related work now cites BRAVE (frozen encoders merged by learnable queries) and Cambrian-1; claim 2 adds "with frozen encoders it loses 1.0 point".
- Terms: "frozen" = all encoder parameters fixed (no block fine-tuned); "C-RADIOv4-H", "SigLIP2-L" as written in Tab. 4.

## Review pass (2026-10-17, conference-reviewer skill)

- Title uses the term: "A Joint Query Head over DINOv3 and CLIP" (was "Joint Attribute Query").
- Spelling: American (color, distill, labeling).
- No "backbone", "score"/"accuracy" for Challenge_Avg, "thus", "hatched", "tailored", "overstate".
- Each result number appears once in text (18.7k only in Protocol; 0.4-0.6 only in Sec. 4.2 text; Limitations without numbers).
- Claims scoped: fusion heads lose "in the mean" (per-attribute weights win Market by 0.01); swaps lose "on PETA" (C-RADIOv4-H + CLIP wins Market); "in our setting" for frozen-encoder findings.

## Restructure (2026-10-21, user: imitate the teammate's UniQuery draft, /media/tan/F/RWS2027/main.pdf)

- Title "UniQuery: ..."; abstract = PAR value -> cross-domain difficulty -> Track 1 solution -> separate ensemble (fixed mix, never co-trained) -> UniQuery -> LODO + ablations -> hidden test 0.8001.
- Intro: application -> task -> challenges -> Track 1 -> question (does a per-attribute mix learned after / during training transfer?) -> UniQuery -> evaluation discipline -> 3 finding bullets. No numbers.
- Related work: cross-domain PAR (+ UPAR editions, AttriVision, ASRL), target-domain adaptation vs DG (AdaGPAR, LDCD), foundation encoders + unified learning (UniHCP, HumanBench, CLEAR), combining encoders (+ C2T-Net). Each paragraph ends with a positioning sentence.
- Method opens with an input -> output overview; latency moved to the end of the decision-rule paragraph.
- Tab. var: + (h)/(i) region-slot joint query head (RSCG v2, seeds 1/2), (j) noise-adaptation layer. Tab. test: + final leaderboard entry (separate ensemble + post-hoc offset 0.42 (L_a - d_a), 0.8011), stated as post-hoc.

## Related work rewrite (2026-10-21, user: "khai thác embedding ảnh và văn bản của CLIP để tổng quát hoá tốt hơn khai thác thế nào. Rõ ra cho các công trình liên quan")

- Every cited work is now described by how it works, each detail checked against the paper or code (sources in /tmp/rw_dl at the time): AttriVision (CLIP ViT-B/32, both encoders fine-tuned, one sentence per attribute value, cosine prediction, focal-weighted contrastive FCE loss), Jia et al. (identity overlap in PETA/RAP -> PETA_ZS/RAP_ZS), UPAR (40 shared attributes, leave-one-out protocol), ASRL, AdaGPAR (training-free TTA, global + attribute memory banks), LDCD-PAR (UDA, GRL, discriminator on global and part features), UniHCP (task queries, one query per attribute for PAR), HumanBench/PATH (shared encoder, task projectors, dataset heads), CLEAR (GPT pseudo-description, adapters, frozen PAR net), Fang et al., Eyes Wide Shut (CLIP-blind pairs, interleaved MoF), Cambrian-1, Krogh-Vedelsby decomposition, soups, WiSE-FT, AM-RADIO (per-teacher MLP heads), C-RADIOv4 (SigLIP2/DINOv3/SAM3), stacking, BRAVE (frozen encoders, MEQ-Former), C2T-Net, Query2Label, ML-Decoder, DeepMAR (per-attribute weight from the positive ratio; our w+ = exp(1-p), w- = exp(p) matches the DeepMAR author's PyTorch code), logit adjustment.
- Correction: Cambrian-1 does NOT keep its encoders frozen (its Finding 4: unfreezing helps); only BRAVE is described as frozen. Cambrian-1 reports that SVA attention per encoder changes with the image category, the closest published observation to our per-attribute routing result.
- First rewrite (8 paragraphs, 1361 English words) was rejected as far too long ("dài quá lố r"). Final: the old 6-paragraph structure, one sentence per cited work with its core mechanism only; then "Viết gọn lại" on the foundation-encoder and combining paragraphs (290 -> 205 words): 597 English words in total (old 603), cap 750 enforced by lint; the Conclusion is back on page 7 in both copies.

## CLIP text score: plain explanation + zero-shot evidence (2026-10-22, user: "phải visualize ra thì mới chứng minh được")

- Method (propose + overview only, user 2026-10-22: "phần đó chỉ đề xuất và nói tổng quan làm gì, còn chứng minh nó nằm ở phần sau"): the score is the similarity to the attribute sentence minus the similarity to `a person`, then the formula; idea figure `fig:zsidea` (figs/text_idea.pdf: hat attribute, PETA val, every image a point, two real crops: no-hat row 4268 above in raw similarity but below the diagonal, hat row 2972).
- Experiments: new question paragraph "Does the CLIP text score help?" after "Must the encoders be fine-tuned?": run (k) moved here from Consistency, plus the zero-shot numbers and `fig:zsauc` (figs/text_auc.pdf). Figure script: analysis/make_zs_fig.py (runs locally from remote_results/zs_text/zs_scores.npz).
- Data: analysis/zs_viz.py on a rented RTX 5090 (2026-10-22), timm CLIP-L/14 at 224x112 + the exact v31 text file (remote_results/duf2/clip_l14_text.pt), all val images of PETA (6963) and Market-1501 (16458) + first 3000 PA100k (reproduces the old make_clip_text log 0.7494 / 0.7179 / 0.7720 exactly). Results: analysis/zs_auc.json; per-image scores in remote_results/zs_text/zs_scores.npz; figure script analysis/make_zs_fig.py.
- Numbers used in the text: mean zero-shot AUC 0.68 -> 0.76 (PETA), 0.67 -> 0.71 (Market); rises for 37/40 and 29/34 attributes (colours 22/24 and 15/18); drops > 0.05 only for Age-Adult (both) and Hair-Length-Short (Market); mean correlation between similarity to an attribute sentence and to `a person` 0.74 (PETA), 0.65 (Market).
- Built 2026-10-22: 9 pages, Conclusion on page 7 in both copies; Fig. 3 (idea) top of page 4, Fig. 6 (AUC) page 7.

## Joint tokens: linear layer only (2026-10-22, user: "thì nói mie là lớp linear đi ... vẽ cái hình nữa với fact check")

- The code adds a learned source embedding (self.src) after each per-encoder nn.Linear, but nn.Linear already has a bias (ph has 1280*1024 + 1024 params), so W t + b + e = W t + (b + e): the source embedding folds into the bias. Text, Figure 1 and its caption now say "one linear layer per encoder (W_D, W_C), then concatenation"; no e_D / e_C anywhere.
- Figure 1 fact-check against code and v31_min.json (all match): DINOv3 103 tokens -> 99 after dropping 4 registers (heads.py: CLS + patches), CLIP 129, joint 228 x 1024; depth 32 / 24 with --freeze-blocks 8 -> 24 / 16 fine-tuned; hat logits h 5.139, j 4.654, c 6.414 (includes text 1.688); mean 5.402, sigmoid 0.996, ground truth hat = 1 (39/39 labelled attributes correct on this image); query heads 1 layer, 8 attention heads, d = 1024 (DINOv3 head 1280).

## Table of retrainings and variants removed (2026-10-22, user: "bảng này k có ích cho phân tích")

- tab:var dropped from both copies. Kept in text: the four UniQuery training runs (two seeds, small training changes) at 84.42-84.49, winning 8/8 comparisons on PETA and 7/8 on Market (Consistency); run (k) as "retraining UniQuery without the text score" (-0.15 / -0.05). Variants (e)-(j) (mid tokens, text-initialized queries, flatter colour attention, region slots, noise adaptation) are no longer reported; slotattn and noiseadapt drop out of the references.
- Labels that pointed to the table were rewritten: pick-figure caption ("two seeds of a UniQuery variant with a few small training changes"), hidden-test rows "UniQuery (submitted)" / "UniQuery, variant".
- Same pass fixed the hidden-test paragraph: the 0.42 offset is "the post-hoc rule that tau of UniQuery is matched to", not "the per-attribute shift that maximizes balanced accuracy".

## Encoder choice, hypothesis and text-score ablation (2026-10-22, user: "lựa máy cái chứng minh sao chọn 2 cái mô hình này mà k phải mô hình khác đi, rồi giả thuyết vì sao, với lại chứng minh nhánh thêm text, ablation")

- "Why two different encoders?" replaced by two paragraphs, "Why DINOv3 and CLIP?" and "Why do DINOv3 and CLIP help each other?"; the tab:two run description moved into its caption; the SigLIP2/C-RADIOv4 swap sentence moved from the fine-tuning paragraph to the choice paragraph.
- New tab:partner (analysis/partner_study.py, PETA fold, kdhp teacher recipe = weighted BCE, 3 epochs, freeze 8, one run each, DINOv3 = kdhp P_dinoHp224 = rare HB_s1, 80.47 alone). CLIP row = five runs of the same recipe (kdhp P_clipL224, C_s2; rare B_s1-B_s3): alone 80.56-80.73, with DINOv3 81.04-81.23. Only AIMv2-L (81.07) falls in that range; the other 10 partners are lower. None of the 66 pairs without DINOv3 exceeds 80.95 (CLIP + EVA02-CLIP). corr(partner alone, pair) = 0.86, corr(overlap, pair) = 0.14. Caveat in the text: recipe tuned on DINOv3 and CLIP. MobileCLIP2 reference added (arXiv 2508.20691, checked via the arXiv API).
- Error overlap (analysis/encoder_overlap.py, partner_study.py): share of one model's wrong (image, attribute) cells that the other also gets wrong, mean of both directions, plain threshold. tab:two weighted-BCE runs: same encoder 88.7-92.8% (PETA DINOv3 x2 90.6, CLIP x2 88.7; Market 92.8 / 90.7), DINOv3 vs CLIP 72% PETA / 77% Market. Partner matrix (PETA): DINOv3 vs each partner 66.3-72.1%, two non-DINOv3 encoders 68.2-87.0% (mean 73.4; SigLIP2 So400m /14 vs /16 87.0). Per attribute (seed-2 ensemble of tab:main, HL32/TL32): PETA DINOv3 > CLIP by > 1 pt on 12, CLIP > DINOv3 on 10; Market 12 vs 2. The lint's B11 rule rejects "DINOv3 learns from images alone", so the paragraph only says the encoders differ in pretraining.
- Text-score ablation (analysis/text_ablation.py, DU32 vs NT32, seed 1): mean of heads -0.148 PETA / -0.051 Market without the score; CLIP query head -0.203 / +0.067; DINOv3 query head (never reads the score) -0.096 / -0.018; joint +0.081 / -0.021. An identical rerun of DU32 on PETA (duf2) moved the mean 0.02 and single heads up to 0.38. Written as: drops of the size of the seed gap of the independent ensemble, CLIP head gain on PETA only, so the post-training evidence is weak; the zero-shot AUC part stays. Per-head absolute numbers are not printed because tab:main's per-head rows come from another evaluation of run (a).

## Consistency table, new Fig. pick, merged encoder paragraph, clearer second half (2026-10-22)

- User: "này cần bảng hợp lí" (Consistency). tab:main now ends with the three runs of the UniQuery variant (DTB s1 82.53/86.44/84.49, DTB s2 82.51/86.32/84.42, DTJ two joint query heads 82.51/86.43/84.47), so the 8/8 and 7/8 comparisons can be checked in one table without repeating rows. The caption defines the variant in plain words: richer sentence set plus region text scores for the 24 clothing colors in the CLIP query head (dual_tr, clip_l14_text_new.pt), and branch dropout 0.25 (each encoder hidden from the joint query head with probability 0.25 per batch). The Experiments opening no longer claims "higher than every other configuration" (the variant rows are higher on the mean).
- User: "đồ thị này khó đọc theo dõi và khúc này viết khó hình dung"; "hình vẽ cx dễ theo dõi, dễ hiểu". Fig. pick redrawn by analysis/make_pick_fig.py: one row per attribute (two-seed mean of DTB, both folds, gap >= 2 points, >= 50 positives, 17 rows), raw balanced accuracy, DINOv3 blue dot / CLIP orange square / joint purple open diamond, blocks "other attributes" and "clothing colors". Statistics over all 148 (attribute, run) pairs: joint > mean of the two 82.4%; >= better 52.7%; non-colors >= better 68.8%, colors 40.5%. The paragraph now says this pattern first, then the attention evidence (92% of reads put > 99% on one encoder; backpack 6.4 of 8 heads on DINOv3, upper red / pink 2.6; route_DU32_PETA_s1). Contribution 3 in the intro scoped the same way. Per-attribute gap vs number of DINOv3 heads correlates only weakly (Pearson 0.16), so the text does not claim that attention follows the better head.
- User: "2 này gom lại và bỏ bảng 3 đi". The choice and hypothesis paragraphs are one paragraph; tab:partner removed (the 12 encoders are listed with citations in the text; numbers kept: CLIP five runs 81.04-81.23, only AIMv2-L inside, overlap 66-72% with DINOv3, CLIP alone >= 0.5 points above the others, recipe caveat).
- User: "viết dễ hiểu hơn ... từ khúc này đến hết bài". Fusion heads, fine-tuning, text score, hidden test, Limitations and Conclusion rewritten as question -> what was done -> main numbers -> meaning; implementation details moved to captions (zero-start correction and loss in tab:frozen, 32.6M parameters in tab:swap); the text-score paragraph now gives the zero-shot evidence first and the weak post-training ablation second; the hidden-test paragraph refers to the post-hoc rule in words (formula only in the Method).

## Table swap completed (2026-10-22)

- Market fold of the two swap runs (vast RTX PRO 6000, remote_results/duf/{go17,boot17}.sh, logits remote_results/duf/logits/D{HR,US}_Market_s1.npz): DINOv3 + C-RADIOv4-H 86.50 (mean 84.31), DINOv3 + SigLIP2-L 86.07 (mean 83.94); PETA values reproduced from saved logits (82.11, 81.81, DRC 81.87 / 86.43). Text now says the swaps lower the two-fold mean, the drop is on PETA, and the two C-RADIOv4-H pairs are slightly above UniQuery on Market.

## Review fixes (2026-10-22, conference-reviewer subagent, user: "sửa đi")

- tab:main UniQuery rows now all come from one saved logit set (remote_results/duf/logits/DU32_{PETA,Market}_s1.npz, keys h_/c_/j_/logits_w1.00): DINOv3 head 81.26/86.07/83.67, CLIP head 81.45/85.56/83.51, joint 82.26/86.27/84.26, NEW DINOv3 + CLIP heads without the joint head 82.24/86.32/84.28, three heads 82.54/86.39/84.46 (old rows 82.55/86.37 etc. came from an evaluation whose logits are not stored). tab:swap submitted row updated to match. Text: the gain needs the joint query head (DINOv3 + CLIP heads of the jointly trained model and the joint head alone are both below the ensemble). The old "0.52 over the DINOv3 head" sentence is gone.
- Protocol: the 0.42 factor behind tau and the encoder choice were selected on the same folds; the hidden test is the only evaluation not affected. mA / Challenge_Avg now defined at first use (related work); text score defined at first use; DeepMAR weights e^{1-p} / e^{p} given in the Method; latency at batch 128; tab:two caption explains tau 0.42 vs 0.32 and that the two DINOv3 copies are two runs of one recipe; Market caveat for tab:two; consistency paragraph says the extra runs are a variant, not reseeds.
- tab:frozen gets a Source column (two-fold mean of the 15% held-out source images, remote_results/fusion/{PETA,Market}_bb1.json "inner"; train sets 22,478 / 14,407 images); text: the five correction heads beat the logit average by 1.1-1.7 points on source images; claim scoped to "the fusion heads we tried".
- Contributions 1-2 and Conclusion scoped (small hidden-test margin; joint query head needed; "consistent with" for the pretraining explanation); Limitations adds "most results come from a single seed".

## English copy finalized (2026-10-22, user: "làm bản tiếng anh đi")

- A read-only comparison agent checked main.tex against main_vi.tex: all numbers, table cells and bold marks match. 40 wording fixes applied to main.tex (banned words encoder-specific / joint-model / hatched / grey / bare "heads"; restored hedges "rigorously", "self-supervised", "only fits", "about the size of"; "leaderboard" removed for anonymity; long sentences split; "In this paper"). Both copies: Tab. swap sentence now says the drop is mostly on PETA and the SigLIP2-L pair is also lower on Market. Fig. 5c title now "Number of the 8 attention heads ...". Only the OpenReview paper ID is missing.
