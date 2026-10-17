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
| DINOv3 | DINOv3 ViT-H+/16; full name only where the encoders are introduced | H+, DINO, DINOv3-H |
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
| joint model | DINOv3 and CLIP fine-tuned in one network with three query heads, logits of the three averaged | dual model, v31 |
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
