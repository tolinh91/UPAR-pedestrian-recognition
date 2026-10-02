# Experiment plan, 2026-10-02 (UPAR 2027 Track 1 and Track 2)

## Known so far

Track 1 (attribute recognition, metric Challenge_Avg = harmonic mean of mA and instance F1, fixed threshold 0.5,
hidden test from an unseen domain, CPU-only server: 6 threads, 3600 s for ~29k images):
- Leaderboard: ViT-L/16 176x96 + query head = 0.7857 (best), ViT-B linear head = 0.7775, ViT-B query head = 0.7750.
- Leave-PETA-out proxy (train Market1501 + PA100k, score PETA val after offsets fit on seen domains, x0.6):
  ViT-B linear 0.7904 (AUC 0.908), ViT-B query 0.7958 (AUC 0.901-0.904), ViT-L 176 query 0.8005 (AUC 0.910),
  ViT-L 224 query 0.8071 (AUC 0.910). ~30 head / loss / augmentation variants on ViT-B: none beat the query head by
  0.003; AUC of all ViT-B variants 0.898-0.908.
- The proxy mispredicted two threshold-type changes (query head vs linear on ViT-B, offset scale 1.0 vs 0.6) and
  predicted the backbone change correctly. Noise estimate from only 2 runs (0.7961 vs 0.7956), no seeds.
- Threshold error (source offsets vs oracle offsets fit on PETA) ~0.02, mostly colour attributes; hypothesis:
  labelling conventions (PETA multi-colour labels). Not verified.
- CPU budget: ViT-L 176x96 ~98 ms/img on the server (48 min); ViT-L 224x112 ~149 ms/img (too slow);
  dynamic int8 destroys accuracy.

Track 2 (retrieval: query = 40-bit attribute vector, rank the gallery, match = all 40 equal, metric mADM = AP-like
with credit for near misses; same runtime limits):
- Track 1 models reused: predict 40 probabilities per image, score each (query, image) pair.
- Val (in-domain): log-likelihood scoring beats the official baseline's L1 by +0.04..0.08 mADM; Track 1 offsets hurt.
- PETA held-out: mADM 0.631 (ViT-B linear) vs 0.626 (ViT-B query), loglik; L1 0.57-0.58.
- First submission built (ViT-L 176 query + loglik), not yet submitted.

## Running now (2x RTX 5090, 5-epoch proxies, every run scored for both tracks by score_run.py)

1. ViT-B query and ViT-B linear, holdout PETA, seed 1 and seed 2 (noise with fixed seeds)
2. ViT-L 176 query and ViT-L 176 linear, holdout PETA
3. ViT-B query and linear, holdout Market1501 and holdout PA100k (multi-domain proxy)
4. Query head with linear attention (elu+1 kernel) and with sigmoid attention, holdout PETA

## Candidates for the next batch

- Truncated ViT-L at 224x112: keep the first 16 / 18 of 24 blocks (CPU ~99 / ~112 ms/img).
- Dense feature distillation ViT-L 224 -> ViT-B 256 (cosine on patch tokens + patch-affinity matrices).
- Track 2: loss without DeepMAR weighting (pure BCE, better calibrated probabilities for log-likelihood ranking);
  group-structured probabilities (softmax within one-of-K groups) at scoring time; temperature scaling;
  mixing log-likelihood with expected degree of match for mADM.
- ViT-L 176 on the Market1501 / PA100k holdouts; more seeds for the final candidates.
