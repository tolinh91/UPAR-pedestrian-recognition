"""Uniform model soup: average the weights of runs fine-tuned from the same pretrained model with the same recipe
(Wortsman et al. 2022). Writes <out>/last.pt (+ args.json) so score_run.py / make_submission.py can use it.

python soup.py ../runs/soup_kdL ../runs/kdL_s2 ../runs/kdL_s3 ../runs/kdL_s4
"""
import json
import shutil
import sys
from pathlib import Path

import torch

if __name__ == "__main__":
    out, runs = Path(sys.argv[1]), [Path(r) for r in sys.argv[2:]]
    cks = [torch.load(r / "last.pt", map_location="cpu", weights_only=False) for r in runs]
    soup = dict(cks[0])
    soup["model"] = {k: (sum(c["model"][k].float() for c in cks) / len(cks)).to(v.dtype) if v.is_floating_point() else v
                     for k, v in cks[0]["model"].items()}
    soup["soup_of"] = [str(r) for r in runs]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(soup, out / "last.pt")
    if (runs[0] / "args.json").exists():
        shutil.copy(runs[0] / "args.json", out / "args.json")
    json.dump({"soup_of": soup["soup_of"]}, open(out / "soup.json", "w"))
    print("soup of", len(runs), "->", out / "last.pt")
