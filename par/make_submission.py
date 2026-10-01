"""Build a Codabench zip from a training run.

python par/make_submission.py ../runs/dinov3_b16 --offset-scale 0.6 --name dinov3_b16_v1
"""
import argparse
import json
import shutil
from pathlib import Path

import timm
import torch

from upar.models import cfg_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--offset-scale", type=float, default=0.6, help="shrink val-fitted offsets for the unseen domain")
    ap.add_argument("--no-offset", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-budget", action="store_true", help="process every image even past the time limit")
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    mcfg = cfg_from_checkpoint(ck)
    out = ROOT / "submissions" / args.name
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)

    shutil.copy(ROOT / "submission_template" / "run.py", out / "run.py")
    shutil.copy(ROOT / "UPAR-Challenge-2027/examples/task1/sample_code_submission/metadata.yaml", out / "metadata.yaml")
    shutil.copytree(Path(timm.__file__).parent, out / "vendor" / "timm",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # the model code used in training, without data / losses (they need pandas, torchvision)
    upar = Path(__file__).parent / "upar"
    (out / "vendor" / "upar").mkdir(parents=True)
    for f in ["__init__.py", "registry.py"]:
        shutil.copy(upar / f, out / "vendor" / "upar" / f)
    shutil.copytree(upar / "models", out / "vendor" / "upar" / "models", ignore=shutil.ignore_patterns("__pycache__"))
    if mcfg["backbone"].startswith("nvidia/"):  # C-RADIO patch generator needs einops (pure python)
        import einops
        shutil.copytree(Path(einops.__file__).parent, out / "vendor" / "einops", ignore=shutil.ignore_patterns("__pycache__"))
    torch.save({k: v.half() for k, v in ck["model"].items()}, out / "assets" / "model.pt")

    names = json.load(open(ROOT / "UPAR-Challenge-2027/examples/task1/sample_code_submission/assets/attribute_prior.json"))["attribute_names"]
    offset = [0.0] * len(names)
    if not args.no_offset and (run / "offsets.json").exists():
        off = json.load(open(run / "offsets.json"))
        assert off["attribute_names"] == names
        offset = off["offset"]
    cfg = {"model": mcfg, "backbone": mcfg["backbone"], "size": mcfg["size"], "attribute_names": names,
           "offset": offset, "offset_scale": 0.0 if args.no_offset else args.offset_scale,
           "source": {"run": str(run), "ckpt": args.ckpt, "epoch": ck["epoch"], "val": ck["val"]}}
    if args.no_budget:
        cfg["budget_s"] = None
    json.dump(cfg, open(out / "assets" / "config.json", "w"), indent=1)

    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    print("built", zip_path, f"{Path(zip_path).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
