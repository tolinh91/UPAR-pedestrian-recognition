"""UPAR 2027 Track 2 submission (attribute-based person retrieval): DINOv3 ViT + attribute head.

The ingestion program calls load_model() once, then rank_gallery(sample) with all queries and the whole gallery.
Attribute probabilities are predicted once per gallery image (same inference code as Track 1, offsets off), then
every query-image pair is scored by the log-probability that the image has exactly the query's 40 attributes.
timm and upar.models (the training model code) are vendored in ./vendor: the evaluation image ships neither.

The platform hides logs, so this file is defensive:
- GPU is smoke-tested at load time; any CUDA problem falls back to CPU.
- CPU threads follow the container's cgroup quota, not the host core count.
- A time budget: TTA is dropped when the projected run exceeds it, and as a last resort the remaining
  images get the running mean prediction, so the run never hits the wall-clock limit.
- Unreadable images, NaNs and attribute-name mismatches never fail the run.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T_IMPORT = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from upar.models import build_model  # noqa: E402  (vendored, same code as training)
from PIL import Image, ImageFile  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

BATCH_SIZE = 64
# platform limit is 3600 s; the ingestion clock and ours agree within a few seconds (server log, 29,248 images)
BUDGET_S = float(os.environ.get("UPAR_BUDGET_S", 3200))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_MODEL = None
_CFG = None
_OFFSET = None
_DEVICE = "cpu"
_AMP_DTYPE = None
_TTA = True
_POOL = None
_S = {"n": 0, "t0": None, "bad": 0, "total": None, "fallback": 0, "sum": None, "int8": False}


def log(msg):
    print(f"[run.py] {msg}", flush=True)


def _cpu_limit():
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period))))
    except Exception:
        try:  # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except Exception:
            pass
    return max(1, n)


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _forward(x):
    with torch.autocast("cuda", dtype=_AMP_DTYPE or torch.float32, enabled=_AMP_DTYPE is not None):
        logit = _MODEL(x).float()
        if _TTA:
            logit = (logit + _MODEL(torch.flip(x, dims=[3])).float()) / 2
    return logit


def _gpu_diagnostics():
    """Why CUDA is (not) usable here: logged once, since the platform shows no other hardware information."""
    import glob
    import subprocess
    import warnings
    info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "dev_nvidia": sorted(glob.glob("/dev/nvidia*"))[:6]}
    try:
        info["driver"] = open("/proc/driver/nvidia/version").readline().strip()[:120]
    except Exception:
        info["driver"] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = (out.stdout or out.stderr).strip()[:200]
    except Exception as e:
        info["nvidia_smi"] = repr(e)[:120]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info["is_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count() if info["is_available"] else 0
    info["cuda_warnings"] = [str(w.message)[:200] for w in caught]
    log(f"gpu diagnostics: {json.dumps(info)}")


def load_model() -> None:
    global _MODEL, _CFG, _OFFSET, _AMP_DTYPE, _TTA, _POOL, _DEVICE
    t0 = time.time()
    try:
        _gpu_diagnostics()
    except Exception as e:  # diagnostics must never break the run
        log(f"gpu diagnostics failed: {e!r}")
    with open(os.path.join(HERE, "assets", "config.json"), encoding="utf-8") as fh:
        _CFG = json.load(fh)
    # "model" holds backbone / size / head / head_kw; older configs only have backbone + size (linear head)
    mcfg = dict(_CFG.get("model") or {"backbone": _CFG["backbone"], "size": _CFG["size"]})
    mcfg["num_attrs"] = len(_CFG["attribute_names"])
    global MEAN, STD  # normalisation of the backbone's pretraining; ImageNet when not stored (DINOv3 checkpoints)
    if mcfg.get("mean") and mcfg.get("std"):
        MEAN = torch.tensor(mcfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        STD = torch.tensor(mcfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
    model = build_model(mcfg, pretrained=False)
    state = torch.load(os.path.join(HERE, "assets", "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()
    h, w = _CFG["size"]

    gpu = "-"
    if torch.cuda.is_available():
        try:
            _DEVICE = "cuda"
            _MODEL = model.to("cuda")
            gpu = torch.cuda.get_device_name(0)
            # the model was trained in bf16; fp16 can overflow in ViT activations -> NaN
            _AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
            _TTA = _CFG.get("tta", True)
            with torch.no_grad():
                out = _forward(torch.zeros(2, 3, h, w, device="cuda"))
            if not torch.isfinite(out).all():
                raise RuntimeError("non-finite GPU smoke test")
            torch.cuda.synchronize()
        except Exception as e:
            log(f"GPU unusable ({e!r}), falling back to CPU")
            _DEVICE, _AMP_DTYPE, gpu = "cpu", None, "-"
            model = model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if _DEVICE == "cpu":
        _MODEL = model
        torch.set_num_threads(_cpu_limit())
        _TTA = _CFG.get("tta_cpu", False)
        mode = os.environ.get("UPAR_CPU_INT8", str(_CFG.get("cpu_int8", "0")))
        if mode in ("1", "all"):
            # every Linear: fastest, but destroys DINOv3 accuracy (val Avg 0.885 -> 0.703), kept for reference
            _MODEL = torch.ao.quantization.quantize_dynamic(_MODEL, {torch.nn.Linear}, dtype=torch.qint8)
            _S["int8"] = "all"
        elif mode == "partial":
            # only layers fed by LayerNorm (qkv, fc1): still loses too much (val Avg 0.885 -> 0.809). Default: off
            names = {n: torch.ao.quantization.default_dynamic_qconfig for n, m in _MODEL.named_modules()
                     if isinstance(m, torch.nn.Linear) and n.endswith(("qkv", "fc1"))}
            _MODEL = torch.ao.quantization.quantize_dynamic(_MODEL, names, dtype=torch.qint8)
            _S["int8"] = "partial"
    _POOL = ThreadPoolExecutor(max_workers=min(4, _cpu_limit()))
    global BUDGET_S  # a config may override the time budget (null = no budget: run every image)
    if "budget_s" in _CFG and "UPAR_BUDGET_S" not in os.environ:
        BUDGET_S = float("inf") if _CFG["budget_s"] is None else float(_CFG["budget_s"])
    _OFFSET = torch.tensor(_CFG["offset"], dtype=torch.float32, device=_DEVICE) * _CFG.get("offset_scale", 1.0)
    log(f"device={_DEVICE} gpu={gpu} amp={_AMP_DTYPE} tta={_TTA} threads={torch.get_num_threads()} "
        f"cpu_limit={_cpu_limit()} host_cpus={os.cpu_count()} head={mcfg.get('head', 'linear')} backbone={mcfg.get('backbone')} mean={MEAN.flatten().tolist()} "
        f"int8={_S['int8']} "
        f"load={time.time() - t0:.1f}s")


def _count_images(sample):
    """Number of test images, from images.csv / task.json next to the images folder."""
    try:
        rel, full = sample["image"], sample["image_path"]
        img_dir = full[: len(full) - len(rel)].rstrip("/")
        root = os.path.dirname(img_dir)
        n = 0
        csv = os.path.join(root, "images.csv")
        if os.path.exists(csv):
            with open(csv, encoding="utf-8") as fh:
                n = sum(1 for line in fh if line.strip())
        task = os.path.join(root, "task.json")
        if os.path.exists(task):
            with open(task, encoding="utf-8") as fh:
                for k, v in json.load(fh).items():
                    if isinstance(v, int) and ("image" in k.lower() or k.lower().startswith("n")):
                        n = max(n, v)
        return n or None
    except Exception:
        return None


def _load(path):
    h, w = _CFG["size"]
    try:
        with Image.open(path) as im:
            im.draft("RGB", (w * 2, h * 2))  # fast JPEG decode at reduced scale for large crops
            img = im.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8).copy()
    except Exception as e:  # never fail the whole run on one bad file
        _S["bad"] += 1
        if _S["bad"] <= 5:
            log(f"WARNING cannot read {path}: {e!r}")
        arr = np.full((h, w, 3), 128, np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _order(names):
    ours = [_norm(n) for n in _CFG["attribute_names"]]
    try:
        return [ours.index(_norm(n)) for n in names]
    except ValueError:
        log(f"WARNING attribute names differ, using training order: {names}")
        return list(range(len(ours)))


@torch.no_grad()
def predict_batch(samples: list[dict]) -> list[list[float]]:
    if _MODEL is None:
        load_model()
    if _S["t0"] is None:
        _S["t0"] = time.time()
        _S["total"] = _count_images(samples[0])
        log(f"test images: {_S['total']}")
    names = samples[0].get("attribute_names") or _CFG["attribute_names"]

    if _S["fallback"] and _S["sum"] is not None:
        mean = _S["sum"] / max(1, _S["n"])
        out = np.repeat(mean[None], len(samples), 0)
    else:
        x = torch.stack(list(_POOL.map(_load, [s["image_path"] for s in samples]))).float().div_(255)
        x = ((x - MEAN) / STD).to(_DEVICE)
        logit = torch.nan_to_num(_forward(x), nan=0.0, posinf=20.0, neginf=-20.0)
        out = torch.sigmoid(logit + _OFFSET).clamp_(0.0, 1.0).cpu().numpy().astype(np.float64)
        _S["sum"] = out.sum(0) if _S["sum"] is None else _S["sum"] + out.sum(0)

    _S["n"] += len(samples)
    n_timed = _S["n"] - _S.get("n0", 0)
    if _S["n"] % (BATCH_SIZE * 20) < len(samples):
        dt = time.time() - _S["t0"]
        log(f"{_S['n']}/{_S['total']} images, {time.time() - T_IMPORT:.0f}s since import, "
            f"{dt / max(1, n_timed) * 1000:.1f} ms/img, tta={_TTA}, bad={_S['bad']}")
    if not _S["fallback"]:
        _check_budget(n_timed)
    return out[:, _order(names)].tolist()


def _check_budget(n_timed):
    """Drop TTA, then switch to the mean-prediction fallback, when the projected finish exceeds the budget."""
    global _TTA
    if not _S["total"] or n_timed < 2 * BATCH_SIZE:
        return
    spent = time.time() - T_IMPORT
    rate = (time.time() - _S["t0"]) / n_timed
    projected = spent + rate * (_S["total"] - _S["n"])
    if _TTA and projected > BUDGET_S:
        _TTA = False
        _S["t0"], _S["n0"] = time.time(), _S["n"]
        log(f"projected {projected:.0f}s > budget {BUDGET_S:.0f}s, TTA disabled")
    elif not _TTA and spent > BUDGET_S:
        _S["fallback"] = 1
        log(f"budget spent at {_S['n']}/{_S['total']} images, remaining images get the mean prediction")


def predict_image(sample: dict) -> list[float]:
    return predict_batch([sample])[0]


@torch.no_grad()
def rank_gallery(sample: dict) -> dict:
    """{"distances": (Q, G)} = -(sum_k q_k log p_k + (1 - q_k) log(1 - p_k)), smaller = better match (the key the
    official sample submission returns)."""
    if _MODEL is None:
        load_model()
    gallery, names = sample["gallery"], sample["attribute_names"]
    queries = np.asarray(sample["queries"], dtype=np.float64)
    _S["t0"], _S["total"] = time.time(), len(gallery)
    log(f"queries: {len(queries)}, gallery images: {len(gallery)}")
    probs = []
    for i in range(0, len(gallery), BATCH_SIZE):
        probs += predict_batch(gallery[i:i + BATCH_SIZE])          # rows already in the order of `names`
    p = np.clip(np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=0.5), 1e-4, 1 - 1e-4)
    lp, lq = np.log(p), np.log1p(-p)
    sim = queries @ (lp - lq).T + lq.sum(1)[None, :]
    sim = np.nan_to_num(sim, nan=-1e6, posinf=1e6, neginf=-1e6).astype(np.float32)
    log(f"distances {sim.shape}, {time.time() - T_IMPORT:.0f}s since import, fallback={_S['fallback']}")
    return {"distances": -sim}
