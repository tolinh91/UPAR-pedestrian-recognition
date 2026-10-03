"""Export a fine-tuned pose.py checkpoint to TorchScript for the submission (evaluation image: torch 2.4.1).

python export.py ../runs/ft_l_all/last.pt ../submissions/t3_ft/assets/model_gpu.torchscript
"""
import sys
from pathlib import Path

import torch

from pose import ViTPose


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m, self.plus = model.m, model.plus
        self.register_buffer("index", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        if self.plus:
            return self.m(pixel_values=x, dataset_index=self.index.expand(x.shape[0])).heatmaps
        return self.m(pixel_values=x).heatmaps


if __name__ == "__main__":
    ck_path, out, half = sys.argv[1], Path(sys.argv[2]), "--half" in sys.argv  # --half: fp16 weights, GPU only
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = ViTPose(ck["args"]["model"])
    model.load_state_dict(ck["model"])
    w = Wrapper(model).eval()
    torch.set_grad_enabled(False)
    dev = "cuda" if half else "cpu"
    w = w.to(dev).half() if half else w
    dt = torch.float16 if half else torch.float32
    ex = torch.randn(2, 3, 256, 192, device=dev, dtype=dt)
    traced = torch.jit.freeze(torch.jit.trace(w, ex, strict=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    re = torch.jit.load(str(out))
    for b in (1, 5):
        x = torch.randn(b, 3, 256, 192, device=dev, dtype=dt)
        d = (re(x).float() - w(x).float()).abs().max().item()
        print(f"batch {b}: max |traced - eager| = {d:.2e}")
    print("saved", out, f"{out.stat().st_size / 1e6:.0f} MB", "from", ck["args"]["model"], "epoch", ck.get("epoch"))
