"""Export a fine-tuned pose.py checkpoint to TorchScript for the submission (evaluation image: torch 2.4.1).

python export.py ../runs/ft_l_all/last.pt ../submissions/t3_ft/assets/model_gpu.torchscript
"""
import sys
from pathlib import Path

import torch

import pose
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


class TokenWrapper(Wrapper):
    """Heatmaps and the mean of the last backbone tokens from one forward pass (inputs of the deep rescorer)."""

    def forward(self, x):
        kw = {"dataset_index": self.index.expand(x.shape[0])} if self.plus else {}
        f = self.m.backbone.forward_with_filtered_kwargs(x, **kw).feature_maps[-1]        # (B, h*w, C)
        B, N, C = f.shape
        h, w = x.shape[2] // 16, x.shape[3] // 16
        hm = self.m.head(f.permute(0, 2, 1).reshape(B, C, h, w))
        return hm, f.mean(1)


if __name__ == "__main__":
    ck_path, out, half = sys.argv[1], Path(sys.argv[2]), "--half" in sys.argv  # --half: fp16 weights, GPU only
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    H, W = map(int, ck["args"].get("size", "256x192").split("x"))
    pose.set_size(H, W)
    model = ViTPose(ck["args"]["model"], (H, W), attn="eager")  # plain matmul attention: loadable by torch 2.4.1
    model.load_state_dict(ck["model"])
    tokens = "--tokens" in sys.argv  # also return the mean last-layer token (deep rescorer)
    w = (TokenWrapper(model) if tokens else Wrapper(model)).eval()
    torch.set_grad_enabled(False)
    dev = "cuda" if half else "cpu"
    w = w.to(dev).half() if half else w
    dt = torch.float16 if half else torch.float32
    ex = torch.randn(2, 3, H, W, device=dev, dtype=dt)
    traced = torch.jit.freeze(torch.jit.trace(w, ex, strict=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    re = torch.jit.load(str(out))
    for b in (1, 5):
        x = torch.randn(b, 3, H, W, device=dev, dtype=dt)
        if tokens:
            ref = Wrapper(model).eval().to(dev)(x).float()
            d = max((re(x)[0].float() - ref).abs().max().item(), (re(x)[1].float() - w(x)[1].float()).abs().max().item())
        else:
            d = (re(x).float() - w(x).float()).abs().max().item()
        print(f"batch {b}: max |traced - eager| = {d:.2e}")
    print("saved", out, f"{out.stat().st_size / 1e6:.0f} MB", "from", ck["args"]["model"], "epoch", ck.get("epoch"))
