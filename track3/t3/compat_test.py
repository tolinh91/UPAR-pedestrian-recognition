"""Can torch 2.4.1 (grader image) load a TorchScript exported by the training torch? Uses the pretrained ViTPose+.

python compat_test.py export MODEL_NAME OUT_DIR    (training env: writes cpu fp32 + cuda fp16 TorchScript and a reference)
python compat_test.py check OUT_DIR                 (torch 2.4.1 env: loads both, runs the fp32 one on CPU, compares)
"""
import sys
from pathlib import Path

import torch

if sys.argv[1] == "export":
    from pose import ViTPose
    from export import Wrapper
    name, out = sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    w = Wrapper(ViTPose(name, (256, 192), attn="eager")).eval()
    x = torch.randn(3, 3, 256, 192)
    torch.jit.freeze(torch.jit.trace(w, x, strict=False)).save(str(out / "cpu32.ts"))
    torch.save({"x": x, "y": w(x)}, out / "ref.pt")
    wh = w.cuda().half()
    torch.jit.freeze(torch.jit.trace(wh, x.cuda().half(), strict=False)).save(str(out / "gpu16.ts"))
    print("exported with torch", torch.__version__)
else:
    out = Path(sys.argv[2])
    ref = torch.load(out / "ref.pt")
    m = torch.jit.load(str(out / "cpu32.ts"), map_location="cpu")
    with torch.no_grad():
        d = (m(ref["x"]) - ref["y"]).abs().max().item()
    print("torch", torch.__version__, "cpu32 loads, max |diff| vs training torch:", d)
    g = torch.jit.load(str(out / "gpu16.ts"), map_location="cpu")
    print("gpu16 loads too:", type(g).__name__)
