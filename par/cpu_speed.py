"""Inside the evaluation docker: CPU speed of a packaged Track 1 model under faster inference settings."""
import json, os, sys, time
import torch
sys.path.insert(0, "/app/program/vendor")
from upar.models import build_model
torch.set_num_threads(6)
cfg = json.load(open("/app/program/assets/config.json"))
mcfg = dict(cfg["model"], num_attrs=40)
state = {k: v.float() for k, v in torch.load("/app/program/assets/model.pt", map_location="cpu", weights_only=True).items()}

def bench(model, size, label, n=3):
    x = torch.randn(64, 3, *size)
    with torch.inference_mode():
        model(x)
        t = time.time()
        for _ in range(n):
            model(x)
    print(f"{label:38s} {(time.time() - t) / (64 * n) * 1000:6.1f} ms/img", flush=True)

for size in [(176, 96), (160, 80), (144, 80)]:
    m = build_model(dict(mcfg, size=list(size)), pretrained=False)
    m.load_state_dict(state); m.eval()
    bench(m, size, f"eager fp32 {size}")
    if size == (176, 96):
        with torch.inference_mode():
            tr = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(m, torch.randn(64, 3, *size))))
        bench(tr, size, f"jit freeze+optimize {size}")
        bench(m.to(memory_format=torch.channels_last), size, f"channels_last {size}")
