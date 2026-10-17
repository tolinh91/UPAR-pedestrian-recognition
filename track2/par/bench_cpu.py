"""CPU inference benchmark of a built submission for several thread counts.

python par/bench_cpu.py ../submissions/dinov3_b16_v2 16 8 4
"""
import importlib.util
import sys
import time

import torch

sys.path.insert(0, ".")
from upar.data import DATA, attr_names, load_split  # noqa: E402

spec = importlib.util.spec_from_file_location("run", sys.argv[1] + "/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
run.load_model()
val = load_split("val").sample(256, random_state=0)
S = [{"image_path": str(DATA / p), "attribute_names": attr_names()} for p in val.image]
for th in [int(a) for a in sys.argv[2:]]:
    torch.set_num_threads(th)
    t = time.time(); [run.predict_batch(S[i:i + 64]) for i in range(0, 256, 64)]
    t1 = time.time(); [run._load(s["image_path"]) for s in S]; t2 = time.time()
    print(f"threads={th}: {(t1 - t) / 256 * 1000:.0f} ms/img total, decode {(t2 - t1) / 256 * 1000:.1f} ms/img", flush=True)
