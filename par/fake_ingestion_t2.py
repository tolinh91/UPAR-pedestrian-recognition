"""Local stand-in for the Track 2 ingestion program: builds the sample dict from /app/input_data and calls rank_gallery.
docker run ... -v <input>:/app/input_data:ro -v <submission>:/app/program:ro -w /app/program <image> python /ing.py"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/app/program")
t0 = time.time()
import run
root = "/app/input_data"
names = json.load(open(f"{root}/task.json"))["attribute_names"]
queries = [[int(v) for v in l.split(",")] for l in open(f"{root}/queries.csv") if l.strip() and not l.startswith("#")]
paths = [l.strip() for l in open(f"{root}/images.csv") if l.strip()]
gallery = [{"index": i, "image": p, "image_path": f"{root}/images/{p}"} for i, p in enumerate(paths)]
run.load_model()
out = run.rank_gallery({"attribute_names": names, "queries": queries, "gallery": gallery})
assert list(out) == ["distances"], out.keys()
sim = -np.asarray(out["distances"])
assert sim.shape == (len(queries), len(gallery)) and np.isfinite(sim).all(), sim.shape
np.save("/app/output/sim.npy", sim)
print(f"INGESTION OK {sim.shape} in {time.time() - t0:.1f}s")
