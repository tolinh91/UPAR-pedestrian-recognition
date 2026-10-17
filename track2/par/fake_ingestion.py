"""Mimic the Codabench ingestion program on fake_input_data (run inside the evaluation image)."""
import importlib.util, json, math, sys, time
t = time.time()
sys.path.insert(0, "/app/program")
spec = importlib.util.spec_from_file_location("run", "/app/program/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
names = json.load(open("/app/input_data/task.json"))["attribute_names"]
rows = [l.strip() for l in open("/app/input_data/images.csv") if l.strip()]
samples = [{"index": i, "image": r, "image_path": "/app/input_data/images/" + r, "attribute_names": names}
           for i, r in enumerate(rows)]
if hasattr(run, "load_model"): run.load_model()
bs = getattr(run, "BATCH_SIZE", 32); out = []
for i in range(0, len(samples), bs):
    res = run.predict_batch(samples[i:i + bs])
    assert len(res) == len(samples[i:i + bs])
    for r in res:
        assert len(r) == 40 and all(0 <= float(v) <= 1 and not math.isnan(float(v)) for v in r), r
    out += res
print(f"INGESTION OK {len(out)} predictions in {time.time() - t:.1f}s")
