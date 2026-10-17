import csv, importlib.util, sys, time, numpy as np, torch
print("torch", torch.__version__, sys.version.split()[0])
spec = importlib.util.spec_from_file_location("run", "/app/program/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
rows = list(csv.reader(open("/data/annotations/task1/val/gt.csv")))
names = [n.strip("# ") for n in rows[0][1:]]
rows = rows[1::160][:200]
samples = [{"index": i, "image": r[0], "image_path": "/data/" + r[0], "attribute_names": names} for i, r in enumerate(rows)]
t = time.time(); run.load_model()
out = []
for i in range(0, len(samples), run.BATCH_SIZE): out += run.predict_batch(samples[i:i + run.BATCH_SIZE])
p = np.array(out); g = np.array([[int(x) for x in r[1:]] for r in rows])
print("shape", p.shape, "range", p.min(), p.max(), "time", round(time.time() - t, 1))
print("label acc", ((p > 0.5) == g).mean())
