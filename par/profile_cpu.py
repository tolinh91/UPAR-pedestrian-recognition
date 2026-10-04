"""Time one submission's image loading and model forward separately (run inside the evaluation docker).
docker run ... -v <submission>:/app/program:ro -v <data>:/data:ro -v par/profile_cpu.py:/p.py:ro python /p.py"""
import csv, importlib.util, time, torch
spec = importlib.util.spec_from_file_location("run", "/app/program/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
rows = list(csv.reader(open("/data/annotations/task1/val/gt.csv")))[1::50][:256]
paths = ["/data/" + r[0] for r in rows]
run.load_model()
t = time.time(); xs = [torch.stack(list(run._POOL.map(run._load, paths[i:i + 64]))) for i in range(0, 256, 64)]; tl = time.time() - t
with torch.no_grad():
    run._forward(((xs[0][:8].float().div_(255) - run.MEAN) / run.STD))  # warm-up
    t = time.time()
    for x in xs:
        run._forward(((x.float().div_(255) - run.MEAN) / run.STD))
    tf = time.time() - t
print(f"PROFILE load+decode {tl / 256 * 1000:.1f} ms/img | model {tf / 256 * 1000:.1f} ms/img | threads {torch.get_num_threads()}")
