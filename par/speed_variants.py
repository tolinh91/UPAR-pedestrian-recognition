"""CPU speed of the submitted model in one execution mode and how far it moves the predictions (light: 64 images).
docker run ... -v <submission>:/app/program:ro -v <data>:/data:ro -v par/speed_variants.py:/s.py:ro python /s.py <mode>
modes: eager | jit | bf16"""
import csv, importlib.util, sys, time, torch
mode = sys.argv[1]
spec = importlib.util.spec_from_file_location("run", "/app/program/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
rows = list(csv.reader(open("/data/annotations/task1/val/gt.csv")))[1::100][:64]
run.load_model()
torch.set_num_threads(4)
x = torch.stack(list(run._POOL.map(run._load, ["/data/" + r[0] for r in rows]))).float().div_(255)
x = (x - run.MEAN) / run.STD
m = run._MODEL.eval()
with torch.inference_mode():
    ref = torch.cat([m(x[i:i + 32]).float() for i in range(0, 64, 32)])
    if mode == "jit":
        f = torch.jit.freeze(torch.jit.trace(m, x[:32], check_trace=False).eval())
    elif mode == "bf16":
        def f(z):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return m(z)
    else:
        f = m
    f(x[:32])  # warm-up
    t = time.time(); out = torch.cat([f(x[i:i + 32]).float() for i in range(0, 64, 32)]); dt = (time.time() - t) / 64 * 1000
    t = time.time(); [m(x[i:i + 32]) for i in range(0, 64, 32)]; de = (time.time() - t) / 64 * 1000
flips = ((out > 0) != (ref > 0)).float().mean().item() * 100
print(f"RESULT {mode}: {dt:.1f} ms/img vs eager {de:.1f} ms/img (x{de / dt:.2f}) | max |dlogit| {(out - ref).abs().max().item():.4f} | decision flips {flips:.3f}%")
