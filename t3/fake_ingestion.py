"""Local stand-in for the Track 3 ingestion: a few UPAR crops, one box covering each image, calls run.predict_image."""
import json, sys, time, glob
sys.path.insert(0, "/app/program")
import run
from PIL import Image
paths = sorted(glob.glob("/data/*"))[:20]
t = time.time(); run.load_model(); print("load", round(time.time() - t, 1))
t = time.time()
for i, p in enumerate(paths):
    w, h = Image.open(p).size
    out = run.predict_image({"image_id": i, "file_name": p.split("/")[-1], "image_path": p, "width": w, "height": h,
                             "boxes": [[0, 0, w, h]], "box_ids": [i]})
    assert len(out["keypoints"]) == 1 and len(out["keypoints"][0]) == 17, out
print("ok", len(paths), "images", round((time.time() - t) / len(paths) * 1000), "ms/box", out["scores"], out["keypoints"][0][:3])
