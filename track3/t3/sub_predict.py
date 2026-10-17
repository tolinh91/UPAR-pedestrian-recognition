"""Run a Track 3 submission's run.py (inside the evaluation docker) on UPAR crops and dump COCO predictions.
docker run ... -v <submission>:/app/program:ro -v <t3 data>:/data:ro -v <ann json>:/ann.json:ro python /sp.py N"""
import json, sys, time
sys.path.insert(0, "/app/program")
import run
from PIL import Image
n = int(sys.argv[1])
d = json.load(open("/ann.json"))
ims = {im["id"]: im for im in d["images"]}
anns = [a for a in d["annotations"] if a.get("num_keypoints", 0) > 0][:n]
out, t = [], time.time()
for a in anns:
    im = ims[a["image_id"]]
    r = run.predict_image({"image_id": im["id"], "file_name": im["file_name"], "image_path": "/data/" + im["file_name"],
                           "width": im["width"], "height": im["height"], "boxes": [a["bbox"]], "box_ids": [a["id"]]})
    out.append({"image_id": im["id"], "category_id": 1, "score": r["scores"][0],
                "keypoints": [v for kp in r["keypoints"][0] for v in kp]})
json.dump(out, open("/out/preds.json", "w"))
print("done", len(out), f"{(time.time() - t) / len(out) * 1000:.0f} ms/img")
