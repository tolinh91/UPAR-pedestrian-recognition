"""COCO person keypoints -> the 17 PoseTrack18 slots of the UPAR Track 3 annotations, and the images they need.

python coco_to_pt17.py COCO_KEYPOINTS_JSON OUT_JSON IMG_DIR [--min-kp 8] [--min-h 64] [--threads 64]
Slot map: nose <- COCO nose; head_bottom, head_top and both ears are left unlabelled (v = 0: COCO has no head_bottom /
head_top, and the UPAR data never labels ears, so the ear outputs keep their UPAR behaviour); shoulders .. ankles <- the
same COCO joints (identical order, slots 5..16). Kept: iscrowd 0, at least --min-kp labelled COCO joints, box height of
at least --min-h pixels. OUT_JSON has the UPAR layout (file_name "coco/train2017/<file>") and goes next to
train_{market,pa100k,peta}.json; images are downloaded from images.cocodataset.org into IMG_DIR (skipped if present).
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("img_dir")
    ap.add_argument("--min-kp", type=int, default=8)
    ap.add_argument("--min-h", type=float, default=64)
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    d = json.load(open(args.src))
    ims = {im["id"]: im for im in d["images"]}
    anns, keep_ims = [], {}
    for a in d["annotations"]:
        if a.get("iscrowd", 0) or a.get("num_keypoints", 0) < args.min_kp or a["bbox"][3] < args.min_h:
            continue
        k = a["keypoints"]
        kp = [0.0] * 51
        kp[0:3] = k[0:3]                    # nose
        kp[15:51] = k[15:51]                # shoulders .. ankles (COCO joints 5..16 -> slots 5..16)
        n = sum(1 for j in range(17) if kp[3 * j + 2] > 0)
        anns.append({"id": a["id"], "image_id": a["image_id"], "category_id": 1, "iscrowd": 0, "bbox": a["bbox"],
                     "area": a["area"], "keypoints": kp, "num_keypoints": n})
        im = ims[a["image_id"]]
        keep_ims[im["id"]] = {"id": im["id"], "file_name": "coco/train2017/" + im["file_name"],
                              "width": im["width"], "height": im["height"]}
    json.dump({"images": list(keep_ims.values()), "annotations": anns, "categories": [{"id": 1, "name": "person"}]},
              open(args.out, "w"))
    print(f"{len(anns)} persons in {len(keep_ims)} images -> {args.out}", flush=True)

    os.makedirs(args.img_dir, exist_ok=True)
    names = [ims[i]["file_name"] for i in keep_ims]

    def get(n):
        p = os.path.join(args.img_dir, n)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return 0
        for _ in range(3):
            try:
                urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{n}", p + ".part")
                os.replace(p + ".part", p)
                return 1
            except Exception:
                continue
        return -1

    with ThreadPoolExecutor(args.threads) as ex:
        res = list(ex.map(get, names))
    print(f"downloaded {res.count(1)}, present {res.count(0)}, failed {res.count(-1)}", flush=True)


if __name__ == "__main__":
    main()
