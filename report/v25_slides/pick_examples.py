"""Contact sheet of candidate example pedestrians (val split, PA100k and PETA) for the v25 slides."""
import io
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, image_cache, load_split

A = attr_names(); va = load_split("val"); buf, off = image_cache("val")
y = va[A].values
want = [A.index(a) for a in ("Accessory-Backpack", "Accessory-Hat", "Accessory-Bag", "Gender-Female", "Hair-Length-Long")]
colour = [i for i, a in enumerate(A) if "Color" in a and "Black" not in a and "Grey" not in a and "Other" not in a]
score = y[:, want].sum(1) + 0.5 * y[:, colour].sum(1)
rng = np.random.default_rng(3)
cands = []
for ds in ("PA100k", "PETA"):
    rows = np.where((va.ds.values == ds) & (score >= 2.5))[0]
    rows = rows[[Image.open(io.BytesIO(buf[off[j]:off[j + 1]].tobytes())).size[1] >= 160 for j in rows]]
    cands += list(rng.choice(rows, 12, replace=False))
W, H = 112, 224
sheet = Image.new("RGB", (12 * W, 2 * (H + 20)), (250, 250, 250))
for n, j in enumerate(cands):
    im = Image.open(io.BytesIO(buf[off[j]:off[j + 1]].tobytes())).convert("RGB").resize((W, H))
    x, yy = (n % 12) * W, (n // 12) * (H + 20)
    sheet.paste(im, (x, yy + 20)); ImageDraw.Draw(sheet).text((x + 4, yy + 4), str(j), fill=(0, 0, 0))
sheet.save("contact.png")
for j in cands:
    print(j, va.ds.values[j], [A[k] for k in np.where(y[j] > 0)[0]])
