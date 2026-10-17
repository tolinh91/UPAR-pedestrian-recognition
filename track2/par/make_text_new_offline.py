"""zs_probe.py (new prompt ensembles -> clip_l14_text_new.pt) without internet: open_clip weights from a local file."""
import sys
import open_clip
import zs_probe as m

W = sys.argv[2]
_orig = open_clip.create_model_and_transforms
m.open_clip.create_model_and_transforms = lambda name, pretrained=None, **kw: _orig(name, pretrained=W, **kw)
m.main(sys.argv[1])
