"""make_clip_text.py without internet: open_clip ViT-L-14 'openai' weights read from the local HF cache file."""
import sys
import open_clip
import make_clip_text as m

W = sys.argv[2]
_orig = open_clip.create_model_and_transforms
m.open_clip.create_model_and_transforms = lambda name, pretrained=None, **kw: _orig(name, pretrained=W, **kw)
m.main(sys.argv[1])
