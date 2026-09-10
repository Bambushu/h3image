"""Revert the failed pikes2 box to the pre-pass pixels (shield2.png) and log it as a pass."""
import json, os, time
from PIL import Image
import build
st = build.load_state()
box = [3040, 0, 5216, 800]
out = os.path.join(build.OUT, "pikes2_revert.png")
build.paste_back(st["canvas"], os.path.join(build.OUT, "shield2.png"), box, 32, 64, out)
W, H = Image.open(out).size
st["done"].append("pikes2_revert"); st["canvas"] = out
st["log"].append({"name": "pikes2_revert", "kind": "revert", "box": box, "refs": [], "denoise": None, "wall_s": 0, "size": [W, H], "backend": "pixel"})
build.save_state(st); print("reverted ->", out)
