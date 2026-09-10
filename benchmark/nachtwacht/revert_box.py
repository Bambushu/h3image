"""Revert one pass's box to an earlier canvas in pixel space and log it: python3 revert_box.py <pass> <source_pass>"""
import json, os, sys
from PIL import Image
import build
name, src = sys.argv[1], sys.argv[2]
st = build.load_state()
box = next(e["box"] for e in st["log"] if e["name"] == name)
out = os.path.join(build.OUT, f"{name}_revert.png")
build.paste_back(st["canvas"], os.path.join(build.OUT, f"{src}.png"), box, 32, 64, out)
W, H = Image.open(out).size
st["done"].append(f"{name}_revert"); st["canvas"] = out
st["log"].append({"name": f"{name}_revert", "kind": "revert", "box": box, "refs": [], "denoise": None, "wall_s": 0, "size": [W, H], "backend": "pixel"})
build.save_state(st); print("reverted", name, "->", out)
