#!/usr/bin/env python3
"""Score the build: per pass, SSIM against the previous canvas outside the pass's box (must stay
~1: nothing outside a box may move), seam SSIM in a ring around the box, wall time; OCR on the
shield; a making-of sheet (every pass downscaled) and a faces sheet (each identity pass's box
next to its reference). Deps: Pillow, numpy, scikit-image, tesseract.
"""
import difflib, json, os, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity as ssim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, REFS = os.path.join(HERE, "out"), os.path.join(HERE, "refs")
SHIELD_NAMES = ["RAY WALSH", "LAKEM", "ROSALIE", "H3EDIT 2026"]


def gray(p, size=None):
    im = Image.open(p).convert("L")
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255


def outside_and_seam(prev, cur, box, grow=32, ring=96):
    small = (cur.shape[1] // 4, cur.shape[0] // 4)   # SSIM at quarter res: 16 MP full-res SSIM is slow
    a = np.asarray(Image.fromarray((prev * 255).astype(np.uint8)).resize(small, Image.LANCZOS), dtype=np.float32) / 255
    b = np.asarray(Image.fromarray((cur * 255).astype(np.uint8)).resize(small, Image.LANCZOS), dtype=np.float32) / 255
    _, smap = ssim(a, b, data_range=1.0, full=True)
    if box is None:
        return round(float(smap.mean()), 3), None
    x0, y0, x1, y1 = [v // 4 for v in box]
    g, r = grow // 4, ring // 4
    outside = np.ones_like(smap, bool)
    outside[max(0, y0 - g - r):y1 + g + r, max(0, x0 - g - r):x1 + g + r] = False
    seam = np.zeros_like(smap, bool)
    seam[max(0, y0 - g - r):y1 + g + r, max(0, x0 - g - r):x1 + g + r] = True
    seam[max(0, y0 - g):y1 + g, max(0, x0 - g):x1 + g] = False
    return round(float(smap[outside].mean()), 3), round(float(smap[seam].mean()), 3)


def ocr_shield(cur, box):
    x0, y0, x1, y1 = box
    crop = Image.open(cur).convert("L").crop((x0, y0, x1, y1))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    p = os.path.join(OUT, "_ocr.png"); crop.save(p)
    lines = []
    for im in (crop, ImageOps.invert(crop)):
        im.save(p)
        r = subprocess.run(["tesseract", p, "stdout", "--psm", "6"], capture_output=True, text=True)
        lines += [re.sub(r"[^A-Z0-9]", "", l.upper()) for l in r.stdout.splitlines() if l.strip()]
    os.remove(p)
    res = {}
    for n in SHIELD_NAMES:
        gt = re.sub(r"[^A-Z0-9]", "", n)
        res[n] = round(max((difflib.SequenceMatcher(None, gt, l).ratio() for l in lines if l), default=0.0), 2)
    return res


def main():
    st = json.load(open(os.path.join(OUT, "state.json")))
    log = st["log"]
    rows, prev = [], None
    for e in log:
        cur_p = os.path.join(OUT, e["name"] + ".png")
        cur = gray(cur_p)
        row = dict(e)
        if prev is not None:
            row["outside_ssim"], row["seam_ssim"] = outside_and_seam(prev, cur, e["box"])
        if e["name"] == "shield":
            row["ocr"] = ocr_shield(cur_p, e["box"])
        rows.append(row); prev = cur
        print(row["name"], row.get("outside_ssim"), row.get("seam_ssim"), row.get("ocr", ""), flush=True)
    json.dump(rows, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    # making-of sheet
    w = 640; tiles = []
    for e in log:
        im = Image.open(os.path.join(OUT, e["name"] + ".png")).convert("RGB")
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        if e["box"]:
            s = w / e["size"][0]
            d.rectangle([v * s for v in e["box"]], outline=(255, 220, 0), width=2)
        d.rectangle((0, 0, w, 20), fill=(0, 0, 0)); d.text((6, 4), f"{len(tiles)+1:02d} {e['name']}  {e['wall_s']}s", fill=(255, 255, 255))
        tiles.append(im)
    cols = 4; h = tiles[0].height; rows_n = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * w, rows_n * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % cols) * w, (i // cols) * h))
    S.save(os.path.join(OUT, "sheet_makingof.png"))
    # faces sheet: identity passes, box crop from the FINAL canvas next to the reference
    final = Image.open(os.path.join(OUT, log[-1]["name"] + ".png")).convert("RGB")
    ids = [e for e in log if e["refs"] and not e["refs"][0].startswith(("palette", "shield"))]
    tiles = []
    for e in ids:
        x0, y0, x1, y1 = e["box"]
        crop = final.crop((x0, y0, x1, y1)); crop = crop.resize((int(crop.width * 700 / crop.height), 700), Image.LANCZOS)
        ref = Image.open(os.path.join(REFS, e["refs"][0])).convert("RGB"); ref = ref.resize((int(ref.width * 300 / ref.height), 300), Image.LANCZOS)
        t = Image.new("RGB", (crop.width + ref.width + 10, 700), (20, 20, 20)); t.paste(crop, (0, 0)); t.paste(ref, (crop.width + 10, 0))
        ImageDraw.Draw(t).text((6, 6), e["name"] + " <- " + e["refs"][0], fill=(255, 255, 0))
        tiles.append(t)
    if tiles:
        S = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), 700), (20, 20, 20)); x = 0
        for t in tiles:
            S.paste(t, (x, 0)); x += t.width + 10
        S.save(os.path.join(OUT, "sheet_faces.png"))
    # final at 4K for the post
    final.resize((3840, int(final.height * 3840 / final.width)), Image.LANCZOS).save(os.path.join(OUT, "final_4k.jpg"), quality=92)


if __name__ == "__main__":
    main()
