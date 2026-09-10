#!/usr/bin/env python3
"""Score the renders in out/ -> out/scores.json, out/report.md, out/sheet_*.png.
Deps: Pillow, numpy, scikit-image, the `tesseract` binary (brew install tesseract / apt tesseract-ocr).

Metrics
  ocr        per-tier character accuracy of the plate lettering (difflib ratio of the best OCR line
             against the ground truth, upper-cased, spaces collapsed). 1.0 = every character right.
  preserve   outside-plate SSIM vs the source scene (output resized to source), plus 8 fiducial
             patches (window, sill, coping, plinth, pavement, sky) located by normalized
             cross-correlation: NCC peak and how far it moved.
  neon       lit vs switched-off render, same seed: mean Lab shift in the brick beside the board and
             the pavement below it, and in a far patch that should NOT change (global-drift control).
  seam       --detail composite vs its base: SSIM in a band straddling the paste box edge.
"""
import difflib, json, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.feature import match_template
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops

HERE = os.path.dirname(os.path.abspath(__file__))
INP, OUT = os.path.join(HERE, "inputs"), os.path.join(HERE, "out")
DEMOS = os.path.join(HERE, "..", "demos", "refs")
LAYOUT = json.load(open(os.path.join(INP, "sign_layout.json")))
TIERS = LAYOUT["tiers"]
PLATE_BG = np.array([[[28, 36, 58]]], dtype=np.uint8)
SRC_W, SRC_H = 1280, 720
# source-space fiducials of gable_scene-s77 (x0,y0,x1,y1), all outside the plate box
FIDUCIALS = {"window_top": (1060, 130, 1160, 230), "window_sill": (1050, 555, 1185, 600),
             "coping_right": (1150, 85, 1280, 120), "coping_left": (120, 75, 300, 110),
             "plinth_left": (0, 615, 125, 680), "pavement": (400, 688, 700, 720),
             "sky": (0, 0, 120, 60), "wall_right": (1175, 300, 1280, 430)}
# storefront_scene-s42 regions for the neon task (source space)
NEON = {"brick_left": (300, 150, 430, 235), "brick_right": (860, 150, 990, 235),
        "pavement": (430, 650, 880, 715), "far_control": (1170, 330, 1270, 520)}
NEON_HUE_A, NEON_HUE_B = 70.0, -20.0     # the sign's pink in Lab a*/b*, roughly


def load(name):
    return Image.open(os.path.join(OUT, name + ".png")).convert("RGB")


def to_src(im):
    return im.resize((SRC_W, SRC_H), Image.LANCZOS)


def find_plate(im):
    """Bounding box of the largest near-plate-colour blob, else the scaled prompt box."""
    small = im.resize((im.width // 4, im.height // 4))
    d = deltaE_ciede2000(rgb2lab(np.asarray(small)), rgb2lab(PLATE_BG).reshape(1, 1, 3))
    lab = label(d < 14)
    W, H = im.size
    props = [p for p in regionprops(lab) if p.area > 0.05 * small.width * small.height]
    if not props:
        x0, y0, x1, y1 = LAYOUT["sign_box"]
        return (int(x0 * W / SRC_W), int(y0 * H / SRC_H), int(x1 * W / SRC_W), int(y1 * H / SRC_H)), False
    p = max(props, key=lambda p: p.area)
    y0, x0, y1, x1 = p.bbox
    return (x0 * 4, y0 * 4, x1 * 4, y1 * 4), True


def tesseract(im, psm=6):
    p = os.path.join(OUT, "_ocr_tmp.png")
    im.save(p)
    r = subprocess.run(["tesseract", p, "stdout", "--psm", str(psm)], capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def norm(s):
    """Alphanumerics only, upper-case: the question is whether the CHARACTERS came out."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def acc(gt, line):
    """Best difflib ratio of gt against any window of line of similar length (OCR pads lines
    with border junk such as NNNARBOUR for HARBOUR)."""
    n = len(gt)
    if len(line) <= n + 1:
        return difflib.SequenceMatcher(None, gt, line).ratio()
    return max(difflib.SequenceMatcher(None, gt, line[i:i + w]).ratio()
               for w in (n - 1, n, n + 1) for i in range(0, len(line) - w + 1))


def text_rows(g):
    """Row bands of bright text inside a grayscale plate crop, by horizontal projection profile."""
    a = np.asarray(g, dtype=np.float32)
    thr = a.mean() + 0.5 * a.std()
    prof = (a > thr).mean(1)
    on = prof > 0.01
    rows, y = [], 0
    while y < len(on):
        if on[y]:
            y0 = y
            while y < len(on) and on[y]:
                y += 1
            if y - y0 >= 6:
                rows.append((y0, y))
        y += 1
    return rows


def ocr_plate(im, box):
    """OCR every text row of the plate (rows found by projection, so a re-flowed layout still
    scores), each rescaled to ~50 px cap height, plus two whole-plate passes; best match per tier."""
    x0, y0, x1, y1 = box
    inset = int(0.03 * (y1 - y0))           # drop the painted border
    crop = im.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset))
    g = ImageOps.grayscale(crop)
    lines = []
    for sc in (max(1.0, 2400 / g.width), 40 / max(4, TIERS[-1]["cap_ratio"] * (y1 - y0))):
        gs = g.resize((int(g.width * sc), int(g.height * sc)), Image.LANCZOS)
        lines += tesseract(gs) + tesseract(ImageOps.invert(gs))
    rows = []
    for r0, r1 in text_rows(g):
        pad = int(0.35 * (r1 - r0)) + 2
        c = g.crop((0, max(0, r0 - pad), g.width, min(g.height, r1 + pad)))
        sc = 50 / max(4, (r1 - r0))
        c = c.resize((max(8, int(c.width * sc)), max(8, int(c.height * sc))), Image.LANCZOS)
        rows += tesseract(c, 7) + tesseract(ImageOps.invert(c), 7)
    cands = [norm(l) for l in lines + rows]
    res = {}
    for t in TIERS:
        gt = norm(t["text"])
        res[t["tier"]] = round(max((acc(gt, l) for l in cands if l), default=0.0), 3)
    return res, {"plate": [norm(l) for l in lines], "rows": [norm(l) for l in rows]}


def preservation(im, box):
    out = np.asarray(ImageOps.grayscale(to_src(im)), dtype=np.float32) / 255
    src = np.asarray(ImageOps.grayscale(Image.open(os.path.join(DEMOS, LAYOUT["scene"]))), dtype=np.float32) / 255
    W, H = im.size
    bx = [int(box[0] * SRC_W / W), int(box[1] * SRC_H / H), int(box[2] * SRC_W / W), int(box[3] * SRC_H / H)]
    _, smap = ssim(src, out, data_range=1.0, full=True)
    mask = np.ones_like(smap, bool)
    m = 24
    mask[max(0, bx[1] - m):bx[3] + m, max(0, bx[0] - m):bx[2] + m] = False
    fid = {}
    for k, (x0, y0, x1, y1) in FIDUCIALS.items():
        patch = src[y0:y1, x0:x1]
        pad = 48
        sy0, sx0 = max(0, y0 - pad), max(0, x0 - pad)
        win = out[sy0:y1 + pad, sx0:x1 + pad]
        ncc = match_template(win, patch)
        iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
        fid[k] = {"ncc": round(float(ncc.max()), 3), "shift_px": int(np.hypot(iy + sy0 - y0, ix + sx0 - x0))}
    return {"ssim_outside": round(float(smap[mask].mean()), 3), "fiducials": fid,
            "fiducials_kept": sum(1 for v in fid.values() if v["ncc"] >= 0.6)}


BASE_NAME = "sign_mp4_s1001"


def vs_base(im, base, box):
    """SSIM against the base render outside the plate box (both resized to the base size)."""
    g1 = np.asarray(ImageOps.grayscale(im.resize(base.size, Image.LANCZOS)), dtype=np.float32) / 255
    g2 = np.asarray(ImageOps.grayscale(base), dtype=np.float32) / 255
    _, smap = ssim(g1, g2, data_range=1.0, full=True)
    W, H = base.size
    x0, y0, x1, y1 = [int(v * (W / im.width if i % 2 == 0 else H / im.height)) for i, v in enumerate(box)]
    mask = np.ones_like(smap, bool); m = 32
    mask[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m] = False
    return round(float(smap[mask].mean()), 3)


def region_lab(im, box):
    x0, y0, x1, y1 = box
    return rgb2lab(np.asarray(to_src(im).crop((x0, y0, x1, y1)))).reshape(-1, 3).mean(0)


def neon(lit, unlit):
    res = {}
    for k, box in NEON.items():
        a, b = region_lab(lit, box), region_lab(unlit, box)
        # dE between the two renders, and how much the lit one moved toward the sign's pink
        toward = float(np.dot(a[1:] - b[1:], [NEON_HUE_A, NEON_HUE_B]) / np.hypot(NEON_HUE_A, NEON_HUE_B))
        res[k] = {"dE": round(float(deltaE_ciede2000(a.reshape(1, 1, 3), b.reshape(1, 1, 3))[0, 0]), 2),
                  "toward_pink": round(toward, 2), "dL": round(float(a[0] - b[0]), 2)}
    return res


def seam(comp, base, box):
    g1 = np.asarray(ImageOps.grayscale(comp), dtype=np.float32) / 255
    g2 = np.asarray(ImageOps.grayscale(base), dtype=np.float32) / 255
    x0, y0, x1, y1 = box
    b = 48
    ring = np.zeros(g1.shape, bool)
    ring[max(0, y0 - b):y1 + b, max(0, x0 - b):x1 + b] = True
    ring[y0 + b:y1 - b, x0 + b:x1 - b] = False
    _, smap = ssim(g1, g2, data_range=1.0, full=True)
    return round(float(smap[ring].mean()), 3)


def timings_box(n):
    """Union of the render's own --detail box and the widest one, padded, for the detail sheet."""
    t = json.load(open(os.path.join(OUT, "timings.json")))
    boxes = [t[k]["detail"] for k in t if "_detail" in k and "detail" in t[k]]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    p = 60
    return (max(0, x0 - p), max(0, y0 - p), x1 + p, y1 + p)


def sheet(names, path, crops=None, label_fn=None, cols=4, w=640):
    tiles = []
    for n in names:
        im = load(n)
        if crops:
            im = im.crop(crops[n])
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        txt = label_fn(n) if label_fn else n
        d.rectangle((0, 0, im.width, 22), fill=(0, 0, 0))
        d.text((6, 4), txt, fill=(255, 255, 255))
        tiles.append(im)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % cols) * w, (i // cols) * h))
    S.save(path)


def main():
    timings = json.load(open(os.path.join(OUT, "timings.json")))
    names = [n for n in timings if os.path.exists(os.path.join(OUT, n + ".png"))]
    scores = {}
    for n in names:
        im = load(n)
        s = {"size": im.size, "wall_s": timings[n]["wall_s"], "mp": timings[n]["mp"], "seed": timings[n]["seed"]}
        if n.startswith("sign_"):
            box, found = find_plate(im)
            s["plate_box"], s["plate_found"] = box, found
            s["ocr"], s["ocr_lines"] = ocr_plate(im, box)
            # cap heights of each tier in output pixels, from the located plate
            s["cap_px"] = {t["tier"]: round(t["cap_ratio"] * (box[3] - box[1])) for t in TIERS}
            if "_detail" in n:
                base = n.split("_detail")[0]
                s["seam_ssim"] = seam(im, load(base), timings[n]["detail"])
            else:
                s["preserve"] = preservation(im, box)
        if timings[n].get("pass"):
            s["pass"] = timings[n]["pass"]
            s["vs_base_outside"] = vs_base(im, load(BASE_NAME), box)
        scores[n] = s
        print(n, s.get("ocr"), flush=True)
    if "neon_lit" in scores and "neon_unlit" in scores:
        scores["neon_lit"]["neon"] = neon(load("neon_lit"), load("neon_unlit"))
    json.dump(scores, open(os.path.join(OUT, "scores.json"), "w"), indent=1)
    if os.path.exists(os.path.join(OUT, "_ocr_tmp.png")):
        os.remove(os.path.join(OUT, "_ocr_tmp.png"))
    report(scores)


def report(s):
    tiers = [t["tier"] for t in TIERS]
    L = ["# h3edit benchmark", ""]
    ladder = [n for n in s if n.startswith("sign_mp") and n.endswith("_s1001") and "detail" not in n]
    L += ["## 1. Lettering ladder (seed 1001)", "", "Character accuracy per tier (1.0 = every character right). Cap height in output px in brackets.", "",
          "| MP | output | " + " | ".join(tiers) + " | wall s |", "|---|---|" + "---|" * len(tiers) + "---|"]
    for n in sorted(ladder, key=lambda n: s[n]["mp"]):
        r = s[n]
        L.append(f"| {r['mp']:g} | {r['size'][0]}x{r['size'][1]} | " +
                 " | ".join(f"{r['ocr'][t]:.2f} ({r['cap_px'][t]}px)" for t in tiers) + f" | {r['wall_s']:.0f} |")
    seeds = sorted([n for n in s if n.startswith("sign_mp4_") and "detail" not in n], key=lambda n: s[n]["seed"])
    L += ["", "## 2. Seed stability (4 MP, 8 seeds)", "", "| seed | " + " | ".join(tiers) + " | S+XS+XXS pass (≥0.9) |",
          "|---|" + "---|" * len(tiers) + "---|"]
    for n in seeds:
        r = s[n]
        L.append(f"| {r['seed']} | " + " | ".join(f"{r['ocr'][t]:.2f}" for t in tiers) +
                 f" | {sum(r['ocr'][t] >= 0.9 for t in ('S','XS','XXS'))}/3 |")
    if seeds:
        arr = {t: [s[n]["ocr"][t] for n in seeds] for t in tiers}
        L.append("| **mean** | " + " | ".join(f"{np.mean(arr[t]):.2f}" for t in tiers) + " | |")
        L.append("| **worst** | " + " | ".join(f"{np.min(arr[t]):.2f}" for t in tiers) + " | |")
    det = sorted(n for n in s if "_detail" in n)
    if det:
        b = s[det[0].split("_detail")[0]]
        L += ["", f"## 3. Detail-pass rescue (`--detail` on {det[0].split('_detail')[0]}, the weakest seed)", "",
              "| pass | " + " | ".join(tiers) + " | seam SSIM |", "|---|" + "---|" * len(tiers) + "---|",
              "| single pass | " + " | ".join(f"{b['ocr'][t]:.2f}" for t in tiers) + " | |"]
        for n in det:
            L.append(f"| + --detail{' --mp 4, box incl. plate edge' if n.endswith('edge') else ' --mp 4' if n.endswith('4') else ' (2 MP)'} | " +
                     " | ".join(f"{s[n]['ocr'][t]:.2f}" for t in tiers) + f" | {s[n]['seam_ssim']:.3f} |")
    pres = [n for n in seeds + ladder if "preserve" in s[n]]
    if pres:
        fk = list(FIDUCIALS)
        L += ["", "## 4. Preservation outside the edit (what survives whole-frame regeneration)", "",
              "Outside-plate SSIM vs source, and fiducial patches re-found by NCC (≥0.6 = kept).", "",
              "| render | SSIM outside | kept /8 | " + " | ".join(fk) + " |", "|---|---|---|" + "---|" * len(fk)]
        for n in sorted(set(pres), key=lambda n: (s[n]["mp"], s[n]["seed"])):
            p = s[n]["preserve"]
            L.append(f"| {n} | {p['ssim_outside']:.2f} | {p['fiducials_kept']} | " +
                     " | ".join(f"{p['fiducials'][k]['ncc']:.2f}" for k in fk) + " |")
    if "neon" in s.get("neon_lit", {}):
        nn = s["neon_lit"]["neon"]
        L += ["", "## 5. Light as a source (neon lit vs switched-off, same seed)", "",
              "Mean Lab shift of the lit render relative to the unlit one. `toward_pink` > 0 = region moved toward the sign's colour; `far_control` should stay ~0.", "",
              "| region | ΔE00 | toward pink | ΔL |", "|---|---|---|---|"]
        for k, v in nn.items():
            L.append(f"| {k} | {v['dE']:.1f} | {v['toward_pink']:+.1f} | {v['dL']:+.1f} |")
    passes = [n for n in s if s[n].get("pass")]
    if passes:
        L += ["", "## 6. Second-pass candidates (all on sign_mp4_s1001)", "",
              "| pass | output | " + " | ".join(tiers) + " | SSIM vs base outside plate | wall s |", "|---|---|" + "---|" * len(tiers) + "---|---|"]
        b = s[BASE_NAME]
        L.append(f"| base single pass | {b['size'][0]}x{b['size'][1]} | " + " | ".join(f"{b['ocr'][t]:.2f}" for t in tiers) + " | 1.000 | " + f"{b['wall_s']:.0f} |")
        for n in passes:
            r = s[n]
            L.append(f"| {n.replace(BASE_NAME + '_', '')} | {r['size'][0]}x{r['size'][1]} | " + " | ".join(f"{r['ocr'][t]:.2f}" for t in tiers) +
                     f" | {r['vs_base_outside']:.3f} | {r['wall_s']:.0f} |")
    L += ["", "## Speed", "", "| render | MP | size | wall s |", "|---|---|---|---|"]
    for n in s:
        L.append(f"| {n} | {s[n]['mp']} | {s[n]['size'][0]}x{s[n]['size'][1]} | {s[n]['wall_s']:.0f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    # contact sheets: plate crops of the ladder + seeds, full frames of the rest
    plate = {n: s[n]["plate_box"] for n in s if "plate_box" in s[n]}
    if ladder:
        sheet(sorted(ladder, key=lambda n: s[n]["mp"]), os.path.join(OUT, "sheet_ladder.png"), plate,
              lambda n: f"{s[n]['mp']:g} MP  S={s[n]['ocr']['S']:.2f} XS={s[n]['ocr']['XS']:.2f} XXS={s[n]['ocr']['XXS']:.2f}", cols=3, w=800)
    if det:
        # the detail box region of each pass, with the verdict in the label
        base = det[0].split("_detail")[0]
        bx = timings_box(det[-1])
        crops = {n: bx for n in det}
        verdict = {n: ("PASS" if n.endswith("edge") else "FAIL: plate hallucinated inside the crop") for n in det}
        sheet(det, os.path.join(OUT, "sheet_detail.png"), crops,
              lambda n: f"{'--detail (2 MP)' if n.endswith('_detail') else '--detail --mp 4' if n.endswith('4') else '--detail --mp 4, box incl. plate edge'}  {verdict[n]}", cols=3, w=800)
    if seeds:
        sheet(seeds, os.path.join(OUT, "sheet_seeds.png"), plate,
              lambda n: f"seed {s[n]['seed']}  S={s[n]['ocr']['S']:.2f} XS={s[n]['ocr']['XS']:.2f} XXS={s[n]['ocr']['XXS']:.2f}", cols=4, w=640)
    passes = [n for n in s if s[n].get("pass")]
    if passes:
        bx = plate[BASE_NAME]; pad = 80
        crop = (max(0, bx[0] - pad), max(0, bx[1] - pad), bx[2] + pad, bx[3] + pad)
        names = [BASE_NAME] + passes
        crops = {n: tuple(int(v * (s[n]["size"][i % 2] / s[BASE_NAME]["size"][i % 2])) for i, v in enumerate(crop)) for n in names}
        sheet(names, os.path.join(OUT, "sheet_passes.png"), crops,
              lambda n: f"{n.replace(BASE_NAME + '_', '') if n != BASE_NAME else 'base'}  XS={s[n]['ocr']['XS']:.2f} XXS={s[n]['ocr']['XXS']:.2f} outside-SSIM={s[n].get('vs_base_outside', 1):.2f}", cols=3, w=800)
    if "neon_lit" in s and "neon_unlit" in s:
        sheet(["neon_lit", "neon_unlit"], os.path.join(OUT, "sheet_neon.png"), cols=2, w=900)
    print("\n".join(L))


if __name__ == "__main__":
    main()
