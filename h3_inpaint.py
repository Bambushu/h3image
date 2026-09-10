#!/usr/bin/env python3
"""h3-inpaint: build or repair a large image on ONE canvas, one masked H3 pass at a time.

Each pass is `h3edit --inpaint` on a ~4 MP window cut around a box of the canvas; only the box
(grown, feathered) goes back onto the canvas in pixel space, so nothing outside a box ever moves.
Passes run back to front: a later box overwrites whatever it covers.

  h3-inpaint init  DIR --canvas start.png            # project: plan.json, refs/, prompts/, out/
  h3-inpaint add   DIR NAME --box X0,Y0,X1,Y1 --prompt FILE|TEXT [-r ref.png ...] [--denoise 1.0]
  h3-inpaint run   DIR [--only NAME] [--redo NAME] [--pod NAME]   # every pass not yet done, in plan order
                                                     # --pod NAME = podenv.NAME.sh, whole canvas per pass (~65 s at 16 MP)
  h3-inpaint show  DIR [NAME]                        # 1:1 crop of a pass's box (or 12 audit tiles)
  h3-inpaint revert DIR NAME [--to PASS]             # pixel-space revert of a bad pass, logged
  h3-inpaint score DIR                               # outside/seam SSIM per pass + making-of sheet

The proof build (43 passes, 5440x3072) and every rule learned on it: benchmark/nachtwacht/README.md.
"""
import argparse, json, mimetypes, os, re, subprocess, sys, time, urllib.request, uuid
from PIL import Image, ImageDraw, ImageFilter

WINDOW_MP = 4.0
# --pod: the pass runs on the whole canvas at once through ~/renderpod/h3/drive.py on the shipped
# single-image edit graph (extra nodes via H3_ADDNODES, the pattern the pod driver validated).
H3_KIT = os.path.expanduser(os.environ.get("H3_KIT", "~/renderpod/h3"))
POD_WF = f"{H3_KIT}/workflows/h3_single_image_edit_fullint8_linked.json"
POD_PY = os.path.expanduser(os.environ.get("H3EDIT_POD_PY", "~/klipsmid-render-venv/bin/python"))
POD_INT8 = {"10": {"unet_name": "minimax_h3_fl2va_int8_convrot.safetensors"},
            "11": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"},
            "2": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}}
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


# ----------------------------------------------------------------------------- project state
def paths(d):
    return {k: os.path.join(d, v) for k, v in dict(plan="plan.json", state="state.json", refs="refs", prompts="prompts", out="out").items()}


def load(d):
    p = paths(d)
    if not os.path.exists(p["plan"]):
        sys.exit(f"{d}: no plan.json (run `h3-inpaint init` first)")
    plan = json.load(open(p["plan"]))
    state = json.load(open(p["state"])) if os.path.exists(p["state"]) else {"done": [], "canvas": plan["canvas"], "log": []}
    return p, plan, state


def save(p, plan=None, state=None):
    if plan is not None:
        json.dump(plan, open(p["plan"], "w"), indent=1)
    if state is not None:
        json.dump(state, open(p["state"], "w"), indent=1)


# ----------------------------------------------------------------------------- geometry
def window_for(box, W, H, mp=WINDOW_MP, margin=128):
    """Smallest ~mp window (sides multiples of 32) containing box+margin, clamped to the canvas."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 + 2 * margin, y1 - y0 + 2 * margin
    best = None
    for ar in (16 / 9, 3 / 2, 4 / 3, 1.0, 3 / 4, 2 / 3, 9 / 16):
        w = int((mp * 1e6 * ar) ** 0.5) // 32 * 32
        h = int((mp * 1e6 / ar) ** 0.5) // 32 * 32
        if w >= bw and h >= bh and (best is None or w * h < best[0] * best[1]):
            best = (w, h)
    if best is None:
        sys.exit(f"box {box} does not fit a {mp} MP window; split it into two passes")
    w, h = min(best[0], W), min(best[1], H)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    wx0 = max(0, min(W - w, cx - w // 2)) // 32 * 32
    wy0 = max(0, min(H - h, cy - h // 2)) // 32 * 32
    return [wx0, wy0, wx0 + w, wy0 + h]


def paste_back(prev, ren, box, grow, feather):
    """Feathered pixel-space paste of the box: outside stays bit-exact, no VAE round-trip."""
    if ren.size != prev.size:
        ren = ren.resize(prev.size, Image.LANCZOS)
    x0, y0, x1, y1 = box
    m = Image.new("L", prev.size, 0)
    ImageDraw.Draw(m).rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(feather / 2))
    comp = prev.copy(); comp.paste(ren, (0, 0), m)
    return comp


def palette_card(canvas, out):
    """A colours-only reference from the canvas itself, for passes with no reference image."""
    im = Image.open(canvas).convert("RGB")
    q = im.resize((256, 256)).quantize(colors=8).convert("RGB")
    cols = sorted(q.getcolors(65536), reverse=True)[:8]
    card = Image.new("RGB", (1024, 512))
    d = ImageDraw.Draw(card)
    for i, (_, c) in enumerate(cols):
        d.rectangle((i % 4 * 256, i // 4 * 256, i % 4 * 256 + 256, i // 4 * 256 + 256), fill=c)
    card.save(out)
    return out


# ----------------------------------------------------------------------------- one pass
# ----------------------------------------------------------------------------- pod backend
def podenv(name):
    env = {}
    for line in open(f"{H3_KIT}/podenv.{name}.sh"):
        m = re.match(r"export (\w+)=(.*)", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip('"')
    return f"https://{env['POD_ID']}-{env['COMFY_PORT']}.proxy.runpod.net"


def comfy(url, path, data=None, headers=None):
    req = urllib.request.Request(url.rstrip("/") + path, data, {"User-Agent": "curl/8", **(headers or {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read() or b"{}")


def upload(url, path, tag):
    ext = os.path.splitext(path)[1].lower()
    bnd = uuid.uuid4().hex
    fn = f"{tag}_{uuid.uuid4().hex[:8]}{ext}"
    body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {mimetypes.guess_type(path)[0] or 'application/octet-stream'}\r\n\r\n").encode() \
        + open(path, "rb").read() \
        + (f"\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{bnd}\r\n"
           f"Content-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{bnd}--\r\n").encode()
    r = comfy(url, "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={bnd}"})
    return r.get("name", fn)


def pod_pass(url, p, plan, e, canvas, box, refs, prompt_path, seed, out_raw):
    """One masked pass on the WHOLE canvas: VAEEncode(canvas) -> H3V2VInit(mask over the box) ->
    sampler at `denoise`; refs are <Picture N>, the canvas only enters through the latent."""
    W, H = Image.open(canvas).size
    grow, feather = plan.get("grow", 32), plan.get("feather", 64)
    src = upload(url, canvas, f"h3i_{e['name']}_src")
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle((box[0] - grow, box[1] - grow, box[2] + grow, box[3] + grow), fill=255)
    mp = os.path.join(p["out"], f"{e['name']}_mask.png"); m.save(mp)
    nodeset = {**POD_INT8, "17": {"image": src}, "12": {"noise_seed": seed}, "16": {"filename_prefix": f"h3_edit/h3i_{e['name']}"},
               "1": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": plan.get("window_mp", WINDOW_MP)},
               "7": {"steps": 20, "scheduler": "simple", "denoise": e.get("denoise", 1.0)}, "6": {"sampler_name": "euler"}}
    addnodes = {"9300": {"class_type": "VAEEncode", "inputs": {"pixels": ["17", 0], "vae": ["2", 0]}},
                "9302": {"class_type": "LoadImageMask", "inputs": {"image": upload(url, mp, f"h3i_{e['name']}_mask"), "channel": "red"}},
                "9303": {"class_type": "H3V2VInit", "inputs": {"samples": ["9300", 0], "mask": ["9302", 0], "mask_feather": feather}}}
    apiset = {"13": {"width": W, "height": H}, "8": {"latent_image": ["9303", 0]}}
    for i, r in enumerate(refs):
        addnodes[f"91{i:02d}"] = {"class_type": "LoadImage", "inputs": {"image": upload(url, r, f"h3i_ref_{os.path.splitext(os.path.basename(r))[0]}")}}
        apiset["13"][f"ref_images.ref_image_{i}"] = [f"91{i:02d}", 0]
    env = dict(os.environ, H3_URL=url, H3_NODESET=json.dumps(nodeset), H3_NODEMODES=json.dumps({"18": 4, "19": 4}),
               H3_ADDNODES=json.dumps(addnodes), H3_APISET=json.dumps(apiset), H3_APIEXPECT=json.dumps(apiset))
    r = subprocess.run([POD_PY, f"{H3_KIT}/drive.py", "run", POD_WF, prompt_path, "1", str(plan.get("window_mp", WINDOW_MP)), str(seed), f"h3i_{e['name']}"],
                       env=env, capture_output=True, text=True, cwd=H3_KIT)
    mm = UUID_RE.search(r.stdout)
    if r.returncode != 0 or not mm:
        sys.exit(f"{e['name']}: pod submit failed:\n" + (r.stdout + r.stderr)[-3000:])
    pid = mm.group(0)
    t0 = time.time()
    while time.time() - t0 < 3600:
        h = comfy(url, f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                sys.exit(f"{e['name']}: pod render failed: " + json.dumps(st)[-3000:])
            im = next((o for n in h[pid]["outputs"].values() for o in n.get("images", [])), None)
            q = f"/view?filename={urllib.request.quote(im['filename'])}&subfolder={urllib.request.quote(im.get('subfolder', ''))}&type=output"
            open(out_raw, "wb").write(urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + q, headers={"User-Agent": "curl/8"}), timeout=600).read())
            return
        time.sleep(8)
    sys.exit(f"{e['name']}: pod timeout")


def run_pass(p, plan, state, e, pod_url=None):
    im = Image.open(state["canvas"]).convert("RGB")
    W, H = im.size
    box = e["box"]
    if not (0 <= box[0] < box[2] <= W and 0 <= box[1] < box[3] <= H):
        sys.exit(f"{e['name']}: box {box} is outside the {W}x{H} canvas")
    prompt = e["prompt"]
    pf = os.path.join(p["prompts"], prompt)
    if os.path.exists(pf):
        prompt = open(pf).read()
    refs = [os.path.join(p["refs"], r) for r in e.get("refs") or []]
    if not refs:
        refs = [palette_card(state["canvas"], os.path.join(p["refs"], "_palette.png"))]
    seed = e.get("seed") or (plan.get("seed", 4200) + sum(map(ord, e["name"])) * 7919) % (2**31)
    t0 = time.time()
    if pod_url:
        if not os.path.exists(pf):
            pf = os.path.join(p["out"], f"{e['name']}_prompt.txt"); open(pf, "w").write(prompt)
        print(f"=== {e['name']}  box={box}  whole canvas {W}x{H} on the pod  denoise={e.get('denoise', 1.0)}  refs={[os.path.basename(r) for r in refs]}", flush=True)
        raw = os.path.join(p["out"], f"{e['name']}_raw.png")
        pod_pass(pod_url, p, plan, e, state["canvas"], box, refs, pf, seed, raw)
        comp = paste_back(im, Image.open(raw).convert("RGB"), box, plan.get("grow", 32), plan.get("feather", 64))
        win = None
    else:
        comp, win = local_pass(p, plan, e, im, box, refs, prompt, seed)
    out = os.path.join(p["out"], f"{e['name']}.png"); comp.save(out)
    wall = round(time.time() - t0)
    state["done"].append(e["name"]); state["canvas"] = out
    state["log"].append({"name": e["name"], "kind": "inpaint", "box": box, "window": win, "refs": e.get("refs") or [],
                         "denoise": e.get("denoise", 1.0), "seed": seed, "wall_s": wall, "size": [W, H], "backend": "pod" if pod_url else "local"})
    print(f"    -> {out}  ({wall}s)", flush=True)


def local_pass(p, plan, e, im, box, refs, prompt, seed):
    """Cut a ~4 MP window around the box, `h3edit --inpaint` on it, put the window back whole."""
    W, H = im.size
    win = window_for(box, W, H, plan.get("window_mp", WINDOW_MP))
    wx0, wy0, wx1, wy1 = win
    crop_p = os.path.join(p["out"], f"{e['name']}_window.png"); im.crop(win).save(crop_p)
    rel = [box[0] - wx0, box[1] - wy0, box[2] - wx0, box[3] - wy0]
    out_p = os.path.join(p["out"], f"{e['name']}_window_out.png")
    c = ["h3edit", prompt, "--inpaint", ",".join(map(str, rel)), "--source", crop_p,
         "--denoise", str(e.get("denoise", 1.0)), "--grow", str(plan.get("grow", 32)), "--feather", str(plan.get("feather", 64)),
         "--seed", str(seed), "--name", f"inpaint_{e['name']}", "-o", out_p, "--wait"]
    for r in refs:
        c += ["-r", r]
    print(f"=== {e['name']}  box={box}  window={win} ({wx1-wx0}x{wy1-wy0})  denoise={e.get('denoise', 1.0)}  refs={[os.path.basename(r) for r in refs]}", flush=True)
    r = subprocess.run(c, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_p):
        sys.exit(f"{e['name']}: h3edit failed:\n" + (r.stdout + r.stderr)[-2000:])
    comp = im.copy(); comp.paste(Image.open(out_p).convert("RGB"), (wx0, wy0))
    return comp, win


def cmd_degrid(a):
    """Whole-canvas cell-grid filter: deblock at each --cell (no notch: on a whole canvas it rings). For a
    canvas that was 2x-upscaled from a 4 MP render the cells sit at 32 px; boxes painted since sit at 16."""
    import h3edit
    p, plan, state = load(a.dir)
    im = Image.open(state["canvas"]).convert("RGB")
    for c in a.cells:                      # deblock only: a whole-canvas notch rings in smooth skies
        im = h3edit.deblock_cells(im, cell=c)
    n = sum(1 for e in state["log"] if e["name"].startswith("degrid")) + 1
    name = f"degrid{n}"; out = os.path.join(p["out"], f"{name}.png"); im.save(out)
    state["done"].append(name); state["canvas"] = out
    state["log"].append({"name": name, "kind": "filter", "box": None, "refs": [], "denoise": None, "wall_s": 0, "size": list(im.size), "cells": a.cells})
    save(p, state=state); print(f"{out}  (deblock cells {a.cells})")


# ----------------------------------------------------------------------------- commands
def cmd_init(a):
    p = paths(a.dir)
    for k in ("refs", "prompts", "out"):
        os.makedirs(p[k], exist_ok=True)
    im = Image.open(a.canvas).convert("RGB")
    W, H = im.size
    if W % 32 or H % 32:
        W2, H2 = W // 32 * 32, H // 32 * 32
        print(f"canvas {W}x{H} cropped to {W2}x{H2} (sides must be multiples of 32)")
        im = im.crop((0, 0, W2, H2))
    start = os.path.join(p["out"], "canvas.png"); im.save(start)
    plan = {"canvas": start, "grow": 32, "feather": 64, "window_mp": WINDOW_MP, "seed": 4200, "passes": []}
    save(p, plan, {"done": [], "canvas": start, "log": [{"name": "canvas", "kind": "canvas", "box": None, "refs": [], "denoise": None, "wall_s": 0, "size": list(im.size)}]})
    print(f"{a.dir}: plan.json + refs/ prompts/ out/ ; canvas {im.size[0]}x{im.size[1]}")


def cmd_add(a):
    p, plan, state = load(a.dir)
    if any(e["name"] == a.name for e in plan["passes"]):
        sys.exit(f"pass {a.name} already in the plan")
    e = {"name": a.name, "box": a.box, "refs": [os.path.basename(r) for r in a.refs], "denoise": a.denoise, "prompt": a.prompt}
    for r in a.refs:
        dst = os.path.join(p["refs"], os.path.basename(r))
        if os.path.abspath(r) != os.path.abspath(dst):
            Image.open(r).save(dst)
    plan["passes"].append(e); save(p, plan)
    print(f"added {a.name} box={a.box} denoise={a.denoise} refs={e['refs']}")


def cmd_run(a):
    p, plan, state = load(a.dir)
    pod_url = podenv(a.pod) if a.pod else None
    names = [e["name"] for e in plan["passes"]]
    if a.redo:
        keep = names[:names.index(a.redo)]
        state["done"] = [d for d in state["done"] if d in keep or d.endswith("_revert") and d[:-7] in keep]
        state["log"] = [l for l in state["log"] if l["name"] == "canvas" or l["name"] in state["done"]]
        state["canvas"] = os.path.join(p["out"], f"{state['done'][-1]}.png") if state["done"] else plan["canvas"]
        save(p, state=state)
    for e in plan["passes"]:
        if e["name"] in state["done"] or (a.only and a.only != e["name"]):
            continue
        run_pass(p, plan, state, e, pod_url)
        save(p, state=state)
    print(f"canvas: {state['canvas']}")


def cmd_show(a):
    p, plan, state = load(a.dir)
    im = Image.open(state["canvas"]).convert("RGB")
    W, H = im.size
    if a.name:
        e = next((l for l in state["log"] if l["name"] == a.name), None) or next((e for e in plan["passes"] if e["name"] == a.name), None)
        if not e or not e.get("box"):
            sys.exit(f"no box for {a.name}")
        x0, y0, x1, y1 = e["box"]; m = 96
        out = os.path.join(p["out"], f"{a.name}_show.png")
        im.crop((max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))).save(out)
        print(out)
        return
    cols, rows = 4, 3
    tw, th = W // cols, H // rows
    for r in range(rows):
        for c in range(cols):
            out = os.path.join(p["out"], f"_audit_{r}{c}.png")
            im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)).save(out)
    print(f"{rows*cols} audit tiles in {p['out']}/_audit_RC.png (look at every one at 1:1)")


def cmd_revert(a):
    p, plan, state = load(a.dir)
    e = next((l for l in state["log"] if l["name"] == a.name), None)
    if not e:
        sys.exit(f"{a.name} not in the log")
    to = a.to or state["log"][[l["name"] for l in state["log"]].index(a.name) - 1]["name"]
    src = plan["canvas"] if to == "canvas" else os.path.join(p["out"], f"{to}.png")
    prev = Image.open(state["canvas"]).convert("RGB")
    # hard paste of the grown+feathered region: the box comes back bit-exact from `to`
    x0, y0, x1, y1 = e["box"]; g = plan.get("grow", 32) + 2 * plan.get("feather", 64)   # the feather's gaussian tails reach ~2x feather
    comp = prev.copy(); comp.paste(Image.open(src).convert("RGB").crop((x0 - g, y0 - g, x1 + g, y1 + g)), (x0 - g, y0 - g))
    out = os.path.join(p["out"], f"{a.name}_revert.png"); comp.save(out)
    state["done"].append(f"{a.name}_revert"); state["canvas"] = out
    state["log"].append({"name": f"{a.name}_revert", "kind": "revert", "box": e["box"], "refs": [], "denoise": None, "wall_s": 0, "size": list(prev.size), "from": to})
    save(p, state=state)
    print(f"reverted {a.name}'s box to {to} -> {out}. Add a new pass under a new name (new seed) to retry.")


def cmd_score(a):
    try:
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        sys.exit("score needs numpy and scikit-image: uv tool install --force 'h3edit[score]' (or pip install numpy scikit-image)")
    p, plan, state = load(a.dir)
    prev, rows, tiles = None, [], []
    for e in state["log"]:
        cur_p = plan["canvas"] if e["name"] == "canvas" else os.path.join(p["out"], e["name"] + ".png")
        im = Image.open(cur_p).convert("L")
        small = (im.width // 4, im.height // 4)
        g = np.asarray(im.resize(small, Image.LANCZOS), dtype=np.float32) / 255
        row = dict(e)
        if prev is not None and e["box"]:
            _, smap = ssim(prev, g, data_range=1.0, full=True)
            x0, y0, x1, y1 = [v // 4 for v in e["box"]]
            gr, ring = plan.get("grow", 32) // 4, 24
            outside = np.ones_like(smap, bool); outside[max(0, y0 - gr - ring):y1 + gr + ring, max(0, x0 - gr - ring):x1 + gr + ring] = False
            seam = np.zeros_like(smap, bool); seam[max(0, y0 - gr - ring):y1 + gr + ring, max(0, x0 - gr - ring):x1 + gr + ring] = True
            seam[max(0, y0 - gr):y1 + gr, max(0, x0 - gr):x1 + gr] = False
            row["outside_ssim"], row["seam_ssim"] = round(float(smap[outside].mean()), 3), round(float(smap[seam].mean()), 3)
            print(f"{e['name']:20s} outside {row['outside_ssim']}  seam {row['seam_ssim']}  {e.get('wall_s', 0)}s")
        rows.append(row); prev = g
        t = Image.open(cur_p).convert("RGB"); t = t.resize((640, int(t.height * 640 / t.width)), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        if e["box"]:
            s = 640 / e["size"][0]; d.rectangle([v * s for v in e["box"]], outline=(255, 220, 0), width=2)
        d.rectangle((0, 0, 640, 20), fill=(0, 0, 0)); d.text((6, 4), f"{len(tiles)+1:02d} {e['name']}", fill=(255, 255, 255))
        tiles.append(t)
    json.dump(rows, open(os.path.join(p["out"], "scores.json"), "w"), indent=1)
    cols = 4; h = tiles[0].height; n = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * 640, n * h), (20, 20, 20))
    for i, t in enumerate(tiles):
        S.paste(t, ((i % cols) * 640, (i // cols) * h))
    S.save(os.path.join(p["out"], "sheet_makingof.png"))
    print(f"scores.json + sheet_makingof.png in {p['out']}  (outside SSIM must read 1.000: a lower value means a box leaked)")


def main():
    ap = argparse.ArgumentParser(description="Build or repair a large image on one canvas, one masked H3 pass at a time (see README).")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("init", help="new project from a starting canvas"); s.add_argument("dir"); s.add_argument("--canvas", required=True); s.set_defaults(f=cmd_init)
    s = sp.add_parser("add", help="append a pass to the plan"); s.add_argument("dir"); s.add_argument("name")
    s.add_argument("--box", required=True, type=lambda v: [int(x) for x in v.split(",")], metavar="X0,Y0,X1,Y1")
    s.add_argument("--prompt", required=True, help="prompt text, or a file name in DIR/prompts/")
    s.add_argument("-r", "--ref", dest="refs", action="append", default=[], help="reference image (copied into DIR/refs/); none = a palette card from the canvas")
    s.add_argument("--denoise", type=float, default=1.0, help="1.0 composes a new thing in the box; 0.8-0.85 adds a prop to bare ground or corrects lettering")
    s.set_defaults(f=cmd_add)
    s = sp.add_parser("run", help="run every pass not yet done"); s.add_argument("dir"); s.add_argument("--only"); s.add_argument("--redo", help="drop this pass and everything after it, then run")
    s.add_argument("--pod", metavar="NAME", help="render on the pod in ~/renderpod/h3/podenv.NAME.sh: whole canvas per pass, base model 20 steps (default: local h3edit on a ~4 MP window)"); s.set_defaults(f=cmd_run)
    s = sp.add_parser("show", help="1:1 crop of a pass's box, or 12 audit tiles"); s.add_argument("dir"); s.add_argument("name", nargs="?"); s.set_defaults(f=cmd_show)
    s = sp.add_parser("revert", help="pixel-space revert of a bad pass"); s.add_argument("dir"); s.add_argument("name"); s.add_argument("--to", help="pass whose canvas to restore the box from (default: the one before)"); s.set_defaults(f=cmd_revert)
    s = sp.add_parser("degrid", help="whole-canvas cell-grid deblock, logged as a pass"); s.add_argument("dir")
    s.add_argument("--cells", type=lambda v: [int(x) for x in v.split(",")], default=[32, 16], help="cell sizes to deblock, e.g. 32,16 for a 2x-upscaled base with 16 px boxes (default)"); s.set_defaults(f=cmd_degrid)
    s = sp.add_parser("score", help="outside/seam SSIM per pass + making-of sheet (needs numpy, scikit-image)"); s.add_argument("dir"); s.set_defaults(f=cmd_score)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
