#!/usr/bin/env python3
"""Build a Night-Watch-scale group painting from a blank canvas with H3 only, one masked pass at a
time on a single 16 MP canvas. Nothing is regenerated whole after the canvas pass.

  python3 build.py --pod nw            # runs every pass not yet in out/state.json
  python3 build.py --pod nw --only captain
  python3 build.py --pod nw --redo dog # drop a pass (and everything after it) and rebuild

Pass kinds (all through ~/renderpod/h3/drive.py on the shipped edit graph, extra nodes by the
same H3_ADDNODES pattern the pod driver uses):
  canvas    one R2V sample at --base-mp with the palette card as the only reference, then the
            quality-lane refine (LBH 2x latent upscale + 4-step er_sde) -> the 2x canvas
  inpaint   VAEEncode(canvas) -> H3V2VInit(mask over the box) -> sampler from `denoise`;
            the reference stills are <Picture 1>.., the canvas is NOT a reference
  harmonize full-frame H3V2VInit without a mask at low denoise: unify brushwork and light
"""
import argparse, importlib.util, json, os, sys, time
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, REFS, PROMPTS = os.path.join(HERE, "out"), os.path.join(HERE, "refs"), os.path.join(HERE, "prompts")
spec = importlib.util.spec_from_file_location("passes", os.path.join(HERE, "..", "passes.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
pod = P.pod
SEED = 4200

STYLE = ("an oil painting in the manner of a Dutch Golden Age militia group portrait: deep umber and "
         "black shadows, warm ochre and gold light falling from the upper left, visible brushwork, "
         "glazed highlights on metal and lace, a fine craquelure over the whole surface")

# (name, kind, box as fractions of W,H or None, refs, denoise, prompt file)
# Back to front: a later box overwrites an earlier one in pixel space, so front figures go last.
# Every pass's render is pasted back onto the previous canvas through a feathered mask of its box
# (pixel space): the model sees the frozen latent for context, but the canvas outside the box
# never takes a VAE round-trip. Without this the whole painting drifted dark and muddy by pass 7
# (each pass re-decoded the full latent; measured 2026-09-10).
PLAN = [
    ("canvas",    "canvas",    None,                      ["palette.png"],      None, "canvas.txt"),
    ("banner",    "inpaint",   (0.36, 0.00, 0.64, 0.30),  ["palette.png"],      1.0,  "banner.txt"),
    ("steps",     "inpaint",   (0.00, 0.70, 1.00, 1.00),  ["palette.png"],      1.0,  "steps.txt"),
    ("pikes",     "inpaint",   (0.56, 0.00, 0.80, 0.34),  ["palette.png"],      1.0,  "pikes.txt"),
    ("shield",    "inpaint",   (0.80, 0.02, 0.96, 0.22),  ["shield_art.png"],   1.0,  "shield.txt"),
    ("ensign",    "inpaint",   (0.38, 0.03, 0.58, 0.48),  ["ray_b.jpg"],        1.0,  "ensign.txt"),
    ("ranks_r",   "inpaint",   (0.62, 0.12, 0.82, 0.72),  ["palette.png"],      1.0,  "ranks_r.txt"),
    ("ranks_l",   "inpaint",   (0.16, 0.08, 0.36, 0.50),  ["palette.png"],      1.0,  "ranks_l.txt"),
    ("girl",      "inpaint",   (0.14, 0.32, 0.31, 0.88),  ["rosalie.png"],      1.0,  "girl.txt"),
    ("musketeer", "inpaint",   (0.00, 0.20, 0.16, 1.00),  ["lakem_a.jpg"],      1.0,  "musketeer.txt"),
    ("drummer",   "inpaint",   (0.80, 0.26, 1.00, 1.00),  ["speaker.jpg"],      1.0,  "drummer.txt"),
    ("lieutenant","inpaint",   (0.50, 0.26, 0.66, 1.00),  ["ray_a.jpg"],        1.0,  "lieutenant.txt"),
    ("captain",   "inpaint",   (0.30, 0.22, 0.50, 1.00),  ["lakem_b.jpg"],      1.0,  "captain.txt"),
    ("dog",       "inpaint",   (0.66, 0.72, 0.84, 1.00),  ["palette.png"],      1.0,  "dog.txt"),
    ("harmonize", "harmonize", None,                      [],                   0.35, "harmonize.txt"),
]
INT8 = P.INT8


def state_path():
    return os.path.join(OUT, "state.json")


def load_state():
    return json.load(open(state_path())) if os.path.exists(state_path()) else {"done": [], "canvas": None, "log": []}


def save_state(s):
    json.dump(s, open(state_path(), "w"), indent=1)


def px_box(frac, W, H):
    x0, y0, x1, y1 = frac
    snap = lambda v: int(round(v / 16) * 16)
    return [snap(x0 * W), snap(y0 * H), min(W, snap(x1 * W)), min(H, snap(y1 * H))]


def canvas_pass(url, name, refs, prompt, a):
    palette = pod.upload(url, os.path.join(REFS, refs[0]), "nw_palette")
    nodeset = {**INT8, "17": {"image": palette}, "12": {"noise_seed": SEED}, "16": {"filename_prefix": f"h3_edit/nw_{name}"},
               "1": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": a.base_mp},
               "7": {"steps": 20, "scheduler": "simple"}, "6": {"sampler_name": "euler"}}
    addnodes = {
        "9910": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 1]}},
        "9911": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {"latent": ["9910", 0], "model_name": P.LBH, "mode": "scale by multiplier",
                 "mode.scale": 2.0, "align": 32, "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
        "9912": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["9911", 0], "audio_latent": ["9910", 1]}},
        "9913": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 100000}},
        "9914": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "9915": {"class_type": "ManualSigmas", "inputs": {"sigmas": P.REFINE_SIGMAS}},
        "9916": {"class_type": "BasicGuider", "inputs": {"model": ["14", 0], "conditioning": ["13", 0]}},
        "9917": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9913", 0], "guider": ["9916", 0], "sampler": ["9914", 0],
                 "sigmas": ["9915", 0], "latent_image": ["9912", 0]}}}
    apiset = {"5": {"samples": ["9917", 0]}, "4": {"samples": ["9917", 0]}}
    return P.submit(url, prompt, f"nw_{name}", a.base_mp, nodeset, addnodes, apiset)


def masked_pass(url, name, canvas, box, refs, denoise, prompt, a):
    im = Image.open(canvas).convert("RGB")
    W, H = im.size
    src = pod.upload(url, canvas, f"nw_{name}_src")
    nodeset = {**INT8, "17": {"image": src}, "12": {"noise_seed": SEED + len(name) * 7919}, "16": {"filename_prefix": f"h3_edit/nw_{name}"},
               "1": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": a.base_mp},
               "7": {"steps": 20, "scheduler": "simple", "denoise": denoise}, "6": {"sampler_name": "euler"}}
    addnodes = {"9300": {"class_type": "VAEEncode", "inputs": {"pixels": ["17", 0], "vae": ["2", 0]}}}
    init = {"samples": ["9300", 0]}
    apiset = {"13": {"width": W, "height": H}, "8": {"latent_image": ["9303", 0]}}
    if box is not None:
        x0, y0, x1, y1 = box
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle((x0 - a.grow, y0 - a.grow, x1 + a.grow, y1 + a.grow), fill=255)
        mp = os.path.join(OUT, f"{name}_mask.png"); m.save(mp)
        addnodes["9302"] = {"class_type": "LoadImageMask", "inputs": {"image": pod.upload(url, mp, f"nw_{name}_mask"), "channel": "red"}}
        init.update(mask=["9302", 0], mask_feather=a.feather)
    addnodes["9303"] = {"class_type": "H3V2VInit", "inputs": init}
    for i, r in enumerate(refs):
        nid = f"91{i:02d}"
        addnodes[nid] = {"class_type": "LoadImage", "inputs": {"image": pod.upload(url, os.path.join(REFS, r), f"nw_ref_{r.split('.')[0]}")}}
        apiset["13"][f"ref_images.ref_image_{i}"] = [nid, 0]
    return P.submit(url, prompt, f"nw_{name}", a.base_mp, nodeset, addnodes, apiset)


def paste_back(prev_canvas, render, box, grow, feather, out):
    """Feathered pixel-space paste of the box: outside stays bit-exact, no VAE round-trip."""
    from PIL import ImageFilter
    prev = Image.open(prev_canvas).convert("RGB")
    ren = Image.open(render).convert("RGB")
    if ren.size != prev.size:
        ren = ren.resize(prev.size, Image.LANCZOS)
    x0, y0, x1, y1 = box
    m = Image.new("L", prev.size, 0)
    ImageDraw.Draw(m).rectangle((x0 - grow, y0 - grow, x1 + grow, y1 + grow), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(feather / 2))
    comp = prev.copy(); comp.paste(ren, (0, 0), m); comp.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="nw")
    ap.add_argument("--base-mp", type=float, default=4.0, help="canvas base; the refine doubles it (4 -> 5440x3072)")
    ap.add_argument("--grow", type=int, default=32)
    ap.add_argument("--feather", type=int, default=64)
    ap.add_argument("--only")
    ap.add_argument("--redo", help="drop this pass and everything after it from the state, then rebuild")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    url = pod.podenv(a.pod)
    st = load_state()
    names = [p[0] for p in PLAN]
    if a.redo:
        keep = names[:names.index(a.redo)]
        st["done"] = [d for d in st["done"] if d in keep]
        st["canvas"] = os.path.join(OUT, f"{st['done'][-1]}.png") if st["done"] else None
        save_state(st)
    for name, kind, frac, refs, denoise, pfile in PLAN:
        if name in st["done"] or (a.only and a.only != name):
            continue
        prompt = os.path.join(PROMPTS, pfile)
        t0 = time.time()
        if kind == "canvas":
            pid = canvas_pass(url, name, refs, prompt, a)
            box = None
        else:
            W, H = Image.open(st["canvas"]).size
            box = px_box(frac, W, H) if frac else None
            pid = masked_pass(url, name, st["canvas"], box, refs, denoise, prompt, a)
        print(f"=== {name} ({kind}) box={box} refs={refs} denoise={denoise} -> {pid}", flush=True)
        out = os.path.join(OUT, f"{name}.png")
        raw = os.path.join(OUT, f"{name}_raw.png") if box else out
        wall = P.wait(url, pid, raw, timeout=3600)
        if box:
            paste_back(st["canvas"], raw, box, a.grow, a.feather, out)
        w, h = Image.open(out).size
        st["done"].append(name); st["canvas"] = out
        st["log"].append({"name": name, "kind": kind, "box": box, "refs": refs, "denoise": denoise, "wall_s": round(wall), "size": [w, h]})
        save_state(st)
        print(f"    {w}x{h} in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
