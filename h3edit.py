#!/usr/bin/env python3
"""h3edit — instruction-based image editing on MiniMax H3, locally.

H3 is a video model. It edits images by rendering ONE FRAME of R2V through a VAE retrained for
single images. Technique: Patient_Ratio4177, r/StableDiffusion 1vo1ab3 (2026-08-14).
Measured dials and failure modes: README.md.

    h3edit "Task: Reference-guided generation. ..." -r car.jpg -r logo.png -o out.png

api_graph.json is NOT hand-built. It is the frontend's own graphToPrompt() output, exported once
from the established local R2V workflow, so the autogrow reference inputs carry their dotted
`ref_images.ref_image_N` keys -- the thing hand-assembled H3 graphs drop silently. This CLI only
sets values in it, and refuses to queue if those keys have gone missing. Re-export with --export
after editing the graph in the GUI.
"""
import argparse, json, math, os, random, shutil, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "api_graph.json")
COMFY = os.environ.get("H3EDIT_COMFY", "http://127.0.0.1:8288")
INPUT_DIR = os.path.expanduser(os.environ.get("H3EDIT_INPUT", "~/ComfyUI-h3/input"))
OUTPUT_DIR = os.path.expanduser(os.environ.get("H3EDIT_OUTPUT", "~/ComfyUI-h3/output"))

# Node ids in api_graph.json. Verified by class_type at load, so a re-export that renumbers
# fails loudly instead of writing a value into the wrong node.
N = {"prompt": ("138", "PrimitiveStringMultiline"), "res": ("115", "ResolutionSelector"),
     "steps": ("124", "BasicScheduler"), "seed": ("129", "RandomNoise"),
     "r2v": ("136", "MiniMaxH3ReferenceToVideo"), "sampler": ("123", "KSamplerSelect"),
     "save": ("664", "SaveImage"), "ksampler": ("125", "SamplerCustomAdvanced"),
     "vae": ("119", "VAELoader")}
REF_NODES = [("137", "ref_images.ref_image_0"), ("139", "ref_images.ref_image_1")]
# Slots 2..4 are added at queue time by cloning the exported LoadImage entry (node 139) and
# linking it under the next dotted key -- the same pattern the pod driver (H3_ADDNODES/H3_APISET) validated on a 5090. Nothing is typed by hand.
MAX_REFS = 5
# Default lane (bench 2026-09-09, truck-cab plate, seed 1001, 5 refs): PlagueKind Parasyte turbo at
# 8 steps / er_sde / beta57 / strength 1.5 held the 20-step base model's detail at 4 MP in 13 min
# on the M5 vs 28 min. The lightx2v 8-step LoRA it replaces is deprecated. beta57 is registered
# by the comfyui-obvpm pack; --doctor checks for it.
TURBO_LORA = "H3-PK-Parasyte-Turbo.safetensors"

ASPECTS = {"1:1": "1:1 (Square)", "2:3": "2:3 (Portrait Photo)", "3:2": "3:2 (Photo)",
           "3:4": "3:4 (Portrait Standard)", "4:3": "4:3 (Standard)",
           "9:16": "9:16 (Portrait Widescreen)", "16:9": "16:9 (Widescreen)",
           "21:9": "21:9 (Ultrawide)"}


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def combo_options(node, widget):
    """Options of a COMBO input: v[1]['options'] on the current schema, v[0] on the old one."""
    v = api(f"/api/object_info/{node}")[node]["input"]["required"][widget]
    return v[1]["options"] if isinstance(v[0], str) else v[0]


def doctor():
    ok = True
    try:
        api("/system_stats")
        print("ok   ComfyUI reachable at", COMFY)
    except Exception as e:
        return print(f"FAIL ComfyUI not reachable at {COMFY}: {e}") or False
    # The single-frame rebind lives in the h3_single_frame custom node and only takes effect on
    # restart, so the running server is the only thing worth asking. Ask for the marker node
    # rather than the length schema: /object_info is serialized at registration, BEFORE custom
    # nodes load, so it still advertises min=5 even when the rebind is live. h3edit drives
    # length through a linked ComfyMathExpression, where the range is not validated anyway.
    # /object_info answers {} with HTTP 200 for an unknown node, so require the key itself.
    if "H3SingleFrameEnabled" in api("/api/object_info/H3SingleFrameEnabled"):
        print("ok   single-frame rebind is live (h3_single_frame custom node loaded)")
    else:
        ok = False
        print("FAIL h3_single_frame custom node is not loaded. Symlink it into ComfyUI's "
              "custom_nodes/ and RESTART ComfyUI.")
    g = json.load(open(GRAPH))
    for key, (nid, cls) in N.items():
        if nid not in g or g[nid]["class_type"] != cls:
            ok = False
            print(f"FAIL api_graph.json node {nid} is not {cls} -- re-export with --export")
    # length must arrive via a LINK from the PrimitiveInt: literal widget values are
    # range-validated at /prompt (min=5), linked inputs are only type-checked.
    if g.get("131", {}).get("class_type") != "PrimitiveInt" or g["131"]["inputs"].get("value") != 1 \
            or g[N["r2v"][0]]["inputs"].get("length") != ["131", 0]:
        ok = False
        print("FAIL length is not linked from PrimitiveInt(1) -- the graph would render a clip")
    try:
        import PIL
        print("ok   pillow", PIL.__version__, "(--detail needs it)")
    except ImportError:
        ok = False
        print("FAIL pillow missing in this install -- run: uv tool install --force -e .  (stale tool venv)")
    if "H3V2VInit" in api("/api/object_info/H3V2VInit"):
        print("ok   H3V2VInit registered (ComfyUI-MAINodes; --inpaint)")
    else:
        print("warn --inpaint unavailable: install ComfyUI-MAINodes (H3V2VInit) and restart")
    if "beta57" in combo_options("BasicScheduler", "scheduler"):
        print("ok   beta57 scheduler registered (comfyui-obvpm)")
    else:
        ok = False
        print("FAIL beta57 scheduler missing -- install comfyui-obvpm, or run with --scheduler simple")
    if TURBO_LORA in combo_options("LoraLoaderModelOnly", "lora_name"):
        print("ok   turbo LoRA present:", TURBO_LORA)
    else:
        ok = False
        print(f"FAIL {TURBO_LORA} not in models/loras -- download it (Plaguekind/H3-Lora) or run --lora off")
    refs = g[N["r2v"][0]]["inputs"]
    missing = [k for _, k in REF_NODES if k not in refs]
    if missing:
        ok = False
        print(f"FAIL reference inputs {missing} are absent -- renders would silently ignore "
              "your images. Re-export with --export.")
    else:
        print("ok   graph:", len(g), "nodes,", ", ".join(k for _, k in REF_NODES))
    return ok


def build(args):
    g = json.load(open(GRAPH))
    for key, (nid, cls) in N.items():
        if nid not in g or g[nid]["class_type"] != cls:
            sys.exit(f"api_graph.json node {nid} is not {cls}. Re-export with --export.")

    r2v = g[N["r2v"][0]]["inputs"]
    missing = [k for _, k in REF_NODES if k not in r2v]
    if missing:
        sys.exit(f"reference inputs {missing} are absent from the graph. This is the silent "
                 "failure mode -- the render would ignore your images. Re-export with --export.")

    g[N["prompt"][0]]["inputs"]["value"] = args.prompt
    g[N["res"][0]]["inputs"].update(aspect_ratio=ASPECTS[args.ar], megapixels=args.mp)
    g[N["steps"][0]]["inputs"].update(steps=args.steps, scheduler=args.scheduler)
    g[N["sampler"][0]]["inputs"]["sampler_name"] = args.sampler
    g[N["seed"][0]]["inputs"]["noise_seed"] = args.seed
    g[N["r2v"][0]]["inputs"]["ref_image_size"] = args.ref_size
    g[N["save"][0]]["inputs"]["filename_prefix"] = "h3_edit/" + args.name
    if args.dit:
        g["665"]["inputs"]["unet_name"] = args.dit
    if args.vae:
        g[N["vae"][0]]["inputs"]["vae_name"] = args.vae
    if args.frames != 1:                        # grid diagnostics: render N frames, keep the middle one
        g["131"]["inputs"]["value"] = args.frames
        g["663"]["inputs"]["batch_index"] = args.frames // 2
    if args.te:
        # Swap the ClipProj encoder for the full GGUF text encoder. The entry mirrors MacMax's
        # CLIPLoaderGGUF node 13 (widgets: clip_name, type=minimax); keeping id 661 keeps the
        # R2V node's clip link intact.
        g["661"] = {"class_type": "CLIPLoaderGGUF",
                    "inputs": {"clip_name": args.te, "type": "minimax"},
                    "_meta": {"title": "CLIPLoader (GGUF)"}}
    if args.lora == "off":
        # Bypass the LoRA loader the way the GUI's mode-4 does: both model consumers take the
        # DiT loader's output directly, and the loader entry leaves the graph.
        for nid in ("124", "126"):
            g[nid]["inputs"]["model"] = ["665", 0]
        del g["666"]
    else:
        g["666"]["inputs"].update(lora_name=args.lora, strength_model=args.lora_strength)
    names = [stage_ref(p) for p in args.refs]
    for i, name in enumerate(names):
        if i < len(REF_NODES):
            nid, key = REF_NODES[i]
        else:
            nid, key = f"91{i:02d}", f"ref_images.ref_image_{i}"
            g[nid] = json.loads(json.dumps(g[REF_NODES[1][0]]))
            r2v[key] = [nid, 0]
        g[nid]["inputs"]["image"] = name
    if len(names) == 1:
        # One ref: point both slots at it rather than unwiring a slot, which would need the
        # frontend to re-serialize the autogrow input.
        g[REF_NODES[1][0]]["inputs"]["image"] = names[0]
    if args.inpaint:
        inpaint_wire(g, args)
    return g


def inpaint_wire(g, args):
    """Masked partial denoise (benchmark 2026-09-10, 5090): the source enters ONLY as an encoded
    latent through H3V2VInit (ComfyUI-MAINodes), whose noise mask freezes every latent cell
    outside the box; the sampler starts at --denoise of the schedule. The -r images are the
    references (<Picture 1>..) -- the source is deliberately NOT a reference, because with the
    source as <Picture 1> the model reproduces its mistakes verbatim (a wrong digit survived
    denoise 0.6, 0.85 and 1.0). Artwork-only at 0.85 corrected the digit, kept the plate's size
    and position, and left the rest of the frame pixel-frozen, in 27 s.
    """
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = args.inpaint
    src = Image.open(args.source).convert("RGB")
    W, H = src.size
    if W % 32 or H % 32:
        sys.exit(f"--inpaint source must be a multiple of 32 px per side (an h3edit render is); got {W}x{H}")
    if x1 - x0 < 32 or y1 - y0 < 32 or x1 > W or y1 > H:
        sys.exit(f"--inpaint box {args.inpaint} does not fit {args.source} ({W}x{H})")
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle((x0 - args.grow, y0 - args.grow, x1 + args.grow, y1 + args.grow), fill=255)
    mask_path = os.path.join(INPUT_DIR, f"{args.name}_mask.png")
    mask.save(mask_path)
    src_name = stage_ref(args.source)
    g["9300"] = {"class_type": "LoadImage", "inputs": {"image": src_name}}
    enc_vae = [N["vae"][0], 0]
    if args.encode_vae:                          # grid diagnostics: a different VAE for the encode side only
        g["9304"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.encode_vae}}
        enc_vae = ["9304", 0]
    pixels = ["9300", 0]
    if args.frames != 1:                         # grid diagnostics: encode the source as an N-frame still clip
        g["9305"] = {"class_type": "RepeatImageBatch", "inputs": {"image": ["9300", 0], "amount": args.frames}}
        pixels = ["9305", 0]
    g["9301"] = {"class_type": "VAEEncode", "inputs": {"pixels": pixels, "vae": enc_vae}}
    if args.encode_tiled:                        # grid diagnostics: tiled encode (less memory)
        g["9301"] = {"class_type": "VAEEncodeTiled", "inputs": {"pixels": pixels, "vae": enc_vae, "tile_size": 512, "overlap": 64,
                                                                "temporal_size": 64, "temporal_overlap": 8}}
    g["9302"] = {"class_type": "LoadImageMask", "inputs": {"image": os.path.basename(mask_path), "channel": "red"}}
    g["9303"] = {"class_type": "H3V2VInit", "inputs": {"samples": ["9301", 0], "mask": ["9302", 0], "mask_feather": args.feather}}
    g[N["ksampler"][0]]["inputs"]["latent_image"] = ["9303", 0]
    if args.save_latent:                          # grid diagnostics: dump the encoded and the sampled latent
        g["9306"] = {"class_type": "SaveLatent", "inputs": {"samples": ["9301", 0], "filename_prefix": f"latents/{args.name}_enc"}}
        g["9307"] = {"class_type": "SaveLatent", "inputs": {"samples": [N["ksampler"][0], 0], "filename_prefix": f"latents/{args.name}_out"}}
    g[N["r2v"][0]]["inputs"].update(width=W, height=H)      # the latent fixes the size
    g[N["steps"][0]]["inputs"]["denoise"] = args.denoise
    if args.decode_crop:                          # decode only the box + margin (latent cells of 16 px)
        m = args.grow + args.feather + 32
        cx0, cy0 = max(0, x0 - m) // 16, max(0, y0 - m) // 16
        cx1, cy1 = min(W, x1 + m + 15) // 16, min(H, y1 + m + 15) // 16
        args.decode_crop = (cx0 * 16, cy0 * 16)
        g["9308"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [N["ksampler"][0], 0]}}
        g["9309"] = {"class_type": "LatentCut", "inputs": {"samples": ["9308", 0], "dim": "y", "index": cy0, "amount": cy1 - cy0}}
        g["9310"] = {"class_type": "LatentCut", "inputs": {"samples": ["9309", 0], "dim": "x", "index": cx0, "amount": cx1 - cx0}}
        g["122"]["inputs"]["samples"] = ["9310", 0]


def stage_ref(path):
    """Copy a reference into ComfyUI's input dir and return its basename."""
    name = os.path.basename(path)
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.copy(path, dst)
    return name


def run(args):
    g = build(args)
    r = api("/prompt", {"prompt": g, "client_id": "h3edit"})
    pid = r.get("prompt_id")
    if not pid:
        sys.exit("queue rejected: " + json.dumps(r)[:600])
    print(f"queued {pid}  {args.steps} steps / {args.mp} MP / refs {args.ref_size}")
    if not args.wait:
        return
    t0 = time.time()
    while True:
        h = api(f"/history/{pid}")
        if h:
            v = next(iter(h.values()))
            if v["status"]["status_str"] != "success":
                sys.exit("render failed: " + json.dumps(v["status"])[:600])
            im = next(i for o in v["outputs"].values() for i in o.get("images", []))
            src = os.path.join(OUTPUT_DIR, im.get("subfolder", ""), im["filename"])
            if args.out:
                shutil.copy(src, args.out)
                src = args.out
            print(f"{src}  ({int(time.time() - t0)}s)")
            return src
        time.sleep(10)


def notch_grid(im, periods=(16, 8), width=1):
    """Remove the exact latent-cell harmonics (16 px and 8 px, rows and columns) from an image. The H3
    decoder lays a faint cell grid over any frame sampled with encoded (V2V) context; a plain R2V
    frame is clean (16-px harmonic ~1-5x background vs 50-200x on inpaint output, 2026-09-10)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im, np.float32); H, W = a.shape[:2]
    for c in range(a.shape[2]):
        F = np.fft.fft2(a[..., c])
        for p in periods:
            ky, kx = int(round(H / p)), int(round(W / p))
            for d in range(-width, width + 1):
                F[(ky + d) % H, :] = 0; F[(-ky - d) % H, :] = 0
                F[:, (kx + d) % W] = 0; F[:, (-kx - d) % W] = 0
        a[..., c] = np.real(np.fft.ifft2(F))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def deblock_cells(im, cell=16, T=10.0, r=3):
    """Smooth across every latent-cell boundary where the step is small (< T levels) and both sides
    are flat: the decoder's per-cell tone steps in skies, walls and paint (a 1-3 level mosaic) go,
    real edges stay. r px each side become a linear ramp. Use cell=32 on a 2x-upscaled canvas."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im, np.float32).copy()
    for axis in (0, 1):
        v = a if axis == 0 else a.transpose(1, 0, 2)
        for b in range(cell, v.shape[0] - r, cell):
            lo, hi = v[b - r - 1], v[b + r]
            flat = (np.abs(v[b - 1] - lo) < T) & (np.abs(hi - v[b]) < T) & (np.abs(v[b] - v[b - 1]) < T)
            for k in range(2 * r):
                v[b - r + k] = np.where(flat, lo + (hi - lo) * (k + 1) / (2 * r + 1), v[b - r + k])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def tone_match(ren, src, box, grow, band=64):
    """Correct the whole-box tone-shift of a V2V render. The masked-latent decode shifts the box's
    colour a few levels (a visible rectangle of "different" sand/wall against a clean base); notch and
    deblock remove the periodic grid but not this DC offset. Match ren to src on a ring just around
    the grown box (same frozen-context content on both, so the median ratio isolates the tone shift),
    then apply that per-channel gain to ren before the paste. (Phuket 2026-09-12.)"""
    import numpy as np
    from PIL import Image
    r = np.asarray(ren, np.float32); s = np.asarray(src, np.float32)
    H, W = r.shape[:2]; x0, y0, x1, y1 = box
    ox0, oy0 = max(0, x0 - grow - band), max(0, y0 - grow - band)
    ox1, oy1 = min(W, x1 + grow + band), min(H, y1 + grow + band)
    ix0, iy0 = max(0, x0 - grow), max(0, y0 - grow)
    ix1, iy1 = min(W, x1 + grow), min(H, y1 + grow)
    ring = np.zeros((H, W), bool); ring[oy0:oy1, ox0:ox1] = True; ring[iy0:iy1, ix0:ix1] = False
    if ring.sum() < 100:
        return ren
    rr, ss = r[ring], s[ring]
    bright = ss.sum(1) > np.percentile(ss.sum(1), 45)   # stable bright surfaces (sand/wall), not saturated props
    rr, ss = rr[bright], ss[bright]
    gain = np.array([np.clip(np.median(ss[:, c]) / max(1.0, np.median(rr[:, c])), 0.8, 1.5) for c in range(3)])
    return Image.fromarray(np.clip(r * gain, 0, 255).astype(np.uint8))


def inpaint_run(args):
    """--inpaint: render, then paste only the (grown, feathered) box back onto --source. The model
    saw the frozen latent for context; the pixels outside the box never take a VAE round-trip
    (chaining masked passes without this drifted a 16 MP canvas dark by the 7th pass)."""
    from PIL import Image, ImageDraw, ImageFilter
    out = args.out
    args.out = None
    args.wait = True
    ren = Image.open(run(args)).convert("RGB")
    src = Image.open(args.source).convert("RGB")
    if args.decode_crop:                          # the render is the decoded crop: put it in place
        full = src.copy(); full.paste(ren, args.decode_crop); ren = full
    if ren.size != src.size:
        ren = ren.resize(src.size, Image.LANCZOS)
    if not args.no_notch:
        ren = deblock_cells(notch_grid(ren))
    x0, y0, x1, y1 = args.inpaint
    if not args.no_tonematch:
        ren = tone_match(ren, src, (x0, y0, x1, y1), args.grow)
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).rectangle((x0 - args.grow, y0 - args.grow, x1 + args.grow, y1 + args.grow), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(args.feather / 2))
    comp = src.copy(); comp.paste(ren, (0, 0), m)
    if out:
        comp.save(out)
        print(f"{out}  (inpaint box {x0},{y0},{x1},{y1}, denoise {args.denoise}, pasted back)")


def detail(args):
    """Two-pass detail: re-render a crop of --source at full resolution, paste it back feathered.

    H3's VAE is 16 px per latent cell, so lettering ~50 px tall in a full frame cannot resolve at
    any reference quality. Rendering only the crop gives the same letters 3-4x the cells; the crop
    goes in as <Picture 1> (geometry authority), your -r images follow as <Picture 2>.., and the
    result is scaled back to the box and blended in with a feathered edge (measured 2026-09-09:
    door plate, fleet number, phone and URL all legible, seam invisible, +6.5 min on the M5).
    """
    from PIL import Image, ImageFilter
    x0, y0, x1, y1 = args.detail
    full = Image.open(args.source).convert("RGB")
    w, h = x1 - x0, y1 - y0
    if w < 64 or h < 64 or x1 > full.width or y1 > full.height:
        sys.exit(f"--detail box {args.detail} does not fit {args.source} ({full.width}x{full.height})")
    crop_path = os.path.join(INPUT_DIR, f"{args.name}_crop.png")
    full.crop((x0, y0, x1, y1)).save(crop_path)
    args.refs = [crop_path] + args.refs
    args.ar = min(ASPECTS, key=lambda k: abs(int(k.split(":")[0]) / int(k.split(":")[1]) - w / h))
    out = args.out
    args.out = None
    args.wait = True
    ren = Image.open(run(args)).convert("RGB").resize((w, h), Image.LANCZOS)
    f = args.feather
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (f, f, w - f, h - f))
    mask = mask.filter(ImageFilter.GaussianBlur(f / 2))
    comp = full.copy()
    comp.paste(ren, (x0, y0), mask)
    comp.save(out)
    print(f"{out}  (detail box {x0},{y0},{x1},{y1} at {args.ar}, feather {f}px)")


_UP_PROMPT = ("Task: Reference-guided generation. <Picture 1> is a crop of a finished photo. Reproduce "
              "it exactly -- same content, framing, lighting and colours -- rendered sharper with fine "
              "natural detail. Continue the style. Paint only inside the frame. No text.")


def _up_boxes(W, H, tile_mp, overlap):
    """Tile the image into ~tile_mp crops, origins+sizes snapped to the 16px VAE grid, with overlap."""
    tpx = max(256, int(((tile_mp * 1e6) ** 0.5) // 32) * 32)
    def axis(n):
        t = min(tpx, (n // 32) * 32) or 32
        step = max(32, int(t * (1 - overlap)) // 32 * 32)
        starts = list(range(0, max(1, n - t) + 1, step)) or [0]
        if n > t and starts[-1] != n - t:
            starts.append(((n - t) // 32) * 32)
        return sorted(set(starts)), t
    xs, tw = axis(W); ys, th = axis(H)
    return [(x, y, min(x + tw, W), min(y + th, H)) for y in ys for x in xs]


def _edge_score(crop):
    """Laplacian variance of a crop (grey) -- high = real edges/texture, low = flat."""
    import numpy as np
    from scipy.ndimage import laplace
    g = np.asarray(crop.convert("L"), np.float32)
    return float(laplace(g).var())


def _wavelet_recombine(base_crop, rendered, sigma):
    """Low frequencies from the scaffold (owns lighting), high frequencies from the H3 render
    (adds texture) -> tile-to-tile lighting drift is impossible by construction."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from PIL import Image
    b = np.asarray(base_crop, np.float32); r = np.asarray(rendered, np.float32)
    low = np.stack([gaussian_filter(b[..., c], sigma) for c in range(3)], -1)
    hi = r - np.stack([gaussian_filter(r[..., c], sigma) for c in range(3)], -1)
    return Image.fromarray(np.clip(low + hi, 0, 255).astype(np.uint8))


def upscale(args):
    """Enlarge (Lanczos scaffold) then add real H3 detail: tile into ~tile_mp crops, skip flat crops
    (saliency gate -> no hallucination + fewer renders), render each at 4 MP, recombine wavelet-style
    (base low-pass + H3 high-pass so lighting can't drift), feather-paste. Verified panel design 2026-09-12."""
    from PIL import Image, ImageFilter, ImageDraw
    base = args.name
    src = Image.open(args.upscale).convert("RGB")
    W0, H0 = src.size
    W = max(32, int(round(W0 * args.scale)) // 32 * 32)
    H = max(32, int(round(H0 * args.scale)) // 32 * 32)
    big = src.resize((W, H), Image.LANCZOS)            # scaffold
    boxes = _up_boxes(W, H, args.tile_mp, args.overlap)
    scored = [(b, _edge_score(big.crop(b))) for b in boxes]
    keep = [(b, sc) for b, sc in scored if sc >= args.edge_thresh]
    print(f"upscale: {W0}x{H0} -> {W}x{H}  ({len(boxes)} tiles, {len(keep)} above edge-thresh {args.edge_thresh})")
    if args.dry_run:
        over = big.copy(); d = ImageDraw.Draw(over)
        for b, sc in scored:
            col = (0, 220, 0) if sc >= args.edge_thresh else (200, 60, 60)
            d.rectangle(b, outline=col, width=4)
            d.text((b[0] + 6, b[1] + 6), f"{sc:.0f}", fill=col)
        op = args.out or os.path.join(OUTPUT_DIR, f"{base}_upscale_plan.png")
        over.save(op)
        return print(f"dry-run plan: {op}  (green = will detail, red = skipped-flat; no renders). "
                     f"tune --edge-thresh from the printed scores")
    out = args.out
    big.save(out)                                       # working canvas = scaffold; kept tiles overwrite
    sigma = max(1.0, 0.015 * W)
    f = args.feather
    for i, (b, sc) in enumerate(keep):
        x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0
        print(f"[{i + 1}/{len(keep)}] tile {b} score {sc:.0f}")
        cur = Image.open(out).convert("RGB")
        crop_path = os.path.join(INPUT_DIR, f"{base}_up{i}_crop.png")
        cur.crop(b).save(crop_path)
        args.refs = [crop_path]
        args.ar = min(ASPECTS, key=lambda k: abs(int(k.split(":")[0]) / int(k.split(":")[1]) - w / h))
        args.mp = 4.0
        args.prompt = _UP_PROMPT
        args.name = f"{base}_up{i}"
        args.out = None; args.wait = True
        ren = Image.open(run(args)).convert("RGB").resize((w, h), Image.LANCZOS)
        args.out = out                                  # run() nulled it
        recomb = _wavelet_recombine(cur.crop(b), ren, sigma)
        mask = Image.new("L", (w, h), 0)
        mask.paste(255, (f, f, max(f + 1, w - f), max(f + 1, h - f)))
        mask = mask.filter(ImageFilter.GaussianBlur(f / 2))
        comp = cur.copy(); comp.paste(recomb, (x0, y0), mask); comp.save(out)
    print(f"upscale -> {out} ({W}x{H}, {len(keep)} tiles detailed)")



def dispatch(args):
    """Run one variation in whatever mode the args select; the result lands at args.out."""
    if args.detail:
        return detail(args)
    if args.inpaint:
        return inpaint_run(args)
    return run(args)


def make_sheet(items, out_path):
    """Contact sheet of the variations, each cell captioned with its seed. items = [(path, label)]."""
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p, _ in items]
    n = len(ims)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    tw = 480; cap = 26; pad = 8
    th = max(1, round(tw * ims[0].height / ims[0].width))
    cw, ch = tw + pad, th + cap + pad
    sheet = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, ((_, label), im) in enumerate(zip(items, ims)):
        r, c = divmod(i, cols)
        x, y = pad + c * cw, pad + r * ch
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 4, y + th + 6), label, fill=(235, 235, 235))
    sheet.save(out_path)
    return out_path


def batch(args, seeds):
    """Render one variation per seed, save all, and write a labeled contact sheet (seed-select)."""
    base_out, base_name = args.out, args.name
    if base_out:
        stem, ext = os.path.splitext(base_out)
    else:
        stem, ext = os.path.join(OUTPUT_DIR, base_name), ".png"
    print(f"seed-select: {len(seeds)} variations (sequential; ~{len(seeds)}x a single render)")
    items = []
    for i, s in enumerate(seeds):
        args.seed = s
        args.name = f"{base_name}_s{s}"          # keep per-seed intermediates from colliding
        args.out = f"{stem}_s{s}{ext}"
        args.wait = True
        print(f"[{i + 1}/{len(seeds)}] seed {s} -> {args.out}")
        dispatch(args)
        items.append((args.out, f"seed {s}"))
    sheet = make_sheet(items, f"{stem}_sheet{ext}")
    print(f"sheet: {sheet}\nvariations: " + ", ".join(p for p, _ in items))


def main():
    p = argparse.ArgumentParser(description="Instruction-based image editing on MiniMax H3, local.")
    p.add_argument("prompt", nargs="?", help="edit instruction; see prompts/reference_prompts.txt")
    p.add_argument("-r", "--ref", dest="refs", action="append", default=[],
                   help=f"reference image (repeatable, max {MAX_REFS})")
    p.add_argument("-o", "--out", help="copy the result here")
    p.add_argument("--ar", default="21:9", choices=sorted(ASPECTS), help="aspect ratio")
    # With ref_size=match the references are scaled DOWN to the generation's pixel area (never up),
    # so megapixels caps the reference resolution too. At 1.0 a decal came back airbrushed; 2.0 was
    # clean for large marks but ~50-px lettering stayed mush; 4.0 resolves it (bench 2026-09-09).
    p.add_argument("--mp", type=float, default=None, help="megapixels (also caps the refs); default 4.0, "
                   "2.0 with --detail. 4.0 makes ~50-px lettering legible that 2.0 renders as mush (16 px/latent cell)")
    p.add_argument("--detail", metavar="X0,Y0,X1,Y1", type=lambda v: [int(x) for x in v.split(",")],
                   help="two-pass detail: re-render this box of --source and paste it back (see README)")
    p.add_argument("--source", help="full image the --detail box is cut from; becomes <Picture 1>")
    p.add_argument("--feather", type=int, default=48, help="edge feather in px (--detail paste-back, --inpaint mask)")
    p.add_argument("--inpaint", metavar="X0,Y0,X1,Y1", type=lambda v: [int(x) for x in v.split(",")],
                   help="masked re-denoise of this box of --source; -r images are the references, the rest of the frame is frozen")
    p.add_argument("--denoise", type=float, default=0.85, help="--inpaint: fraction of the schedule to run (0.85 corrects lettering and keeps geometry; 1.0 re-composes the box)")
    p.add_argument("--grow", type=int, default=32, help="--inpaint: dilate the box by this many px")
    p.add_argument("--steps", type=int, default=8, help="8 with the turbo LoRA; 20 with --lora off")
    p.add_argument("--sampler", default="er_sde", help="KSamplerSelect sampler_name (euler with --lora off)")
    p.add_argument("--scheduler", default="beta57", help="BasicScheduler scheduler (simple with --lora off)")
    p.add_argument("--lora", default=TURBO_LORA, help="LoRA file in models/loras, or 'off' (base model, 20 steps)")
    p.add_argument("--lora-strength", type=float, default=1.5)
    p.add_argument("--dit", default=None, help="override the GGUF DiT (unet_name)")
    p.add_argument("--vae", default=None, help="override the VAE file (node 119); default = the graph's video VAE")
    p.add_argument("--frames", type=int, default=1, help="diagnostic: render N frames and keep the middle one (default 1)")
    p.add_argument("--encode-vae", default=None, help="diagnostic: VAE file for the --inpaint encode side only")
    p.add_argument("--encode-tiled", action="store_true", help="diagnostic: VAEEncodeTiled for the --inpaint encode")
    p.add_argument("--save-latent", action="store_true", help="diagnostic: SaveLatent of the encoded and sampled latents (output/latents/)")
    p.add_argument("--no-notch", action="store_true", help="--inpaint: skip the cell-grid filter (default: the 16/8 px harmonics are notched and the cell boundaries deblocked in the render before the paste)")
    p.add_argument("--no-tonematch", action="store_true", help="--inpaint: skip matching the box's tone to the surrounding source (default: a per-channel gain from a ring around the box removes the V2V decode's whole-box colour shift)")
    p.add_argument("--decode-crop", action="store_true", help="--inpaint: decode only the box (+grow+feather+32 px) instead of the whole frame")
    p.add_argument("--te", default=None, help="GGUF text encoder file instead of ClipProj")
    p.add_argument("--ref-size", default="match", choices=["match", "max"],
                   help="'max' pins refs to a 2048px short edge: ~10x slower, no better")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n", type=int, default=1, help="seed-select: render N variations (seed, seed+1, ...) and a labeled contact sheet; pick one by eye")
    p.add_argument("--seeds", type=lambda v: [int(s) for s in v.split(",")], default=None,
                   help="seed-select: exact seeds, comma-separated (overrides --n)")
    p.add_argument("--name", default="h3_edit", help="output filename prefix")
    p.add_argument("--upscale", metavar="IMAGE",
                   help="enlarge + add real H3 detail: Lanczos scaffold, then saliency-gated 4 MP detail "
                        "tiles recombined wavelet-style. Plain faithful upscale = the MLX-DLSS workflow instead")
    p.add_argument("--scale", type=float, default=2.0, help="--upscale: enlargement factor (1 = re-detail in place)")
    p.add_argument("--tile-mp", dest="tile_mp", type=float, default=1.2, help="--upscale: crop size in MP (rendered at 4 MP)")
    p.add_argument("--overlap", type=float, default=0.2, help="--upscale: tile overlap fraction")
    p.add_argument("--edge-thresh", dest="edge_thresh", type=float, default=6.0,
                   help="--upscale: skip tiles whose Laplacian variance is below this (flat = no detail needed)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="--upscale: save a tile plan (green=detail, red=skip) and render nothing")
    p.add_argument("--wait", action="store_true", help="block until the render lands")
    p.add_argument("--doctor", action="store_true", help="check the local install and exit")
    p.add_argument("--export", metavar="TAB",
                   help="re-export api_graph.json from a ComfyUI tab id on CDP $H3EDIT_CDP")
    args = p.parse_args()

    if args.doctor:
        sys.exit(0 if doctor() else 1)
    if args.export:
        from export_graph import export
        return export(args.export, GRAPH)
    if args.upscale:
        if not (args.upscale and args.out):
            p.error("--upscale needs an IMAGE and -o")
        if not os.path.exists(args.upscale):
            p.error(f"--upscale: image not found: {args.upscale}")
        if args.seed is None:
            args.seed = random.randrange(1, 2**31)
        return upscale(args)
    if args.detail:
        if not (args.prompt and args.source and args.out and len(args.detail) == 4):
            p.error("--detail needs a prompt, --source, -o and a X0,Y0,X1,Y1 box")
        if len(args.refs) > MAX_REFS - 1:
            p.error(f"--detail: max {MAX_REFS - 1} extra references (the crop is <Picture 1>)")
    elif args.inpaint:
        if not (args.prompt and args.source and args.refs and len(args.inpaint) == 4):
            p.error("--inpaint needs a prompt, --source, at least one -r (the artwork) and a X0,Y0,X1,Y1 box")
        if args.detail:
            p.error("--inpaint and --detail are different passes; run one at a time")
    elif not args.prompt or not args.refs:
        p.error("a prompt and at least one --ref are required")
    if len(args.refs) > MAX_REFS:
        p.error(f"max {MAX_REFS} references")
    if args.mp is None:
        args.mp = 2.0 if args.detail else 4.0
    if args.seed is None:
        args.seed = random.randrange(1, 2**31)
    seeds = args.seeds if args.seeds else [args.seed + i for i in range(args.n)]
    if len(seeds) > 1:
        return batch(args, seeds)
    return dispatch(args)


if __name__ == "__main__":
    main()
