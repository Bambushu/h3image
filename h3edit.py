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
import argparse, copy, hashlib, json, math, os, random, shutil, sys, tempfile, time, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
# The graphs ship as package data in h3edit_graphs/, a sibling of this module in both a checkout and
# an installed wheel.
GRAPH_DIR = os.path.join(HERE, "h3edit_graphs")
GRAPH = os.path.join(GRAPH_DIR, "api_graph.json")
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


# --- Graph profiles ------------------------------------------------------------------
# Two SHIPPED graphs, each exported once via the frontend's graphToPrompt (NEVER hand-built):
#   mac  = api_graph.json      (MacMax H3 R2V, MPS; ClipProj TE + Parasyte turbo lane)
#   cuda = api_graph.cuda.json (shipped int8_convrot single-image edit graph; base model, euler/simple)
# int8_convrot has no MPS path and MacMax has no CUDA path, so the split is inherent. Node maps are
# the graphToPrompt ids of each export (cuda ported from the 5090-validated h3edit_pod.py nodeset).
# Pick with --profile / $H3EDIT_PROFILE (default mac). Local CUDA box = HTTP + shared filesystem I/O
# (identical to mac); a REMOTE pod (COMFY host not localhost) switches to /upload/image + /view.
CUDA_GRAPH = os.path.join(GRAPH_DIR, "api_graph.cuda.json")
_PROFILES = {
    "mac": {"graph": GRAPH, "comfy": COMFY, "prompt_key": "value",
            "nodes": dict(prompt="138", res="115", steps="124", seed="129", r2v="136",
                          sampler="123", save="664", ksampler="125", vae="119",
                          length="131", batch="663", unet="665", lora="666", te="661"),
            "ref_nodes": [("137", "ref_images.ref_image_0"), ("139", "ref_images.ref_image_1")],
            "models": {},
            "lane": dict(sampler="er_sde", scheduler="beta57", steps=8)},
    "cuda": {"graph": CUDA_GRAPH, "comfy": os.environ.get("H3EDIT_COMFY", ""), "prompt_key": "prompt",
             "nodes": dict(prompt="13", res="1", steps="7", seed="12", r2v="13",
                           sampler="6", save="16", ksampler="8", vae="2",
                           length="20", batch="15", unet="10", lora=None, te=None),
             "ref_nodes": [("17", "ref_images.ref_image_0")],
             "models": {"10": ("unet_name", "minimax_h3_fl2va_int8_convrot.safetensors"),
                        "11": ("clip_name", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                        "2": ("vae_name", "minimax_h3_video_vae_fp16.safetensors")},
             "lane": dict(sampler="euler", scheduler="simple", steps=20)},
}
PROFILE = "mac"
PROF = _PROFILES["mac"]


def apply_profile(name):
    global PROFILE, PROF, GRAPH, COMFY
    if name not in _PROFILES:
        sys.exit(f"unknown --profile {name!r} (choose: mac, cuda)")
    PROFILE, PROF = name, _PROFILES[name]
    GRAPH = PROF["graph"]
    if PROF["comfy"]:
        COMFY = PROF["comfy"]


def _remote():
    """True = move files over /upload/image + /view; False = share ComfyUI's input/output dirs.
    Defaults to the host heuristic; set H3EDIT_TRANSPORT=upload for an SSH tunnel to localhost whose
    server keeps its files elsewhere (or =shared for a remote host on a shared mount)."""
    t = os.environ.get("H3EDIT_TRANSPORT", "").lower()
    if t in ("upload", "shared"):
        return t == "upload"
    from urllib.parse import urlparse
    return (urlparse(COMFY).hostname or "") not in ("127.0.0.1", "localhost", "::1", "")


_WORK = None


def workdir():
    """Per-process scratch dir for generated inputs (masks, crops, cards). They reach ComfyUI through
    stage_ref, so nothing here depends on a local ComfyUI input dir existing."""
    global _WORK
    if _WORK is None:
        import atexit
        _WORK = tempfile.mkdtemp(prefix="h3edit_")
        atexit.register(shutil.rmtree, _WORK, True)
    return _WORK


def _upload(path, name=None):
    """POST an image to a remote ComfyUI's /upload/image and return the stored name."""
    import mimetypes, uuid
    name = name or os.path.basename(path)
    bnd = uuid.uuid4().hex
    ct = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ct}\r\n\r\n").encode() + open(path, "rb").read() + \
        (f"\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n"
         f"--{bnd}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{bnd}--\r\n").encode()
    req = urllib.request.Request(COMFY + "/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={bnd}", "User-Agent": "curl/8"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read() or b"{}").get("name", name)


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY + path, data=data,
                                 headers={"Content-Type": "application/json", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def combo_options(node, widget):
    """Options of a COMBO input: v[1]['options'] on the current schema, v[0] on the old one."""
    info = api(f"/api/object_info/{node}")
    if node not in info:            # unknown node -> ComfyUI returns {} (HTTP 200); let callers FAIL cleanly
        return []
    v = info[node]["input"]["required"][widget]
    return v[1]["options"] if isinstance(v[0], str) else v[0]


def doctor_cuda():
    ok = True
    try:
        api("/system_stats")
        print("ok   ComfyUI reachable at", COMFY, "(remote pod)" if _remote() else "(local)")
    except Exception as e:
        print("FAIL ComfyUI not reachable at", COMFY, "--", str(e)[:120]); return False
    if not os.path.exists(GRAPH):
        print("FAIL", GRAPH, "missing -- export it once from a ComfyUI with the H3 edit graph loaded"); return False
    g = json.load(open(GRAPH))
    nd = PROF["nodes"]
    missing = [f"{k}={nid}" for k, nid in nd.items() if nid and nid not in g]
    if missing:
        print("FAIL", os.path.basename(GRAPH), "lacks nodes", ", ".join(missing), "-- re-export it"); return False
    print("ok   graph:", os.path.basename(GRAPH), f"({len(g)} nodes, node contract intact)")
    # One frame: length must arrive as a LINK from PrimitiveInt(1), same contract as the mac graph.
    if g[nd["length"]]["inputs"].get("value") != 1 or g[nd["r2v"]]["inputs"].get("length") != [nd["length"], 0]:
        ok = False; print("FAIL length is not linked from PrimitiveInt(1) -- the graph would render a clip")
    if PROF["ref_nodes"][0][1] not in g[nd["r2v"]]["inputs"]:
        ok = False; print("FAIL reference input ref_images.ref_image_0 absent -- renders would ignore your images")
    for cls, need in (("MiniMaxH3ReferenceToVideo", True), ("ResolutionSelector", True), ("LoadImageMask", True),
                      ("H3V2VInit", False)):
        if cls in api(f"/api/object_info/{cls}"):
            print(f"ok   {cls} registered")
        elif need:
            ok = False; print(f"FAIL {cls} not registered on the CUDA ComfyUI")
        else:
            print(f"warn {cls} missing -- --inpaint/--outpaint unavailable: install ComfyUI-MAINodes and restart")
    for nid, (widget, want) in PROF["models"].items():
        cls = g[nid]["class_type"]
        if want in combo_options(cls, widget):
            print(f"ok   {want} present ({cls})")
        else:
            ok = False; print(f"FAIL {want} not offered by {cls}.{widget} -- download it into the matching models/ dir")
    if "H3SingleFrameEnabled" in api("/api/object_info/H3SingleFrameEnabled"):
        print("ok   h3_single_frame node loaded")
    else:
        print("info h3_single_frame node not loaded (not required by this check on CUDA; install it if one-frame renders fail)")
    return ok


def doctor():
    if PROFILE == "cuda":
        return doctor_cuda()
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
    if not os.path.exists(GRAPH):
        return print(f"FAIL {GRAPH} missing -- export it once with --export") or False
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
    try:
        import cv2  # noqa: F401
        print("ok   opencv present (--autofix face detection available)")
    except ImportError:
        print("info --autofix unavailable: install the extra with  uv tool install --force -e '.[autofix]'")
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
    if N["r2v"][0] not in g:      # already reported FAIL in the node loop; avoid a KeyError traceback here
        return ok
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
    if PROFILE == "cuda":
        return build_cuda(args)
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
    mask_path = os.path.join(workdir(), f"{args.name}_mask.png")
    mask.save(mask_path)
    mask_name = stage_ref(mask_path)
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
    g["9302"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
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


def build_cuda(args):
    """Drive the shipped int8_convrot edit graph over HTTP, widgets by name (ported from the
    5090-validated h3edit_pod.py nodeset). Generate = a neutral card as the only reference."""
    if args.te or args.dit or args.encode_vae or args.encode_tiled or args.save_latent \
            or args.decode_crop or args.frames != 1:
        sys.exit("--profile cuda does not support the mac-only diagnostic flags "
                 "(--te/--dit/--encode-vae/--encode-tiled/--save-latent/--decode-crop/--frames)")
    if args.lora not in ("off", TURBO_LORA):
        sys.exit("--profile cuda runs the base model; --lora is mac-only (drop it or pass --lora off)")
    g = json.load(open(GRAPH))
    nd = PROF["nodes"]
    for key, nid in nd.items():
        if nid and nid not in g:
            sys.exit(f"api_graph.cuda.json is missing node {nid} ({key}). Re-export it.")
    steps, sampler, scheduler = args.steps, args.sampler, args.scheduler   # lane defaults resolved in main()
    for nid, (k, v) in PROF["models"].items():
        if nid in g:
            g[nid]["inputs"][k] = v
    g[nd["prompt"]]["inputs"][PROF["prompt_key"]] = args.prompt
    g[nd["res"]]["inputs"].update(aspect_ratio=ASPECTS[args.ar], megapixels=args.mp)
    g[nd["steps"]]["inputs"].update(steps=steps, scheduler=scheduler)
    g[nd["sampler"]]["inputs"]["sampler_name"] = sampler
    g[nd["seed"]]["inputs"]["noise_seed"] = args.seed
    g[nd["r2v"]]["inputs"]["ref_image_size"] = args.ref_size
    if args.vae:
        g[nd["vae"]]["inputs"]["vae_name"] = args.vae
    g[nd["save"]]["inputs"]["filename_prefix"] = "h3_edit/" + args.name
    names = [stage_ref(pth) for pth in args.refs]
    r2v = g[nd["r2v"]]["inputs"]
    slot0 = PROF["ref_nodes"][0][0]
    for i, name in enumerate(names):
        if i == 0:
            g[slot0]["inputs"]["image"] = name
        else:
            nid = f"91{i:02d}"
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
            r2v[f"ref_images.ref_image_{i}"] = [nid, 0]
    if args.inpaint:
        cuda_inpaint_wire(g, args)
    return g


def cuda_inpaint_wire(g, args):
    """Masked partial denoise on the cuda graph: the source enters ONLY as an encoded latent through
    H3V2VInit; the -r artworks are the references (already wired by build_cuda). Ported from
    h3edit_pod.py (5090, 2026-09-10)."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = args.inpaint
    src = Image.open(args.source).convert("RGB"); W, H = src.size
    if W % 32 or H % 32 or x1 > W or y1 > H or x1 - x0 < 32 or y1 - y0 < 32:
        sys.exit(f"--inpaint box {args.inpaint} must fit {args.source} ({W}x{H}), sides multiples of 32")
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle((x0 - args.grow, y0 - args.grow, x1 + args.grow, y1 + args.grow), fill=255)
    mpath = os.path.join(workdir(), f"{args.name}_mask.png"); m.save(mpath)
    mask_name = stage_ref(mpath)
    src_name = stage_ref(args.source)
    nd = PROF["nodes"]
    g["9300"] = {"class_type": "LoadImage", "inputs": {"image": src_name}}
    g["9301"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9300", 0], "vae": [nd["vae"], 0]}}
    g["9302"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
    g["9303"] = {"class_type": "H3V2VInit", "inputs": {"samples": ["9301", 0], "mask": ["9302", 0], "mask_feather": args.feather}}
    g[nd["ksampler"]]["inputs"]["latent_image"] = ["9303", 0]
    g[nd["r2v"]]["inputs"].update(width=W, height=H)
    g[nd["steps"]]["inputs"]["denoise"] = args.denoise


def stage_ref(path):
    """Put an image where ComfyUI can load it and return the server-side name. The name carries a
    content hash, so two different files that share a basename (a/ref.png, b/ref.png) or two jobs
    writing the same crop name can never overwrite each other."""
    digest = hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]
    name = f"h3e_{digest}_{os.path.basename(path)}"
    if _remote():
        return _upload(path, name)
    os.makedirs(INPUT_DIR, exist_ok=True)
    dst = os.path.join(INPUT_DIR, name)
    if not os.path.exists(dst):
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
    while time.time() - t0 < 3600:
        h = api(f"/history/{pid}")
        if h:
            v = next(iter(h.values()))
            if v["status"]["status_str"] != "success":
                sys.exit("render failed: " + json.dumps(v["status"])[:600])
            save_nid = PROF["nodes"]["save"]      # the graph's SaveImage, not whatever node emitted first
            outs = [v["outputs"].get(save_nid, {})] + list(v["outputs"].values())
            im = next((i for o in outs for i in o.get("images", [])), None)
            if im is None:
                sys.exit("render reported success but produced no image: " + json.dumps(v["status"])[:300])
            if _remote():
                q = (f"/view?filename={urllib.parse.quote(im['filename'])}"
                     f"&subfolder={urllib.parse.quote(im.get('subfolder', ''))}&type=output")
                data = urllib.request.urlopen(urllib.request.Request(
                    COMFY + q, headers={"User-Agent": "curl/8"}), timeout=600).read()
                src = args.out or os.path.join(OUTPUT_DIR, im["filename"])
                os.makedirs(os.path.dirname(os.path.abspath(src)), exist_ok=True)
                open(src, "wb").write(data)
            else:
                src = os.path.join(OUTPUT_DIR, im.get("subfolder", ""), im["filename"])
                if args.out:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    shutil.copy(src, args.out)
                    src = args.out
            print(f"{src}  ({int(time.time() - t0)}s)")
            return src
        time.sleep(10)
    sys.exit(f"render timed out after 3600s (prompt {pid} never completed)")


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
    if bright.sum() < 16:        # flat/uniform ring: no stable surface to match -> skip (else NaN gain -> black box)
        return ren
    rr, ss = rr[bright], ss[bright]
    gain = np.array([np.clip(np.median(ss[:, c]) / max(1.0, np.median(rr[:, c])), 0.8, 1.5) for c in range(3)])
    return Image.fromarray(np.clip(r * gain, 0, 255).astype(np.uint8))


def inpaint_run(args):
    """--inpaint: render, then paste only the (grown, feathered) box back onto --source. The model
    saw the frozen latent for context; the pixels outside the box never take a VAE round-trip
    (chaining masked passes without this drifted a 16 MP canvas dark by the 7th pass)."""
    from PIL import Image, ImageDraw, ImageFilter
    out = args.out
    a = copy.copy(args)                           # never mutate the caller's args (batch/outpaint reuse them)
    a.out, a.wait = None, True
    ren = Image.open(run(a)).convert("RGB")
    src = Image.open(args.source).convert("RGB")
    if a.decode_crop:                             # the render is the decoded crop: put it in place (offset set by inpaint_wire on `a`)
        full = src.copy(); full.paste(ren, a.decode_crop); ren = full
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
    crop_path = os.path.join(workdir(), f"{args.name}_crop.png")
    full.crop((x0, y0, x1, y1)).save(crop_path)
    a = copy.copy(args)                           # never mutate the caller's args: refs must not accumulate across seeds
    a.refs = [crop_path] + list(args.refs)
    a.ar = min(ASPECTS, key=lambda k: abs(int(k.split(":")[0]) / int(k.split(":")[1]) - w / h))
    a.out, a.wait = None, True
    ren = Image.open(run(a)).convert("RGB").resize((w, h), Image.LANCZOS)
    f = min(args.feather, min(w, h) // 4)          # a 48 px feather on a 64 px box inverts the mask and pastes nothing
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (f, f, w - f, h - f))
    mask = mask.filter(ImageFilter.GaussianBlur(f / 2))
    comp = full.copy()
    comp.paste(ren, (x0, y0), mask)
    comp.save(args.out)
    print(f"{args.out}  (detail box {x0},{y0},{x1},{y1} at {a.ar}, feather {f}px)")


AUTOFIX_MODELS = os.path.expanduser("~/.cache/h3edit/models")
# OpenCV Zoo's YuNet face detector (CPU, ~230 KB ONNX); fetched once to the cache on first --autofix.
# mediapipe was the plan but every arm-mac / py3.13 wheel is a Tasks-only build whose vision graphs
# abort on a Metal service check (2026-09-12). YuNet runs clean on CPU. Hands: pass --regions.
_YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

_FIX_PROMPT = {
    "face": ("Task: Reference-guided generation. <Picture 1> is a crop of a finished image showing "
             "a person's face. Reproduce it exactly -- same identity, pose, expression, framing, "
             "lighting and colours -- rendered sharper and anatomically correct: two eyes, natural "
             "symmetric features and skin. Continue the crop's style. Paint only inside the frame. No text."),
    "region": ("Task: Reference-guided generation. <Picture 1> is a crop of a finished image. "
               "Reproduce it exactly -- same content, framing, lighting and colours -- rendered sharper "
               "and anatomically correct, with natural proportions and exactly five fingers on any hand. "
               "Continue the crop's style. Paint only inside the frame. No text."),
}


def _fetch_model(url):
    """Download a detector model to the cache on first use; return the local path."""
    os.makedirs(AUTOFIX_MODELS, exist_ok=True)
    path = os.path.join(AUTOFIX_MODELS, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        print(f"fetching {os.path.basename(path)} from github.com/opencv/opencv_zoo -> {path}")
        tmp = path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)          # atomic: an interrupted download can't poison the cache
    return path


def detect_faces(img_path, min_size, conf=0.5):
    """Face boxes (pixels) via OpenCV YuNet: [(x0,y0,x1,y1,'face'), ...] plus (W,H)."""
    try:
        import cv2
    except ImportError:
        sys.exit("--autofix needs opencv: uv tool install --force -e '~/h3edit[autofix]'")
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"--autofix: cannot read image {img_path}")
    H, W = img.shape[:2]
    det = cv2.FaceDetectorYN.create(_fetch_model(_YUNET_URL), "", (W, H), score_threshold=conf)
    det.setInputSize((W, H))
    _, faces = det.detect(img)
    out = []
    for fdet in (faces if faces is not None else []):
        x, y, w, h = (int(v) for v in fdet[:4])
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if min(x1 - x0, y1 - y0) >= min_size:
            out.append((x0, y0, x1, y1, "face"))
    return out, (W, H)


def _pad_box(box, pad, W, H):
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))


def autofix(args):
    """Re-render each face (auto, YuNet) or --regions box through the grid-free --detail path IN
    PLACE. Detection finds WHERE faces are, not whether they are malformed, so every one is refined;
    the detail crop preserves identity/colour and the feathered paste blends the edge. Hands have no
    reliable CPU detector on this box, so fix them by pointing --regions at the bad one."""
    from PIL import Image, ImageDraw
    src = args.autofix
    base_name = args.name
    if args.regions:
        W, H = Image.open(src).size
        boxes = [(max(0, x0), max(0, y0), min(x1, W), min(y1, H), "region")
                 for (x0, y0, x1, y1) in args.regions]
    else:
        boxes, (W, H) = detect_faces(src, args.min_size)
    padded = [(_pad_box(b[:4], args.pad, W, H), b[4]) for b in boxes]
    padded = [pb for pb in padded if min(pb[0][2] - pb[0][0], pb[0][3] - pb[0][1]) >= 64]  # detail() floor
    if not padded:
        return print("autofix: no regions >= min-size found; nothing to do "
                     "(hands: pass --regions x0,y0,x1,y1)")
    print(f"autofix: {len(padded)} region(s): " + ", ".join(f"{k} {b}" for b, k in padded))
    if args.dry_run:
        over = Image.open(src).convert("RGB"); d = ImageDraw.Draw(over)
        for (x0, y0, x1, y1), k in padded:
            d.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=6)
            d.text((x0 + 8, y0 + 8), k, fill=(255, 0, 0))
        op = args.out or os.path.join(OUTPUT_DIR, f"{base_name}_autofix_overlay.png")
        over.save(op)
        return print(f"dry-run overlay: {op}  ({len(padded)} boxes, no renders run)")
    out = args.out or os.path.join(OUTPUT_DIR, f"{base_name}_autofix.png")
    Image.open(src).convert("RGB").save(out)   # working copy; region fixes accumulate here
    for i, ((x0, y0, x1, y1), k) in enumerate(padded):
        print(f"[{i + 1}/{len(padded)}] {k} {x0},{y0},{x1},{y1}")
        args.prompt = _FIX_PROMPT[k]
        args.source = out
        args.out = out
        args.detail = [x0, y0, x1, y1]
        args.refs = []
        args.name = f"{base_name}_af{i}"       # unique crop filename per region
        detail(args)
    print(f"autofix -> {out}")


_OP_OVER = 48   # px of real content pulled into each strip so it anchors + gives --detail an edge


def _reframe_sides(size, ar, anchor):
    """px to add per side (L,T,R,B) to reach aspect `ar` by EXTENDING only (never cropping)."""
    W, H = size
    aw, ah = (int(x) for x in ar.split(":"))
    target, cur = aw / ah, W / H
    snap = lambda v: max(0, (int(v) // 32) * 32)
    L = T = R = B = 0
    if cur < target - 1e-6:                       # too narrow -> add width
        add = snap(round(H * target) - W)
        if anchor == "left": R = add
        elif anchor == "right": L = add
        else: L = snap(add // 2); R = add - L
    elif cur > target + 1e-6:                      # too wide -> add height
        add = snap(round(W / target) - H)
        if anchor == "top": B = add
        elif anchor == "bottom": T = add
        else: T = snap(add // 2); B = add - T
    return L, T, R, B


def _extend_side(args, side, add, palette, seed, base, work):
    """Grow the working file `work` by `add` px on one side: edge-replicate into the new margin, then
    inpaint it (the frozen original conditions the continuation). No --detail finish: a detail crop of
    a blurred, edgeless margin hallucinates (graph-paper, droplets); the notched+tone-matched inpaint
    strip is the clean result."""
    from PIL import Image
    cur = Image.open(work).convert("RGB")
    W, H = cur.size
    over = _OP_OVER
    if side in ("left", "right"):
        canvas = Image.new("RGB", (W + add, H))
        canvas.paste(cur, (add, 0) if side == "left" else (0, 0))
        if side == "right":
            canvas.paste(cur.crop((W - 1, 0, W, H)).resize((add, H)), (W, 0))
            box = (W - over, 0, W + add, H)
        else:
            canvas.paste(cur.crop((0, 0, 1, H)).resize((add, H)), (0, 0))
            box = (0, 0, add + over, H)
    else:
        canvas = Image.new("RGB", (W, H + add))
        canvas.paste(cur, (0, add) if side == "top" else (0, 0))
        if side == "bottom":
            canvas.paste(cur.crop((0, H - 1, W, H)).resize((W, add)), (0, H))
            box = (0, H - over, W, H + add)
        else:
            canvas.paste(cur.crop((0, 0, W, 1)).resize((W, add)), (0, 0))
            box = (0, 0, W, add + over)
    canvas.save(work)
    user = args.prompt
    # inpaint the margin (denoise 1.0; the frozen original conditions it)
    args.out = work
    args.source = work
    args.inpaint = list(box); args.detail = None
    args.denoise = 1.0
    args.refs = [palette]
    args.seed = seed
    args.name = f"{base}_op_{side}_{seed}"
    args.prompt = (user + f"\nThe frame extends the photo on the {side} side; continue the scene "
                   "seamlessly from the existing edge. <Picture 1> is the colour palette only. "
                   "Paint only inside the frame. No text.")
    inpaint_run(args)
    args.prompt = user      # restore for the next side


def outpaint(args):
    """Extend an image past its borders (--outpaint L,T,R,B) or to an aspect ratio (--reframe AR) by
    GENERATING the new margins: per-side strips, top+bottom then left+right so corners get two
    populated neighbours; a single strip per side up to 25% of the current dimension, chunked and
    re-encoded beyond that. Verified 2026-09-12: H3 continues a scene coherently on one anchored edge."""
    from PIL import Image, ImageFilter, ImageDraw
    base = args.name
    src0 = Image.open(args.source).convert("RGB")
    if args.reframe:
        L, T, R, B = _reframe_sides(src0.size, args.reframe, args.anchor)
    else:
        L, T, R, B = args.outpaint
    L, T, R, B = (int(round(v / 32)) * 32 for v in (L, T, R, B))   # /32: strips render at /32, so snap to avoid overshoot
    plan = [("top", T), ("bottom", B), ("left", L), ("right", R)]
    fw = src0.width + L + R
    fh = src0.height + T + B
    if not any(v > 0 for _, v in plan):
        return print(f"outpaint: nothing to add (source already {src0.width}x{src0.height} for that target)")
    print(f"outpaint: {src0.width}x{src0.height} -> {fw}x{fh}  (L{L} T{T} R{R} B{B})")
    if args.dry_run:
        over = Image.new("RGB", (fw, fh), (40, 40, 40)); over.paste(src0, (L, T))
        d = ImageDraw.Draw(over); d.rectangle((L, T, L + src0.width - 1, T + src0.height - 1),
                                              outline=(0, 220, 0), width=6)
        op = args.out or os.path.join(OUTPUT_DIR, f"{base}_outpaint_plan.png")
        over.save(op)
        return print(f"dry-run plan: {op}  (green = original, grey = margins to generate; no renders)")
    work = args.out
    src0.save(work)
    palette = os.path.join(workdir(), f"{base}_op_palette.png")
    src0.resize((16, 16)).resize((256, 256), Image.NEAREST).filter(ImageFilter.GaussianBlur(20)).save(palette)
    si = 0
    for side, total in plan:
        remaining = total
        while remaining > 0:
            cur = Image.open(work).size
            dim = cur[1] if side in ("top", "bottom") else cur[0]
            cap = max(32, (int(dim * 0.25) // 32) * 32)
            add = max(32, (min(remaining, cap) // 32) * 32)
            print(f"[{side}] +{add}px (remaining {remaining})")
            _extend_side(args, side, add, palette, args.seed + si, base, work)
            si += 1
            remaining -= add
    final = Image.open(work).size
    print(f"outpaint -> {work} ({final[0]}x{final[1]})")


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
        crop_path = os.path.join(workdir(), f"{base}_up{i}_crop.png")
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
        a = copy.copy(args)                      # a fresh request per variation: nothing leaks between seeds
        a.seed = s
        a.name = f"{base_name}_s{s}"             # keep per-seed intermediates from colliding
        a.out = f"{stem}_s{s}{ext}"
        a.wait = True
        print(f"[{i + 1}/{len(seeds)}] seed {s} -> {a.out}")
        dispatch(a)
        items.append((a.out, f"seed {s}"))
    sheet = make_sheet(items, f"{stem}_sheet{ext}")
    print(f"sheet: {sheet}\nvariations: " + ", ".join(p for p, _ in items))


def main():
    p = argparse.ArgumentParser(description="Instruction-based image editing on MiniMax H3, local.")
    p.add_argument("--profile", default=os.environ.get("H3EDIT_PROFILE", "mac"), choices=["mac", "cuda"],
                   help="mac = MacMax MPS graph (default); cuda = shipped int8_convrot edit graph "
                        "(local CUDA ComfyUI via $H3EDIT_COMFY, or a remote pod)")
    p.add_argument("prompt", nargs="?", help="edit instruction; see prompts/reference_prompts.txt")
    p.add_argument("-r", "--ref", dest="refs", action="append", default=[],
                   help=f"reference image (repeatable, max {MAX_REFS})")
    p.add_argument("-o", "--out", help="write the result here (implies --wait)")
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
    p.add_argument("--steps", type=int, default=None, help="default: 8 on the mac turbo lane; 20 with --lora off or --profile cuda")
    p.add_argument("--sampler", default=None, help="KSamplerSelect sampler_name (default er_sde turbo lane; euler base lane)")
    p.add_argument("--scheduler", default=None, help="BasicScheduler scheduler (default beta57 turbo lane; simple base lane)")
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
    p.add_argument("--autofix", metavar="IMAGE",
                   help="detect faces (OpenCV YuNet) and re-render each through the grid-free --detail "
                        "path in place; needs the [autofix] extra (opencv). Hands: use --regions")
    p.add_argument("--regions", default=None,
                   type=lambda v: [[int(n) for n in b.split(",")] for b in v.split(";")],
                   help="--autofix: skip face detection, fix these boxes 'x0,y0,x1,y1;...' (e.g. a bad hand)")
    p.add_argument("--min-size", dest="min_size", type=int, default=64,
                   help="--autofix: ignore detected faces smaller than this (px)")
    p.add_argument("--pad", type=float, default=0.35,
                   help="--autofix: grow each box by this fraction per side (give --detail a real edge)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="write an overlay/plan of what would run and render nothing (only --autofix/--outpaint/--reframe/--upscale; other modes refuse it)")
    p.add_argument("--outpaint", metavar="L,T,R,B", type=lambda v: [int(x) for x in v.split(",")],
                   default=None, help="extend the image by these px per side, generating the margins")
    p.add_argument("--reframe", metavar="W:H", default=None,
                   help="extend (never crop) to reach this aspect ratio; use with --anchor")
    p.add_argument("--anchor", default="center", choices=["center", "left", "right", "top", "bottom"],
                   help="--reframe: where the original sits inside the new frame")
    p.add_argument("--generate", action="store_true",
                   help="text-to-image: generate from the prompt alone, no -r needed (auto-injects a "
                        "neutral reference). Up to 16 MP via --mp; pairs with --n for seed-select. "
                        "Ideogram 4 stays sharper for small stills -- use --generate for large-format")
    p.add_argument("--upscale", metavar="IMAGE",
                   help="[WIP - BROKEN on multi-tile scenes] enlarge + add H3 detail: Lanczos scaffold, then "
                        "saliency-gated 4 MP detail tiles recombined wavelet-style. The tile compositor does not "
                        "register cleanly yet (collaged output); needs --allow-wip to run. For a faithful upscale use a dedicated upscaler")
    p.add_argument("--scale", type=float, default=2.0, help="--upscale: enlargement factor (1 = re-detail in place)")
    p.add_argument("--tile-mp", dest="tile_mp", type=float, default=1.2, help="--upscale: crop size in MP (rendered at 4 MP)")
    p.add_argument("--overlap", type=float, default=0.2, help="--upscale: tile overlap fraction")
    p.add_argument("--edge-thresh", dest="edge_thresh", type=float, default=6.0,
                   help="--upscale: skip tiles whose Laplacian variance is below this (flat = no detail needed)")
    p.add_argument("--allow-wip", dest="allow_wip", action="store_true",
                   help="opt in to run features flagged WIP/broken (currently: --upscale)")
    p.add_argument("--wait", action="store_true", help="block until the render lands")
    p.add_argument("--doctor", action="store_true", help="check the local install and exit")
    p.add_argument("--export", metavar="TAB",
                   help="re-export api_graph.json from a ComfyUI tab id on CDP $H3EDIT_CDP")
    args = p.parse_args()
    apply_profile(args.profile)
    # Lane defaults are applied only to options left unset, so an explicit --steps 8 on cuda is honored.
    lane = PROF["lane"] if PROFILE == "mac" and args.lora != "off" else _PROFILES["cuda"]["lane"]
    for k in ("steps", "sampler", "scheduler"):
        if getattr(args, k) is None:
            setattr(args, k, lane[k])

    if args.doctor:
        sys.exit(0 if doctor() else 1)
    # Validate once, before anything is staged, saved or queued.
    if args.detail and args.inpaint:
        p.error("--inpaint and --detail are different passes; run one at a time")
    for flag in ("detail", "inpaint"):
        box = getattr(args, flag)
        if box is not None and (len(box) != 4 or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
            p.error(f"--{flag} takes X0,Y0,X1,Y1 with 0 <= X0 < X1 and 0 <= Y0 < Y1")
    if not 0 < args.denoise <= 1:
        p.error("--denoise must be in (0, 1]")
    if args.feather < 0 or args.grow < 0:
        p.error("--feather and --grow must be >= 0")
    if args.mp is not None and args.mp <= 0:
        p.error("--mp must be > 0")
    if args.n < 1:
        p.error("--n must be >= 1")
    if args.outpaint and any(v < 0 for v in args.outpaint):
        p.error("--outpaint margins must be >= 0 (it only extends; crop separately)")
    if args.export:
        from export_graph import export
        return export(args.export, GRAPH)
    if args.autofix:
        if not os.path.exists(args.autofix):
            p.error(f"--autofix: image not found: {args.autofix}")
        if args.mp is None:
            args.mp = 4.0
        if args.seed is None:
            args.seed = random.randrange(1, 2**31)
        return autofix(args)
    if args.outpaint or args.reframe:
        if not (args.source and args.out):
            p.error("--outpaint/--reframe need --source and -o")
        if not args.prompt:
            p.error("--outpaint/--reframe need a prompt describing the scene to continue")
        if args.outpaint and len(args.outpaint) != 4:
            p.error("--outpaint takes L,T,R,B (four integers)")
        if args.outpaint and args.reframe:
            p.error("--outpaint and --reframe are alternatives; pass one")
        if args.mp is None:
            args.mp = 4.0
        if args.seed is None:
            args.seed = random.randrange(1, 2**31)
        return outpaint(args)
    if args.dry_run and not args.upscale:
        p.error("--dry-run is only supported with --autofix, --outpaint, --reframe and --upscale; "
                "the other modes have nothing to plan and would render")
    if args.generate:
        if not args.prompt:
            p.error("--generate needs a prompt")
        if args.detail or args.inpaint:
            p.error("--generate is text-to-image; do not combine it with --detail/--inpaint")
        if PROFILE == "mac" and args.mp is not None and args.mp > 16:
            p.error("--generate is capped at 16 MP on mac (the ResolutionSelector rejects more); go bigger by "
                    "tiling up with --detail / --outpaint, or use --profile cuda")
        if not args.refs:
            from PIL import Image
            gray = os.path.join(workdir(), "generate_neutral.png")
            Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)
            args.refs = [gray]     # cold-start T2I: neutral card, prompt drives (verified 2026-09-12)
    if args.upscale:
        if not args.allow_wip and not args.dry_run:
            p.error("--upscale is WIP and produces broken (collaged) output on multi-tile scenes; the tile "
                    "compositor does not register yet. Re-run with --allow-wip if you want it anyway, or use "
                    "a dedicated upscaler for a faithful result.")
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
        if not (args.prompt and args.source and args.out and args.refs and len(args.inpaint) == 4):
            p.error("--inpaint needs a prompt, --source, -o, at least one -r (the artwork) and a X0,Y0,X1,Y1 box")
    elif not args.prompt or not args.refs:
        p.error("a prompt and at least one --ref are required")
    if len(args.refs) > MAX_REFS:
        p.error(f"max {MAX_REFS} references")
    if args.mp is None:
        args.mp = 2.0 if args.detail else 4.0
    if args.seed is None:
        args.seed = random.randrange(1, 2**31)
    if args.out:
        args.wait = True          # -o means "put the result here", which needs the render to land
    seeds = args.seeds if args.seeds else [args.seed + i for i in range(args.n)]
    if len(seeds) > 1:
        return batch(args, seeds)
    args.seed = seeds[0]          # a one-element --seeds is an explicit seed
    return dispatch(args)


if __name__ == "__main__":
    main()
