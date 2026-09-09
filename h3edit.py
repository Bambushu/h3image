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
import argparse, json, os, random, shutil, sys, time, urllib.request

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
     "save": ("664", "SaveImage")}
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
    return g


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
    p.add_argument("--feather", type=int, default=48, help="paste-back edge feather in px (--detail)")
    p.add_argument("--steps", type=int, default=8, help="8 with the turbo LoRA; 20 with --lora off")
    p.add_argument("--sampler", default="er_sde", help="KSamplerSelect sampler_name (euler with --lora off)")
    p.add_argument("--scheduler", default="beta57", help="BasicScheduler scheduler (simple with --lora off)")
    p.add_argument("--lora", default=TURBO_LORA, help="LoRA file in models/loras, or 'off' (base model, 20 steps)")
    p.add_argument("--lora-strength", type=float, default=1.5)
    p.add_argument("--dit", default=None, help="override the GGUF DiT (unet_name)")
    p.add_argument("--te", default=None, help="GGUF text encoder file instead of ClipProj")
    p.add_argument("--ref-size", default="match", choices=["match", "max"],
                   help="'max' pins refs to a 2048px short edge: ~10x slower, no better")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default="h3_edit", help="output filename prefix")
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
    if args.detail:
        if not (args.prompt and args.source and args.out and len(args.detail) == 4):
            p.error("--detail needs a prompt, --source, -o and a X0,Y0,X1,Y1 box")
        if len(args.refs) > MAX_REFS - 1:
            p.error(f"--detail: max {MAX_REFS - 1} extra references (the crop is <Picture 1>)")
    elif not args.prompt or not args.refs:
        p.error("a prompt and at least one --ref are required")
    if len(args.refs) > MAX_REFS:
        p.error(f"max {MAX_REFS} references")
    if args.mp is None:
        args.mp = 2.0 if args.detail else 4.0
    if args.seed is None:
        args.seed = random.randrange(1, 2**31)
    if args.detail:
        return detail(args)
    run(args)


if __name__ == "__main__":
    main()
