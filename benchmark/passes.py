#!/usr/bin/env python3
"""Second-pass experiments on a pod, all anchored to one base render (out/sign_mp4_s1001.png):

  latup    quality-lane refine on the edit graph: base sample -> LBH 2x latent upscale -> 4-step
           er_sde refine (the run_r2v_quality_latup.sh chain, nodes 9910-9917) -> decode
  inpaint  VAEEncode the base render -> H3V2VInit with a mask over the plate -> sampler at
           partial denoise (BasicScheduler.denoise < 1). Everything outside the mask is frozen.
  tiles    N x N overlapping --detail boxes rendered one after another and composited back.

Driven through ~/renderpod/h3/drive.py on the shipped edit graph, values by widget name, extra
nodes via H3_ADDNODES exactly like the pod driver adds reference loaders.

  python3 passes.py latup   --base-mp 2 --pod bench
  python3 passes.py inpaint --denoise 0.6 --pod bench
  python3 passes.py tiles   --grid 2 --pod bench
"""
import argparse, importlib.util, json, os, subprocess, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP, OUT = os.path.join(HERE, "inputs"), os.path.join(HERE, "out")
H3 = os.path.expanduser("~/renderpod/h3")
WF = f"{H3}/workflows/h3_single_image_edit_fullint8_linked.json"
PY = os.path.expanduser("~/klipsmid-render-venv/bin/python")
DRIVER = os.path.expanduser(os.environ.get("H3EDIT_POD_DRIVER") or sys.exit("set H3EDIT_POD_DRIVER to the pod driver script (not in this repo)"))
spec = importlib.util.spec_from_file_location("h3edit_pod", DRIVER)
pod = importlib.util.module_from_spec(spec); spec.loader.exec_module(pod)
LAYOUT = json.load(open(os.path.join(INP, "sign_layout.json")))
BASE = "sign_mp4_s1001"
SEED = 1001
REFINE_SIGMAS = "0.9035, 0.8000, 0.6316, 0.3158, 0.0000"
LBH = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
INT8 = {"10": {"unet_name": "minimax_h3_fl2va_int8_convrot.safetensors"},
        "11": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"},
        "2": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}}


def submit(url, prompt_path, name, mp, nodeset, addnodes, apiset, nodemodes=None):
    env = dict(os.environ, H3_URL=url, H3_NODESET=json.dumps(nodeset), H3_NODEMODES=json.dumps(nodemodes or {"18": 4, "19": 4}),
               H3_ADDNODES=json.dumps(addnodes), H3_APISET=json.dumps(apiset), H3_APIEXPECT=json.dumps(apiset))
    r = subprocess.run([PY, f"{H3}/drive.py", "run", WF, prompt_path, "1", str(mp), str(SEED), name],
                       env=env, capture_output=True, text=True, cwd=H3)
    m = pod.UUID_RE.search(r.stdout)
    if r.returncode != 0 or not m:
        sys.exit("submit failed:\n" + (r.stdout + r.stderr)[-3000:])
    return m.group(0)


def wait(url, pid, out, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = pod.comfy(url, f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                sys.exit("render failed: " + json.dumps(st)[-3000:])
            im = next((o for n in h[pid]["outputs"].values() for o in n.get("images", [])), None)
            q = f"/view?filename={urllib.request.quote(im['filename'])}&subfolder={urllib.request.quote(im.get('subfolder',''))}&type=output"
            data = urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + q, headers={"User-Agent": "curl/8"}), timeout=600).read()
            open(out, "wb").write(data)
            return time.time() - t0
        time.sleep(8)
    sys.exit("timeout")


def record(name, mp, wall, extra=None):
    p = os.path.join(OUT, "timings.json")
    t = json.load(open(p))
    w, h = Image.open(os.path.join(OUT, name + ".png")).size
    t[name] = {"backend": "pod", "mp": mp, "seed": SEED, "wall_s": round(wall, 1), "size": [w, h], **(extra or {})}
    json.dump(t, open(p, "w"), indent=1)
    print(f"done {name} {w}x{h} {wall:.0f}s", flush=True)


def plate_box():
    return json.load(open(os.path.join(OUT, "scores.json")))[BASE]["plate_box"]


def latup(a, url):
    """Base edit render at --base-mp, then LBH 2x + 4-step refine; output is 2x the base."""
    name = f"{BASE}_latup{a.base_mp:g}"
    scene = pod.upload(url, os.path.join(HERE, "..", "demos", "refs", "gable_scene-s77.png"), "bench_scene")
    plate = pod.upload(url, os.path.join(INP, "sign_plate.png"), "bench_plate")
    nodeset = {**INT8, "17": {"image": scene}, "12": {"noise_seed": SEED}, "16": {"filename_prefix": f"h3_edit/{name}"},
               "1": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": a.base_mp},
               "7": {"steps": 20, "scheduler": "simple"}, "6": {"sampler_name": "euler"}}
    addnodes = {
        "9101": {"class_type": "LoadImage", "inputs": {"image": plate}},
        "9910": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 1]}},
        "9911": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {"latent": ["9910", 0], "model_name": LBH, "mode": "scale by multiplier",
                 "mode.scale": 2.0, "align": 32, "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
        "9912": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["9911", 0], "audio_latent": ["9910", 1]}},
        "9913": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 100000}},
        "9914": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "9915": {"class_type": "ManualSigmas", "inputs": {"sigmas": REFINE_SIGMAS}},
        "9916": {"class_type": "BasicGuider", "inputs": {"model": ["14", 0], "conditioning": ["13", 0]}},
        "9917": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9913", 0], "guider": ["9916", 0], "sampler": ["9914", 0],
                 "sigmas": ["9915", 0], "latent_image": ["9912", 0]}}}
    apiset = {"13": {"ref_images.ref_image_1": ["9101", 0]}, "5": {"samples": ["9917", 0]}, "4": {"samples": ["9917", 0]}}
    pid = submit(url, os.path.join(INP, "sign.txt"), name, a.base_mp, nodeset, addnodes, apiset)
    print("queued", pid, name, flush=True)
    wall = wait(url, pid, os.path.join(OUT, name + ".png"))
    record(name, a.base_mp, wall, {"pass": "latup"})


def inpaint(a, url):
    """Freeze everything outside the plate box; re-denoise the plate from the encoded base render."""
    name = f"{BASE}_inpaint{a.denoise:g}"
    base = Image.open(os.path.join(OUT, BASE + ".png")).convert("RGB")
    x0, y0, x1, y1 = plate_box()
    m = Image.new("L", base.size, 0)
    ImageDraw.Draw(m).rectangle((x0 - a.grow, y0 - a.grow, x1 + a.grow, y1 + a.grow), fill=255)
    mpath = os.path.join(OUT, f"{name}_mask.png"); m.save(mpath)
    src = pod.upload(url, os.path.join(OUT, BASE + ".png"), "bench_inpaint_src")
    mask = pod.upload(url, mpath, "bench_inpaint_mask")
    plate = pod.upload(url, os.path.join(INP, "sign_plate.png"), "bench_plate")
    nodeset = {**INT8, "17": {"image": src}, "12": {"noise_seed": SEED + a.seed_offset}, "16": {"filename_prefix": f"h3_edit/{name}"},
               "1": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 4.0},
               "7": {"steps": a.steps, "scheduler": "simple", "denoise": a.denoise}, "6": {"sampler_name": "euler"}}
    addnodes = {
        "9101": {"class_type": "LoadImage", "inputs": {"image": plate}},
        "9300": {"class_type": "VAEEncode", "inputs": {"pixels": ["17", 0], "vae": ["2", 0]}},
        "9302": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "9303": {"class_type": "H3V2VInit", "inputs": {"samples": ["9300", 0], "mask": ["9302", 0], "mask_feather": a.feather}}}
    if a.artref:
        # the source reaches the model ONLY through the encoded latent; the artwork is <Picture 1>
        name += "_artref"
        nodeset["16"] = {"filename_prefix": f"h3_edit/{name}"}
        apiset = {"13": {"ref_images.ref_image_0": ["9101", 0]}, "8": {"latent_image": ["9303", 0]}}
        prompt = os.path.join(INP, "sign_inpaint_artref.txt")
    else:
        apiset = {"13": {"ref_images.ref_image_1": ["9101", 0]}, "8": {"latent_image": ["9303", 0]}}
        prompt = os.path.join(INP, "sign_inpaint.txt")
    pid = submit(url, prompt, name, 4.0, nodeset, addnodes, apiset)
    print("queued", pid, name, flush=True)
    wall = wait(url, pid, os.path.join(OUT, name + ".png"))
    record(name, 4.0, wall, {"pass": "inpaint", "denoise": a.denoise, "detail": [x0, y0, x1, y1]})


def tiles(a, url):
    """N x N overlapping 16:9 boxes, each a --detail pass pasted onto the running composite."""
    name = f"{BASE}_tiles{a.grid}"
    src = os.path.join(OUT, BASE + ".png")
    W, H = Image.open(src).size
    n = a.grid
    frac = (1 + a.overlap * (n - 1)) / n           # tile side as a fraction of the frame
    tw, th = int(W * frac), int(H * frac)
    step_x, step_y = (W - tw) / max(1, n - 1), (H - th) / max(1, n - 1)
    cur = src
    t0 = time.time()
    boxes = []
    for j in range(n):
        for i in range(n):
            x0, y0 = int(i * step_x), int(j * step_y)
            box = [x0, y0, x0 + tw, y0 + th]
            boxes.append(box)
            out = os.path.join(OUT, f"_{name}_{j}{i}.png")
            c = [sys.executable, DRIVER, os.path.join(INP, "sign_tile.txt"), "--source", cur, "--ref", os.path.join(INP, "sign_plate.png"),
                 "--detail", ",".join(map(str, box)), "--mp", str(a.mp), "--seed", str(SEED + j * n + i), "--name", f"bench_{name}_{j}{i}",
                 "-o", out, "--pod", a.pod, "--feather", str(a.feather)]
            r = subprocess.run(c, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"tile {j}{i} failed:\n" + (r.stdout + r.stderr)[-2000:])
            print(f"tile {j}{i} {box} {r.stdout.strip().splitlines()[-1]}", flush=True)
            cur = out
    os.replace(cur, os.path.join(OUT, name + ".png"))
    record(name, a.mp, time.time() - t0, {"pass": "tiles", "grid": n, "boxes": boxes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["latup", "inpaint", "tiles"])
    ap.add_argument("--pod", default="bench")
    ap.add_argument("--base-mp", type=float, default=2.0)
    ap.add_argument("--denoise", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grow", type=int, default=32, help="inpaint mask dilation px")
    ap.add_argument("--feather", type=int, default=48)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--artref", action="store_true", help="inpaint: artwork as <Picture 1>, source only via the latent")
    ap.add_argument("--overlap", type=float, default=0.2, help="tile overlap as a fraction of tile side")
    ap.add_argument("--mp", type=float, default=4.0, help="tile render MP")
    a = ap.parse_args()
    url = pod.podenv(a.pod)
    {"latup": latup, "inpaint": inpaint, "tiles": tiles}[a.mode](a, url)


if __name__ == "__main__":
    main()
