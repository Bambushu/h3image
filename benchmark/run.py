#!/usr/bin/env python3
"""Render the benchmark. Sequential, resumable (skips outputs that exist), timings to out/timings.json.

  python3 run.py --backend local            # h3edit CLI on this machine (turbo LoRA, 8 steps)
  python3 run.py --backend pod --pod bench  # ~/renderpod/h3 pod driver (base model, 20 steps)
  python3 run.py ... --stage detail         # after `score.py`: --detail pass on the worst 4 MP seed

Tasks (12 renders + 3 detail):
  ladder   sign plate on the gable wall at 1 / 2 / 4 MP, seed 1001              3
  seeds    same at 4 MP, seeds 1002..1008                                        7
  neon     lit sign + switched-off control, same seed                            2
  detail   --detail on the lower lines of the weakest 4 MP seed: 2 MP, 4 MP, 4 MP incl. plate edge   3
"""
import argparse, json, os, subprocess, sys, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP, OUT = os.path.join(HERE, "inputs"), os.path.join(HERE, "out")
DEMOS = os.path.join(HERE, "..", "demos")
LAYOUT = json.load(open(os.path.join(INP, "sign_layout.json")))
SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
POD_DRIVER = os.path.expanduser(os.environ.get("H3EDIT_POD_DRIVER", ""))  # the pod driver (not in this repo)


def jobs():
    scene = os.path.join(DEMOS, "refs", "gable_scene-s77.png")
    plate = os.path.join(INP, "sign_plate.png")
    sign = os.path.join(INP, "sign.txt")
    j = [dict(name=f"sign_mp{mp:g}_s{SEEDS[0]}", prompt=sign, source=scene, refs=[plate], mp=mp, seed=SEEDS[0])
         for mp in (1.0, 2.0, 4.0)]
    j += [dict(name=f"sign_mp4_s{s}", prompt=sign, source=scene, refs=[plate], mp=4.0, seed=s) for s in SEEDS[1:]]
    store = os.path.join(DEMOS, "refs", "storefront_scene-s42.png")
    j.append(dict(name="neon_lit", prompt=os.path.join(DEMOS, "prompts", "neon.txt"), source=store,
                  refs=[os.path.join(DEMOS, "refs", "neon_sign-s42.png")], mp=4.0, seed=SEEDS[0]))
    j.append(dict(name="neon_unlit", prompt=os.path.join(INP, "neon_unlit.txt"), source=store,
                  refs=[os.path.join(INP, "neon_unlit.png")], mp=4.0, seed=SEEDS[0]))
    return j


def detail_job(scores):
    """Weakest 4 MP seed by the small tiers -> --detail on a 16:9 box around the four lower lines
    (the box must be a FRACTION of the frame, or the crop render buys no extra pixels)."""
    four = {k: v for k, v in scores.items() if k.startswith("sign_mp4_") and "_detail" not in k}
    if not four:
        sys.exit("score.py first: no sign_mp4_* scores in out/scores.json")
    worst = min(four, key=lambda k: sum(four[k]["ocr"][t] for t in ("S", "XS", "XXS")))
    base = os.path.join(OUT, worst + ".png")
    W, H = Image.open(base).size
    px0, py0, px1, py1 = four[worst]["plate_box"]
    ph = py1 - py0
    tiers = {t["tier"]: t for t in LAYOUT["tiers"]}
    y0 = py0 + tiers["M"]["band"][0] * ph - 0.04 * ph
    y1 = py0 + tiers["XXS"]["band"][1] * ph + 0.04 * ph
    h = y1 - y0
    w = h * 16 / 9
    cx = (px0 + px1) / 2
    box = [int(max(0, cx - w / 2)), int(max(0, y0)), int(min(W, cx + w / 2)), int(min(H, y1))]
    j = dict(name=f"{worst}_detail", prompt=os.path.join(INP, "sign_detail.txt"), source=base,
             refs=[os.path.join(INP, "sign_plate.png")], mp=None, seed=int(worst.rsplit("_s", 1)[1]), detail=box)
    # variant with context: same top, extended down past the plate's bottom edge into the brick
    y1e = min(H, py1 + 0.08 * ph)
    we = (y1e - y0) * 16 / 9
    edge = [int(max(0, cx - we / 2)), int(max(0, y0)), int(min(W, cx + we / 2)), int(y1e)]
    return [j, dict(j, name=f"{worst}_detail4", mp=4.0), dict(j, name=f"{worst}_detail4edge", mp=4.0, detail=edge)]


def cmd(j, a):
    out = os.path.join(OUT, j["name"] + ".png")
    if a.backend == "pod":
        if not POD_DRIVER:
            sys.exit("set H3EDIT_POD_DRIVER to the pod driver script (not in this repo)")
        c = [sys.executable, POD_DRIVER, j["prompt"], "--source", j["source"], "--ar", "16:9",
             "--seed", str(j["seed"]), "--name", "bench_" + j["name"], "-o", out, "--pod", a.pod]
        for r in j["refs"]:
            c += ["--ref", r]
    else:
        c = ["h3edit", open(j["prompt"]).read(), "--ar", "16:9", "--seed", str(j["seed"]),
             "--name", "bench_" + j["name"], "-o", out, "--wait"]
        if j.get("detail"):
            c += ["--source", j["source"]]
        else:
            c += ["-r", j["source"]]
        for r in j["refs"]:
            c += ["-r", r]
    if j.get("detail"):
        c += ["--detail", ",".join(map(str, j["detail"]))]
    if j.get("mp"):
        c += ["--mp", str(j["mp"])]
    return c, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "pod"], default="local")
    ap.add_argument("--pod", default="bench", help="podenv.<name>.sh in ~/renderpod/h3")
    ap.add_argument("--stage", choices=["renders", "detail"], default="renders")
    ap.add_argument("--only", help="substring filter on job name")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tpath = os.path.join(OUT, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}
    if a.stage == "detail":
        todo = detail_job(json.load(open(os.path.join(OUT, "scores.json"))))
    else:
        todo = jobs()
    for j in todo:
        if a.only and a.only not in j["name"]:
            continue
        c, out = cmd(j, a)
        if os.path.exists(out):
            print("skip ", j["name"]); continue
        print("=== ", j["name"], f"mp={j['mp']} seed={j['seed']}" + (f" detail={j['detail']}" if j.get("detail") else ""), flush=True)
        t0 = time.time()
        r = subprocess.run(c, capture_output=True, text=True)
        dt = time.time() - t0
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print("\n".join("     " + l for l in tail), flush=True)
        if r.returncode != 0 or not os.path.exists(out):
            sys.exit(f"FAILED {j['name']}")
        w, h = Image.open(out).size
        timings[j["name"]] = {"backend": a.backend, "mp": j["mp"], "seed": j["seed"], "wall_s": round(dt, 1), "size": [w, h],
                              **({"detail": j["detail"]} if j.get("detail") else {})}
        json.dump(timings, open(tpath, "w"), indent=1)
        print(f"     {w}x{h} in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
