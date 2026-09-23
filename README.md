# h3image

MiniMax H3 image editing + generation, driven headlessly through ComfyUI (CLI).

## What it is

MiniMax H3 is an open-weight 33B video model. Run in reference-to-video (R2V) mode, render for exactly one frame, and decode through the video VAE — this turns it into a strong instruction-based image editor.

The technique was published by [Patient_Ratio4177](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/), originally on CUDA. This repo began as the Apple Silicon (MPS) port and ships a ComfyUI workflow, a single-frame compatibility node, and an `h3edit` CLI (also installed as `h3image`). The CLI is host-agnostic: it queues a bundled graph over HTTP against any running ComfyUI (point `H3EDIT_COMFY` at it) and sets widgets by name. The base graphs are the frontend's own exports; the masked modes add a small, fixed set of nodes (`VAEEncode`, `LoadImageMask`, `H3V2VInit`) and the CLI checks the graph's node contract before it queues.

This is a workflow toolkit around an existing video model, not a new trained image model and not a faithful upscaler.

![mural demo](demos/sheets/gable.png)

*Two references in (a bare wall, a flat artwork), one image out. Brick texture stays visible through the painted area; the design stops at the window opening.*

### What is supported

| status | modes |
|---|---|
| **stable** | edit (`-r` references + instruction), `--generate` |
| **supported, with limits** | `--inpaint` and `--detail` (single region; see the boundary note under [Modes](#modes-detail-and-inpaint)) |
| **experimental** | `--autofix`, `--outpaint` / `--reframe`, `h3-inpaint` canvas builds (case studies, not turnkey) |
| **WIP, gated** | `--upscale` (needs `--allow-wip`; broken on multi-tile scenes) |

## Quickstart

Install the CLI (after the models + node setup under [Install](#install)):

```sh
uv tool install --force -e .        # from a checkout; or: pip install git+https://github.com/Bambushu/h3image
h3edit --doctor                     # verifies ComfyUI, the compat node, the graph
```

Point `H3EDIT_COMFY` at your ComfyUI if it is not on `127.0.0.1:8288`.

```sh
# edit an image with an instruction + reference(s)
h3edit "Task: Reference-guided generation. Put the jacket from <Picture 2> on the person in <Picture 1>." \
  -r person.png -r jacket.png --ar 16:9 -o out.png

# generate a large-format image from text alone (up to 16 MP)
h3edit --generate "a grand old library interior, sunbeams, checkerboard floor" --mp 8 -o library.png
```

Prefer ComfyUI's canvas to a CLI? Drag a workflow from [`workflows/`](workflows) (edit and masked edit, Mac and CUDA) — see [Prefer the GUI?](#prefer-the-gui).

`-o` waits for the render and writes it there. Without `-o` the job is only queued (add `--wait` to block and print the output path).

| mode | flag | what it does |
|---|---|---|
| edit | *(default)* | instruction edit from 1-5 reference images |
| detail | `--detail X0,Y0,X1,Y1` | re-render a region sharper, reference-only (grid-free) |
| inpaint | `--inpaint X0,Y0,X1,Y1` | re-denoise a box of `--source` (composing new content) |
| outpaint | `--outpaint L,T,R,B` / `--reframe` | extend the canvas outward |
| generate | `--generate` | text-to-image, no reference, large-format |
| autofix | `--autofix IMAGE` | detect faces and re-render them cleaner |
| upscale | `--upscale IMAGE` | **WIP / broken** on multi-tile scenes (needs `--allow-wip`); use a dedicated upscaler for a faithful result |

Big multi-figure images and region-by-region repairs are a separate tool, **`h3-inpaint`** — see [Canvas builds](#canvas-builds-h3-inpaint). Running on CUDA or a pod instead of Apple Silicon: [next section](#cuda--non-mac---profile-cuda).

## CUDA / non-Mac (`--profile cuda`)

The CLI drives two shipped graphs, chosen with `--profile` (or `$H3EDIT_PROFILE`, default `mac`):

- `mac` — the MacMax H3 R2V graph (MPS), Parasyte turbo lane. Unchanged.
- `cuda` — the shipped `int8_convrot` single-image edit graph (`api_graph.cuda.json`), base model,
  euler/simple 20 steps. For a CUDA ComfyUI, local or a pod.

```sh
# local CUDA box (own ComfyUI, shared filesystem — same I/O model as mac):
H3EDIT_COMFY=http://127.0.0.1:8188 h3edit --profile cuda --generate "a red sailboat on turquoise sea" --mp 12 -o out.png --wait
h3edit --profile cuda --doctor

# a remote server (COMFY host is not localhost) auto-switches to /upload/image + /view:
H3EDIT_COMFY=https://<pod>-8188.proxy.runpod.net h3edit --profile cuda --generate "..." --mp 12 -o out.png
```

File transport follows the host: localhost = shared filesystem (`H3EDIT_INPUT`/`H3EDIT_OUTPUT`), anything
else = upload/download over HTTP. An SSH tunnel to a server whose files live elsewhere looks like
localhost, so force it with `H3EDIT_TRANSPORT=upload` (or `=shared` for a remote host on a shared mount).

**CUDA install.** ComfyUI with the MiniMax H3 nodes (ComfyUI >= 0.30 and `comfy-kitchen >= 0.2.26`),
[ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) for `--inpaint`/`--outpaint` (`H3V2VInit`), and these models from
the [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) repack:

| file | put at |
|---|---|
| `minimax_h3_fl2va_int8_convrot.safetensors` | `diffusion_models/` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` | `vae/` |

`h3edit --profile cuda --doctor` checks the node contract, the three model files and the nodes. Tested on
Blackwell cards on driver >= 580 (cu130): RTX PRO 6000, 5090, and RTX PRO 4500 (32 GB; full acceptance run
2026-09-23 — edit, generate, inpaint, detail, 2-seed batches, outpaint, autofix and an `h3-inpaint` canvas pass,
plus a plain `pip install` of the package). Other CUDA setups are untested. Verified on an RTX PRO 6000
and a 5090 (2026-09-13): `--generate`, `--inpaint`, `--outpaint` and `--autofix` render correctly
end-to-end through this path and `--doctor` passes. `--upscale`'s CUDA plumbing runs (tiles render,
download and composite) but its OUTPUT is currently broken on multi-tile scenes -- a pre-existing
wavelet-recombine/seam problem in the upscale compositor, reproducible from the raw tiles off-GPU, NOT
specific to CUDA. Treat `--upscale` as WIP on both platforms until that is fixed. The mac-only diagnostic flags
(`--te/--dit/--encode-*/--save-latent/--decode-crop/--frames`) are rejected under `--profile cuda`.

**One-shot resolution ceiling (measured 2026-09-13, PRO 6000).** The `ResolutionSelector` clamps a
one-shot generation at **16.88 MP (5024x3360 at 3:2)** — requesting 20/24/28/32 MP silently returns the
same 16.88 MP frame (no error, unlike the HTTP 400 seen on Mac at 24 MP). Warm render times: 8 MP 76s,
12 MP 109s, 16.88 MP ~150s. Go past 16.88 MP by tiling with `--upscale` / `--outpaint`.

## Demos

The mural at the top of this page:

```sh
h3edit "$(cat demos/prompts/gable.txt)" \
  -r demos/refs/gable_scene-s77.png -r demos/refs/gable_mural-s79.png --ar 16:9 -o out.png
```


All references are AI-generated (Ideogram 4 for artworks, Krea 2 for scenes). All brands in the demos are fictional. Left: inputs. Right: output.

| | observed in this render |
|---|---|
| ![can](demos/sheets/can.png) | Type compresses around the cylinder. Condensation sits on top of the label. |
| ![neon](demos/sheets/neon.png) | The sign behaves as a light source: glow on brick, plus a second coloured reflection in the wet pavement. |
| ![book](demos/sheets/book.png) | Perspective warp. Horizontals converge with the cover's edges, stopping at the boards. |

Reproduce any demo exactly. Every PNG in `demos/out/` embeds its full ComfyUI graph. Drag one into ComfyUI. Seed and prompt are included.

Prompts live in [`demos/prompts/`](demos/prompts). Reference specs live in [`demos/specs/`](demos/specs).

Seeds:

- can: `58413360`
- gable: `123494846`
- neon: `1035877917`
- book: `583954935`
- tattoo: `733870745`

### Not good at everything

![tattoo failure](demos/sheets/tattoo.png)

Two attempts could not make a tattoo read as ink under skin. Both look pasted on. The tiger's face deforms.

Flat graphics and type preserve best. Curvature, relief, perspective and light usually reproduce. Detailed illustration degrades.

Also true everywhere. This is regeneration conditioned on the references. It is not a masked edit. Scene, lighting and composition carry over. Fine detail gets redrawn. It is right for mockups. It is wrong if output must composite pixel-exactly.

## Install

Clone this repo and `cd` into it. Tested on M5 Mac with 48 GB unified memory and ComfyUI 0.32. No minimum-memory configuration has been established. Downloads total about 33 GB. A run wants about 30 GB free.

Required custom node packs:

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj)
- [comfyui-obvpm](https://github.com/obvpm/comfyui-obvpm) — registers the `beta57` scheduler; without it `--scheduler simple` is untested
- [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) — provides `H3V2VInit`, needed by `--inpaint`, `--outpaint` and `h3-inpaint` (not by plain edit/generate)

This section is the Apple Silicon (`mac` profile) setup. CUDA: see [its own install note](#cuda--non-mac---profile-cuda).

### 1. Models

Check licenses before commercial use. Paths are relative to ComfyUI's `models/` directory. Paths must match exactly.

| file | put at | size | from |
|---|---|---|---|
| `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` | `diffusion_models/_h3/` | 14.1 GB | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| `qwen3vl_8b_fp8_scaled.safetensors` | `text_encoders/_h3/` | 10.6 GB | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL/tree/main/text_encoders) |
| `mmh3-8b-ClipProj-celeb-mlp.safetensors` | `clip_projections/` | 0.4 GB | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` (decode, preferred) | `vae/_h3/vae/` | 3.2 GB | Comfy-Org/MiniMax-H3 |
| `minimax_h3_t1_image_vae_step1597.safetensors` (optional, grid-prone) | `vae/_h3/vae/` | 5.2 GB | [Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE) |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/_h3/vae/` | 0.6 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `H3-PK-Parasyte-Turbo.safetensors` | `loras/` | 2.1 GB | [Plaguekind/H3-Lora](https://huggingface.co/Plaguekind/H3-Lora) |

The audio VAE is required even for stills.

### Prefer the GUI?

Four ComfyUI workflows in [`workflows/`](workflows) — drag one onto the canvas:

| workflow | platform | what it does |
|---|---|---|
| [`h3image_edit_mac.json`](workflows/h3image_edit_mac.json) | Apple Silicon | instruction edit from 2 references (add more on the H3 node's next `ref_image_N` slot) |
| [`h3image_inpaint_mac.json`](workflows/h3image_inpaint_mac.json) | Apple Silicon | masked edit: paint the area in the MaskEditor, the rest stays pixel-exact |
| [`h3image_edit_cuda.json`](workflows/h3image_edit_cuda.json) | CUDA | same edit on the int8_convrot models |
| [`h3image_inpaint_cuda.json`](workflows/h3image_inpaint_cuda.json) | CUDA | same masked edit on the int8_convrot models |

Each one is laid out in four numbered groups — **1 · Your inputs** (images, instruction, seed, size), **2 · Models** (set once), **3 · Engine** (leave alone) and **4 · Output** — with a *How to use* note inside group 1. They open pre-loaded with the mural / neon-sign demo: copy `demos/refs/gable_scene-s77.png`, `gable_mural-s79.png`, `neon_sign-s42.png` and `inpaint_example.png` (pre-masked) into ComfyUI's `input/` first.

The inpaint workflows do the CLI's pixel paste-back (grown, feathered mask); the Mac one also pulls the render's tone back to the source with core ComfyUI's `ColorTransfer` (the turbo lane leaves the masked area a few levels darker). Neither does the CLI's grid notch, so smooth surfaces can show a faint cell grid that `h3edit --inpaint` removes.

All four were loaded in the ComfyUI frontend (Mac 1.51.9, CUDA pod 1.52.7), passed server validation and rendered on 2026-09-23.

### 2. Single-frame compatibility node

```sh
ln -s "$(pwd)/custom_nodes/h3_single_frame" /path/to/ComfyUI/custom_nodes/h3_single_frame
```

This enables `length=1` without modifying ComfyUI. Restart ComfyUI after linking.

Stock ComfyUI floors H3 at 5 frames. Frame 0 of a 5-frame render gives grid artifacts under the image VAE.

### 3. The CLI

```sh
uv tool install --force -e .
h3edit --doctor
```

`--doctor` checks ComfyUI, the compatibility node in the running server, and the graph's reference inputs.

If ComfyUI is not on `127.0.0.1:8288`, set `H3EDIT_COMFY`. `H3EDIT_INPUT` and `H3EDIT_OUTPUT` point at `input/` and `output/`. Defaults assume `~/ComfyUI-h3/`. Staged images get a content-hash prefix (`h3e_<hash>_name.png`) so references that share a file name never overwrite each other.

The graphs the CLI queues live in `h3edit_graphs/` and ship inside the wheel; a plain `pip install` works, a checkout is only needed for the custom node and the GUI workflow.

## Basic edit (h3edit)

```sh
h3edit "Task: Reference-guided generation. ..." -r scene.png -r artwork.png \
  --ar 16:9 --seed 42 -o out.png
```

`-r` is a reference image. You can pass up to 5 references. The first reference becomes `<Picture 1>` in the prompt. The second becomes `<Picture 2>`.

## Generate a new image (`--generate`)

Text-to-image from the prompt alone — no reference needed. H3 is a large-format generator: coherent one-shot up to **16 MP** (5472x3072), where Ideogram 4 caps ~1-2 MP.

```sh
h3edit --generate "a grand old library interior, one-point perspective, checkerboard floor, sunbeams" \
  --ar 16:9 --mp 16 -o library.png                 # 16MP one-shot (~43 min on the M5)
h3edit --generate "..." --mp 4 --n 6 -o scout.png  # scout compositions: 6 seeds + a contact sheet
```

- No `-r`: a neutral card is auto-injected and the prompt drives the image (verified cold-start, no tint).
- **`--mp` up to 16** (24 MP is rejected by the graph's ResolutionSelector). Go bigger by tiling up with `--detail` and extending with `--outpaint`.
- Reuses `--n`/`--seeds` (seed-select), `--ar`, `--lora`. Resolution reframes at a fixed seed, so scout prompt/seed at 4 MP, then commit the framing at 16 MP.
- **When to use Ideogram 4 instead:** small, sharp stills and crisp typography. `--generate` is for large-format work and for making a base to then `--inpaint` / `--detail` / `--outpaint` inside h3edit.
- Times (M5 turbo lane): ~3.5 min at 4 MP, ~17 min at 8 MP, ~28 min at 12 MP, ~43 min at 16 MP. Use a pod for volume.

## Modes and flags

| flag | default | why |
|---|---|---|
| `--lora` | `H3-PK-Parasyte-Turbo.safetensors` @ 1.5 | turbo lane; `--lora off` = base model (steps/sampler/scheduler then default to 20 / euler / simple) |
| `--steps` | 8 | with turbo LoRA; 20 with `--lora off` or `--profile cuda` (14–50 is flat) |
| `--sampler` / `--scheduler` | `er_sde` / `beta57` | LoRA author's recipe; equal to 20-step base at 46 % of the time. `euler` / `simple` with `--lora off` or `--profile cuda` |
| `--mp` | 4.0 | reference-resolution and lettering dial; also caps references |
| `-r` | up to 5 | slots 3–5 added by cloning the exported LoadImage entry |
| `--ref-size` | `match` | `max`: 72:48 total; `match`: 7:30. No visible gain from `max` |
| `--ar` | `21:9` | match your scene (all demos are `16:9`) |
| `--n` / `--seeds` | 1 | seed-select: N variations (or exact seeds) plus a labeled contact sheet |
| `--dry-run` | off | plan only, for `--autofix` / `--outpaint` / `--reframe` / `--upscale`; other modes refuse it |

### Megapixels is secretly the reference-resolution dial

It also controls lettering. With `match`, references scale down to the generation's pixel area.

At `--mp 1.0`, a 1024 px wordmark was squashed to about 650 px. It came back airbrushed. 2.0 is clean for large marks.

The VAE is 16 px per latent cell. A ~50 px door plate spans three cells. It comes back as mush at 2.0. At 4.0 the same plate, fleet number and phone digits read. Measured 2026-09-09, seed-matched.

M5 timings:

- About 13 min at 4 MP, 8 steps, 5 refs
- 28 min for 20-step base
- 7–10 min at 2 MP, 2 refs

### Judge at 100 %

Never judge at thumbnail size. Clean-looking small renders can hide deformed letterforms.

### Did not help

Same seed, 2026-09-09.

The fl2va/ref2va `b25-49` hybrid DiT was indistinguishable from FL2VA. Its GGUF carries arch tag `minimax_h3`. It is rejected by city96's loader (vs Abiray's `wan` tag).

Going from 20 to 50 base steps did not help.

Close-up references for small text did not help.

Output pixels per detail is the lever.

## Modes: detail and inpaint

### Detail pass (`--detail`)

```sh
h3edit "$(cat prompts/detail_pass_example.txt)" \
  --detail 1560,700,2320,1460 --source out.png -r door_closeup.png -o out_detailed.png
```

The box is cut from `--source`. It is inserted as `<Picture 1>`. `-r` images follow as `<Picture 2>` and up. The render (2 MP default, aspect snapped to the box) is scaled back. It is blended with a 48-px feathered edge (`--feather`).

Truck cab test: 760-px box of a 4 MP frame. All lettering legible. Seam invisible. Plus 6.5 min.

Prompt as in `prompts/detail_pass_example.txt`: `<Picture 1>` supplies everything. `<Picture 2>` is the lettering authority. "do not add, move or remove anything".

### Inpaint pass (`--inpaint`)

Prefer `--inpaint` when a region is wrong rather than soft. It re-denoises a box of `--source` from an encoded latent. Your `-r` artwork is the only reference.

**The boundary is the grown, feathered box, not the box you typed.** The mask is the box grown by `--grow` (32 px), and the paste-back is feathered by `--feather` (48 px), so pixels up to roughly grow + feather beyond your box blend toward the render. Everything past that is pixel-identical to `--source`. Leave that margin around anything that must not change.

```sh
h3edit "$(cat benchmark/inputs/sign_inpaint_artref.txt)" \
  --inpaint 400,324,2112,1288 --source out.png -r plate.png --denoise 0.85 -o out_fixed.png --wait
```

On the [benchmark](benchmark/README.md#5-second-pass-candidates-what-actually-fixes-lettering) it corrected a wrong digit. It kept the plate's size and position. It left the wall pixel-identical. 27 s on a 5090, base model, 20 steps.

`--denoise 1.0` re-composes the box instead. Needs ComfyUI-MAINodes (`H3V2VInit`; `--doctor` checks).

Do not pass the source as `-r`. As a reference the model copies its own mistakes at any denoise.

Local turbo-LoRA lane is untested for this pass. Measured on the pod's base lane.

Two rules from the [benchmark](benchmark/README.md#3-detail-pass---detail--works-with-two-caveats):

- The box must include a physical edge of the object. A crop that is only flat panel and text gets a whole new plate hallucinated inside it.
- It sharpens, it does not correct. A wrong digit in the base render survives the pass. Reroll the base seed for that.

### Auto-repair faces (`--autofix`)

`--autofix IMAGE` finds every face with OpenCV YuNet and re-renders each through the grid-free `--detail` path in place, so soft or AI faces get a supersampled, anatomically-cleaner pass without hunting boxes by hand. It detects *where* faces are, not whether they are broken, so it refines all of them.

```sh
h3edit --autofix out.png --dry-run -o overlay.png    # see the boxes first, no renders
h3edit --autofix out.png -o out_fixed.png            # refine every face, accumulate in one file
```

- `--dry-run` writes an overlay of the boxes and runs nothing — always look before spending render time.
- `--min-size` (default 64 px) skips tiny background faces; `--pad` (0.35) grows each box to give `--detail` a real edge.
- **Hands have no reliable CPU detector on the Mac** (mediapipe's wheels are Tasks-only and abort on a Metal check), so fix a bad hand by pointing detection-free `--regions x0,y0,x1,y1;...` at it. Same detail+feather pass.
- Needs the `[autofix]` extra (`uv tool install --force -e ".[autofix]"`); the YuNet model is fetched to `~/.cache/h3edit/models` on first use. `--doctor` reports availability.

### Outpaint / reframe (`--outpaint`, `--reframe`)

Extend an image past its borders, or change its aspect ratio, by *generating* the new margins instead of cropping. Each new margin is an `--inpaint` strip where the original image is the frozen context that anchors the continuation.

```sh
h3edit "continue the sunlit courtyard, same light and depth of field, no new people" \
  --outpaint 0,0,256,0 --source in.png -o out.png            # add 256px on the right (L,T,R,B)
h3edit "continue the beach and sky" --reframe 16:9 --anchor center --source in.png -o out.png
h3edit "..." --reframe 3:2 --source in.png -o out.png --dry-run    # plan image, no renders
```

- The positional prompt describes the scene to continue; give it what is *outside* the frame, and say "no new people" so a side margin beside a subject stays empty.
- **Per-side, top+bottom then left+right**, so corners are generated with two populated neighbours. Sizes snap to `/32`; `--reframe` only ever extends (never crops), `--anchor` places the original.
- A single strip per side up to ~25% of the current dimension; larger extensions are split into ≤25% chunks and re-encoded between (each new strip then anchors to real pixels). ~25% per anchored edge is the coherence ceiling — beyond ~50% total it drifts (verified 2026-09-12: H3 continues a scene on one anchored edge, but it is not a trained outpaint model).
- **No `--detail` finish on margins** — a detail crop of a blurred, edgeless margin hallucinates (it invented graph-paper and water drops in testing). The notched, tone-matched inpaint strip is the clean output; the faint quilt on smooth surfaces is the same floor as any `--inpaint`.
- `--dry-run` writes a plan image (green = original, grey = margins) and renders nothing.
### Upscale + re-detail (`--upscale`) — WIP / broken

> **Status: broken on multi-tile scenes, disabled behind `--allow-wip`.** The design below is sound in
> principle, but the tile compositor does not register the H3 renders to the scaffold — each tile
> reinterprets its crop as different content, so a multi-tile result comes out as a visible collage
> rather than a super-resolved image. Reproducible from the raw tiles off-GPU, so it is an algorithmic
> problem, not platform-specific. The real fix (parked) is geometry-preserving low-denoise V2V tiles.
> **For a faithful upscale today, use a dedicated upscaler instead.** The flag is gated:
> pass `--allow-wip` if you want to run it anyway (e.g. a single tile).

Intended behaviour: enlarge an image and add *real* H3 detail — the large-format finisher (generate/commit -> upscale -> outpaint). Lanczos scaffold, then saliency-gated 4 MP detail tiles recombined so lighting can't drift.

```sh
h3edit --upscale in.png --scale 2 -o out.png --dry-run              # tile plan (green=detail, red=skip), no renders
h3edit --upscale in.png --scale 2 -o out.png --allow-wip           # run anyway (collages on multi-tile scenes)
```

- **Lanczos enlarge** to `--scale`x (/32), then tile into ~`--tile-mp` (1.2) crops with `--overlap` (0.2), origins/sizes snapped to the 16 px VAE grid.
- **Saliency gate**: tiles below `--edge-thresh` (Laplacian variance, default 6) are left as the Lanczos scaffold — skips flat regions, which both avoids the detail-crop hallucination and saves renders. `--dry-run` prints per-tile scores to tune it.
- **Wavelet recombine**: each tile keeps the scaffold's low frequencies (lighting/colour) and takes only the H3 render's high frequencies (texture), so tiles can't drift tile-to-tile.
- **Cost is real**: each tile is a 4 MP render (~3.5-12 min on the M5 depending on crop size). A 2x of a 4 MP image is ~28 tiles -> hours locally; use a pod for large jobs.
- **Not for text/faces**: detail tiles mangle letterforms and can shift a face -- repair those with `--inpaint` / `--autofix` after. For a fast, faithful, light re-detail instead, use a dedicated upscaler; `--upscale` is for heavy generative detail. Needs the `[upscale]` extra (scipy).

## Canvas builds (`h3-inpaint`)

Use `h3-inpaint` for an image no single prompt can produce, or for a finished image with a wrong region.

Keep one canvas. Paint it in masked passes, one box at a time, back to front.

Each pass is `h3edit --inpaint` on a ~4 MP window cut around the box, so it runs wherever `h3edit` does: for a CUDA server set `H3EDIT_PROFILE=cuda H3EDIT_COMFY=http://host:port` before `h3-inpaint run`. (`--pod NAME` is the author's own whole-canvas pod backend; it needs a private render kit that is not in this repo and only runs masked inpaint passes.)

Only the grown, feathered box goes back onto the canvas, in pixel space; everything beyond box + grow + ~feather keeps its exact pixels. `h3-inpaint score` reports outside SSIM (a quarter-resolution perceptual check), which read 1.000 pass after pass on the builds below. Projects store paths relative to the project directory, so they open from any working directory; `init` refuses to overwrite an existing project unless you pass `--force`.

### Worked example: the Nachtwacht build

The [Nachtwacht build](benchmark/nachtwacht/README.md) is a 5440x3072 militia group portrait. It was painted from a blank canvas in 43 masked passes. 15 passes were on a pod. 28 passes were local. It is a documented case study: the persona references it used are not in the repo, so the commands below show the workflow rather than reproduce the image byte for byte.

Every prompt is in `benchmark/nachtwacht/prompts/`. The full pass list with boxes, refs and denoise is in its `build.py` PLAN.

```sh
cd benchmark/nachtwacht
h3edit "$(cat prompts/canvas.txt)" -r refs/palette.png --ar 16:9 --mp 4 -o out/canvas.png --wait   # the empty hall, palette card only
h3-inpaint init  . --canvas out/start.png                                      # 5440x3072, sides /32
h3-inpaint add   . banner  --box 1952,0,3488,928     --prompt banner.txt                  # back to front: banner, steps, pikes first
h3-inpaint add   . shield  --box 4352,64,5216,672    --prompt shield.txt  -r shield_art.png --denoise 0.85   # exact lettering from an artwork card
h3-inpaint add   . girl    --box 768,992,1696,2720   --prompt girl.txt    -r rosalie.png   # a persona still as <Picture 1>
h3-inpaint add   . captain --box 1632,672,2720,3072  --prompt captain.txt -r lakem_b.jpg
h3-inpaint add   . dog     --box 3584,2208,4672,3072 --prompt dog.txt                     # no -r: a palette card from the canvas
h3-inpaint run   .                                       # every pass not yet done, in order (local: ~4 MP window, 6-7 min/pass on an M5)
h3-inpaint run   . --pod nw                              # same, on a pod: whole canvas per pass, base model 20 steps, ~65 s at 16 MP
h3-inpaint show  . dog                                   # 1:1 crop of that box: LOOK before the next pass
h3-inpaint revert . pikes2                               # bad pass (a gallery of militiamen in an empty box): pixel revert, add again under a new name
h3-inpaint add   . fin_r0c0 --box 0,0,1600,1280 --prompt fin.txt --kind detail   # finish: reference-only tiles, grid-free, sharper
h3-inpaint degrid .                                      # whole-canvas cell deblock (a 2x-upscaled base)
h3-inpaint score .                                       # outside/seam SSIM per pass + making-of sheet
```

![nachtwacht](benchmark/nachtwacht/out/final_4k.jpg)

*Outside every (grown, feathered) box the canvas stayed at SSIM 1.000, pass after pass; the shield names read at 1:1.*

### Finish every canvas with `--kind detail` tiles

Use ~1.3 MP boxes. Use one "reproduce exactly, sharper" prompt per box.

A masked-latent pass always carries the decoder's 16 px cell grid. A reference-only re-render of the same box does not. It is 2x sharper ([why](benchmark/bangkok/README.md#the-faint-grid-what-it-was-and-what-fixed-it)).

Keep those boxes small. Give each crop a real edge.

### Rules

Each of these cost a pass to learn on the [43-pass Nachtwacht build](benchmark/nachtwacht/README.md).

- Back to front. A later box overwrites whatever it covers, references or not. A box that has to cover a finished neighbour's head gets that head repainted right after.
- Box = the whole thing you are composing plus a strip of finished ground on every side. Too small clips a body or decapitates a neighbour. Too big only costs the neighbours inside it.
- An empty box is a blank page. At `--denoise 1.0` bare wall becomes a fresh picture (a whole gallery of militiamen once). Props on bare ground: `--denoise 0.85`, the ground survives. 1.0 needs existing content on at least two sides. 0.8 will not paint over lit, detailed ground.
- Ask for five fingers and audit at 1:1. `h3-inpaint show` with no pass name writes 12 tiles. A six-fingered hand survived thirty passes unnoticed at half size.
- Identity from a persona still is weak at ~150 px faces. Costume, pose and light carry over. The likeness mostly does not. With no reference the model copies the figures already on the canvas.

Prompt shape per box:

1. What `<Picture 1>` supplies (identity only / colours only / lettering exactly).
2. "The frame shows …" naming what is already there and must stay.
3. The one thing to add and where it sits relative to those anchors.
4. "The frame is a crop of a much larger finished painting: continue its light, palette, scale and brushwork exactly. Paint only inside the frame."

See `benchmark/nachtwacht/prompts/`.

### Local and pod timings

Local turbo lane: 6–7 min per pass on an M5 at 4 MP windows. On a pod the same graph runs at full 16 MP in about 65 s per pass (`h3-inpaint run --pod NAME`).

Second build, photoreal: [Bangkok Chinatown, 24 passes](benchmark/bangkok/README.md). Third, a failure analysis: [Patong Beach](benchmark/phuket/README.md), where large flat regions (sky, water, sand) broke masked composition.

## The 16-px grid: cause and fix

### Video VAE vs image VAE

Decode with the video VAE.

A 2-D FFT grid score on matched seeds showed:

- The image VAE leaves a faint 16-px grid. Harmonic is 33–84× background.
- The video VAE scores 3.6–12×. That is the clean range.

Verified on CUDA and Apple Silicon. On the can demo: 16-px harmonic went 9.8 → 4.0.

Bundled graphs load the video VAE in node 119.

### `--inpaint` carries the grid; `--detail` does not

Measured 2026-09-10/11 (`benchmark/bangkok/out/grid/`).

Every frame sampled with an encoded latent as context (`--inpaint`, `H3V2VInit`) decodes with:

- a 16-px harmonic at 50–200× background
- a 1–3 level per-cell mosaic in smooth areas, on the pod and locally
- on the turbo and the base lane, with fp16, fp32 and CPU decode alike

The encoded latent itself is clean.

A plain R2V frame is clean. `--detail` is clean (the crop as `<Picture 1>`, no latent).

### Workflow rule

- Compose with `--inpaint`.
- Then re-render the box with `--detail` for the final pixels.

`--inpaint` notches the exact harmonics and deblocks cell boundaries in its render before the paste, as a partial mitigation. `--no-notch` turns that off.

## Benchmarks

Six scripted stress tests. Pinned seeds. Shipped inputs. See [`benchmark/`](benchmark/README.md).

### Lettering

On a 5090 at 4 MP:

- 8 of 8 seeds render every line of a five-tier plate.
- Caps down to 23 px at ≥0.96 character accuracy.
- Lettering holds to about 17 px caps.
- Lettering starts inventing characters at 12 px.
- The one miss in 40 lines is a single digit swap (reroll).

### What survives outside the edit

- Large structure: yes.
- Brick texture: no.

### Lighting

- Neon sign's pavement reflection: ΔE 21, local.
- Neon sign's glow on brick: barely measurable.

### Second-pass candidates

Seven candidates against one wrong digit:

- Tiling fails.
- `--inpaint` fixes it.
- A latent-upscale refine fixes it.

### Stress test

A 16 MP group painting built from a blank canvas by H3 alone in 43 masked passes. 15 passes on a pod, 28 local. Outside SSIM (quarter-res) read 1.000 after every pass. See [`benchmark/nachtwacht/`](benchmark/nachtwacht/README.md).

![nachtwacht](benchmark/nachtwacht/out/final_4k.jpg)

![lettering ladder](benchmark/out/sheet_ladder.png)

## Prompt grammar

Cite images as `<Picture 1>` and `<Picture 2>`.

1. `Task: Reference-guided generation.`
2. A negative role assignment for the asset image: "`<Picture 2>` is the label artwork reference and supplies nothing else. It does not supply a scene, a container, a background, or lighting."
3. The edit, with **"exactly one"** stated in words.
4. An exhaustive preserve list for `<Picture 1>`.
5. Closing: "A single coherent photograph shows …".

Every demo prompt follows this shape.

The author's 11 worked prompts: [`prompts/reference_prompts.txt`](prompts/reference_prompts.txt). Some use three or four references. The CLI takes up to five.

### Making reference artwork

H3 transfers art literally. This includes borders, barcodes and garbled micro-text.

Generate clean, asset-only artwork. Avoid category nouns that summon packaging furniture:

- *poster* added a border.
- *label* added an ingredient panel and barcode.
- *mural painting* was clean.
- *banner* was clean.

Keep wordmarks near centre if they must survive a cylinder wrap. Only about 40 % of a full-width strip shows.

## Troubleshooting

### `h3edit --doctor` reports a problem

- Run `--doctor`. It checks ComfyUI, the compatibility node in the running server, and the graph's reference inputs.
- Make sure you restarted ComfyUI after linking the node.
- If ComfyUI is not on `127.0.0.1:8288`, set `H3EDIT_COMFY`.

### Lettering is mush or airbrushed

- Raise `--mp`. The default is 4.0. The VAE is 16 px per latent cell. Small marks need more megapixels.
- Judge at 100 %, not thumbnail size.
- Did you pass `--detail` for a crop of lettering? Detail passes sharpen; they do not correct a wrong base render.

### A region is wrong

- Use `--inpaint`, not `--detail`. It re-denoises a box from an encoded latent.
- `--denoise 0.85` for props on bare ground.
- `--denoise 1.0` re-composes the box and needs existing content on at least two sides.
- Do not pass the source as `-r`. As a reference the model copies its own mistakes at any denoise.

### A faint 16-px grid is visible

- Decode with the video VAE, not the image VAE. Bundled graphs already do this (node 119).
- The notch and deblock in `--inpaint` are on by default and are only a partial mitigation. `--no-notch` turns them off.

### `-o` was ignored / nothing was written

- `-o` implies `--wait`. `--inpaint` and `--detail` require `-o`.
- Without `-o`, a plain edit only queues; the result lands in ComfyUI's `output/h3_edit/`.

### The CLI refuses to queue

- It checks for the `ref_images.ref_image_N` keys that ComfyUI's frontend serializes.
- A hand-assembled graph drops them silently.
- Edit `workflows/h3image_edit_mac.json` (or `_cuda`) in the GUI.
- Re-export with `H3EDIT_CDP=<port> h3edit --export <tab-id>`.

### The scheduler complains

- Install [comfyui-obvpm](https://github.com/obvpm/comfyui-obvpm). It registers `beta57`. Without it `--scheduler simple` is untested.

### The arch tag is rejected

- Use the Abiray GGUF (`wan` arch tag). The `minimax_h3` tag is rejected by city96's loader.

### Hands come out with six fingers

- Ask for five fingers in the prompt.
- Audit at 1:1 with `h3-inpaint show` (no pass name writes 12 tiles).

### Identity drifts at small face sizes

- Identity from a persona still is weak at ~150 px faces.
- Costume, pose and light carry over; the likeness mostly does not.
- With no reference the model copies the figures already on the canvas.

### Reference artwork adds unwanted furniture

- Avoid category nouns like *poster* or *label*. They summon borders, ingredient panels, barcodes.
- Use *mural painting* or *banner* instead.

### A box overpaints a neighbour

- Order passes back to front. A later box overwrites whatever it covers, references or not.
- Make the box include the whole thing you are composing plus a strip of finished ground on every side.

## The graph

The CLI queues `h3edit_graphs/api_graph.json` (mac) or `h3edit_graphs/api_graph.cuda.json` (cuda). Both are ComfyUI's own `graphToPrompt()` exports, used as API templates: the prompt, model names and seed stored in them are placeholders the CLI overwrites at queue time. H3's references only serialize correctly through the frontend. They appear as dotted `ref_images.ref_image_N` keys. A hand-assembled graph drops them silently. The CLI refuses to queue if those keys are missing. The masked modes add three nodes (`VAEEncode`, `LoadImageMask`, `H3V2VInit`) at fixed ids; nothing else is constructed.

To modify wiring:

- Edit `workflows/h3image_edit_mac.json` (or `_cuda`) in the GUI.
- Re-export with `H3EDIT_CDP=<port> h3edit --export <tab-id>`.

## Credits

- Technique: [Patient_Ratio4177](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/)
- Single-image VAE: [Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE)
- Turbo LoRA: [Plaguekind/H3-Lora](https://huggingface.co/Plaguekind/H3-Lora)
- Pruned FL2VA GGUF: [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF)
- ClipProj: [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3), [nicolab28/ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj)
- Text encoder: [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL)
- MiniMax H3: MiniMax, via [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)

This repo's contribution: the Apple Silicon port, the compatibility node, the CLI, and the measured dial table.

## Development

```sh
pip install -e ".[test]" && pytest -q tests     # CPU only: renders are mocked, no ComfyUI needed
```

The tests pin every bug from the 2026-09-14 review (batch isolation, reference staging, tone match, small detail boxes, dry-run, CUDA flags, canvas project state). CI runs them plus a wheel install check.

## License

MIT. The models carry their own licenses. Check before commercial use.
