# h3edit-mac

MiniMax H3 image editing on Apple Silicon (ComfyUI + CLI).

## What it is

MiniMax H3 is an open-weight 33B video model. Run in reference-to-video (R2V) mode, render for exactly one frame, and decode through the video VAE — this turns it into a strong instruction-based image editor.

The technique was published by [Patient_Ratio4177](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/) and was CUDA-only. This repo is the Apple Silicon port. It ships a ComfyUI workflow, a single-frame compatibility node, and an `h3edit` CLI. The CLI queues the bundled workflow through a running ComfyUI server.

![mural demo](demos/sheets/gable.png)

*Two references in (a bare wall, a flat artwork), one image out. Brick texture stays visible through the painted area; the design stops at the window opening.*

```sh
h3edit "$(cat demos/prompts/gable.txt)" \
  -r demos/refs/gable_scene-s77.png -r demos/refs/gable_mural-s79.png \
  --ar 16:9 -o out.png --wait
h3edit --doctor
```

## Demos

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

Drag [`h3_image_edit_mac.json`](h3_image_edit_mac.json) into ComfyUI. It has 21 nodes. It is pre-loaded with the mural demo. Copy the two images from `demos/refs/` into `input/` first.

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

If ComfyUI is not on `127.0.0.1:8288`, set `H3EDIT_COMFY`. `H3EDIT_INPUT` and `H3EDIT_OUTPUT` point at `input/` and `output/`. Defaults assume `~/ComfyUI-h3/`.

## Basic edit (h3edit)

```sh
h3edit "Task: Reference-guided generation. ..." -r scene.png -r artwork.png \
  --ar 16:9 --seed 42 -o out.png --wait
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
| `--lora` | `H3-PK-Parasyte-Turbo.safetensors` @ 1.5 | turbo lane; `--lora off` = base, then use `--steps 20 --sampler euler --scheduler simple` |
| `--steps` | 8 | with turbo LoRA; 20 for base (14–50 is flat) |
| `--sampler` / `--scheduler` | `er_sde` / `beta57` | LoRA author's recipe; equal to 20-step base at 46 % of the time |
| `--mp` | 4.0 | reference-resolution and lettering dial; also caps references |
| `-r` | up to 5 | slots 3–5 added by cloning the exported LoadImage entry |
| `--ref-size` | `match` | `max`: 72:48 total; `match`: 7:30. No visible gain from `max` |
| `--ar` | `21:9` | match your scene (all demos are `16:9`) |

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

Prefer `--inpaint` when a region is wrong rather than soft. It re-denoises a box of `--source` from an encoded latent. Everything outside the box is frozen. Your `-r` artwork is the only reference.

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

## Canvas builds (`h3-inpaint`)

Use `h3-inpaint` for an image no single prompt can produce, or for a finished image with a wrong region.

Keep one canvas. Paint it in masked passes, one box at a time, back to front.

Locally each pass is `h3edit --inpaint` on a ~4 MP window cut around the box. With `--pod NAME` (a `~/renderpod/h3/podenv.NAME.sh`) the pass runs on the whole canvas through the pod kit's `drive.py` on the shipped edit graph.

Either way only the box goes back onto the canvas, in pixel space. Nothing outside a box ever moves. Outside SSIM is 1.000, pass after pass.

### Worked example: the Nachtwacht build

The [Nachtwacht build](benchmark/nachtwacht/README.md) is a 5440x3072 militia group portrait. It was painted from a blank canvas in 43 masked passes. 15 passes were on a pod. 28 passes were local.

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

*Outside every box the canvas stayed at SSIM 1.000, pass after pass; the shield names read at 1:1.*

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

Second build, photoreal: [Bangkok Chinatown, 24 passes](benchmark/bangkok/README.md).

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

A 16 MP group painting built from a blank canvas by H3 alone in 43 masked passes. 15 passes on a pod, 28 local. Every pass leaves the rest of the canvas at SSIM 1.000. See [`benchmark/nachtwacht/`](benchmark/nachtwacht/README.md).

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

### The CLI refuses to queue

- It checks for the `ref_images.ref_image_N` keys that ComfyUI's frontend serializes.
- A hand-assembled graph drops them silently.
- Edit `h3_image_edit_mac.json` in the GUI.
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

The CLI queues the bundled `api_graph.json`. This is ComfyUI's own `graphToPrompt()` export. H3's references only serialize correctly through the frontend. They appear as dotted `ref_images.ref_image_N` keys. A hand-assembled graph drops them silently. The CLI refuses to queue if those keys are missing.

To modify wiring:

- Edit `h3_image_edit_mac.json` in the GUI.
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

## License

MIT. The models carry their own licenses. Check before commercial use.
