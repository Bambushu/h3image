# h3edit-mac — MiniMax H3 image editing on Apple Silicon (ComfyUI + CLI)

MiniMax H3 is an open-weight 33B **video** model. It turns out to be a strong instruction-based
**image editor**: render exactly **one frame** of its reference-to-video (R2V) mode, decoded
through a VAE retrained for single images. The technique is
[Patient_Ratio4177's](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/)
and is CUDA-only as published; **this repo is the Apple Silicon port**. It ships a ComfyUI
workflow, a small compatibility node that enables one-frame H3, and an `h3edit` CLI that queues
the bundled workflow through a running ComfyUI server. The CLI is still called `h3edit`.

![mural demo](demos/sheets/gable.png)

*Two references in (a bare wall, a flat artwork), one image out. Brick and mortar texture
remains visible through the painted area, and the design stops at the window opening.*

```sh
h3edit "$(cat demos/prompts/gable.txt)" \
  -r demos/refs/gable_scene-s77.png -r demos/refs/gable_mural-s79.png \
  --ar 16:9 -o out.png --wait
h3edit --doctor
```

## Demos

All references are AI-generated (Ideogram 4 for the artworks, Krea 2 for the scenes), all
fictional brands. Left: the two inputs. Right: the h3edit output.

| | observed in this render |
|---|---|
| ![can](demos/sheets/can.png) | Type compresses around the **cylinder**; the scene's condensation sits **on top of** the applied label. |
| ![neon](demos/sheets/neon.png) | The added sign behaves as a **light source**: glow on the brick, and a second coloured reflection in the wet pavement alongside the existing warm one. |
| ![book](demos/sheets/book.png) | **Perspective warp** — the artwork's horizontals converge with the cover's own edges, stopping at the boards. |

**Reproduce any demo exactly:** every PNG in `demos/out/` embeds its full ComfyUI graph — drag
one into ComfyUI to load it, seed and prompt included. Prompts are in
[`demos/prompts/`](demos/prompts), reference-generation specs in [`demos/specs/`](demos/specs).
Seeds: can `58413360`, gable `123494846`, neon `1035877917`, book `583954935`, tattoo
`733870745`. Each demo shown was produced in one or two attempts.

### Not good at everything

![tattoo failure](demos/sheets/tattoo.png)

Two attempts — with a texture-rich skin reference and explicit healed-ink language — could not
make a tattoo read as ink **under** skin: both look pasted on, and the drawn tiger's face
deforms where letterforms never did. Across these five tests: flat graphics and type were
preserved best; visible curvature, relief, perspective and emitted light were usually
reproduced; detailed illustration degraded; and neither tattoo attempt produced a subsurface
ink look.

Also true everywhere: this is a **regeneration conditioned on the references, not a masked
edit**. Scene, lighting, shadows and composition carry; fine detail (chrome trim, thin edges)
gets redrawn. Right for mockups and try-ons, wrong if the output must composite pixel-exactly
against the untouched original.

## Install

Clone this repository and `cd` into it; all commands below assume that directory.

Tested configuration: **M5 Mac, 48 GB unified memory**, ComfyUI 0.32. No minimum-memory
configuration has been established — do not assume smaller Macs work. Downloads total ~33 GB,
and a run wants ~30 GB of unified memory free.

Custom node packs required: **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**,
**[ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj)** and
**[comfyui-obvpm](https://github.com/obvpm/comfyui-obvpm)** (registers the `beta57` scheduler the
turbo lane uses; skip it and run `--scheduler simple`, untested) — for the CLI and the GUI workflow alike.

**1. Models (~33 GB).** Review each model's license before commercial use. Paths are relative to
ComfyUI's `models/` dir and must match exactly — the graph refers to these names:

| file | put at | size | from |
|---|---|---|---|
| `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` (the FL2VA checkpoint in the R2V node — see below) | `diffusion_models/_h3/` | 14.1 GB | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| `qwen3vl_8b_fp8_scaled.safetensors` | `text_encoders/_h3/` | 10.6 GB | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL/tree/main/text_encoders) |
| `mmh3-8b-ClipProj-celeb-mlp.safetensors` | `clip_projections/` | 0.4 GB | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` (decode, preferred — see below) | `vae/_h3/vae/` | 3.2 GB | Comfy-Org/MiniMax-H3 |
| `minimax_h3_t1_image_vae_step1597.safetensors` (optional, grid-prone) | `vae/_h3/vae/` | 5.2 GB | [Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE) |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/_h3/vae/` | 0.6 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `H3-PK-Parasyte-Turbo.safetensors` | `loras/` | 2.1 GB | [Plaguekind/H3-Lora](https://huggingface.co/Plaguekind/H3-Lora) |

(The audio VAE is required even for stills — H3 generates audio and video jointly; the audio
side of a single frame is decoded and discarded.)

**Prefer the GUI?** Drag [`h3_image_edit_mac.json`](h3_image_edit_mac.json) into ComfyUI — 21
nodes, titled, and pre-loaded with the mural demo (copy the two images from `demos/refs/` into
ComfyUI's `input/` first). Press Queue and you should reproduce `demos/out/gable.png` exactly.

**2. The single-frame compatibility node** (this repo):

```sh
ln -s "$(pwd)/custom_nodes/h3_single_frame" /path/to/ComfyUI/custom_nodes/h3_single_frame
```

It enables `length=1` without modifying ComfyUI itself, so ComfyUI updates cannot revert it.
**Restart ComfyUI after linking.** (Stock ComfyUI floors H3 at 5 frames in two places; taking
frame 0 of a 5-frame render gives grid artifacts under the image VAE. Details in the node's
docstring. The `H3SingleFrameEnabled` node it registers is a diagnostic marker, not an editor.)

**Decode with the video VAE, not the single-image VAE (2026-09-02).** Measured with a 2-D FFT
grid score on matched seeds: decoding through `minimax_h3_t1_image_vae_step1597` leaves a faint
16-px grid (harmonic 33–84× background, visible as waxy skin; it is in the demo outputs in
`demos/out/` too), while `minimax_h3_video_vae_fp16` on the same length-1 latent scores 3.6–12×,
the same range as clean source images, with real skin texture. The bundled graphs now load the
video VAE in node 119 (it feeds both the reference encode and the decode). Verified on a CUDA pod
with the int8 stack and on Apple Silicon (the `can` demo, seed 58413360: 16-px harmonic 9.8 → 4.0).

**3. The CLI:**

```sh
uv tool install --force -e .
h3edit --doctor
```

`--doctor` checks that ComfyUI answers, that the compatibility node is loaded in the *running*
server, and that the graph still carries its reference inputs. If ComfyUI is not on
`127.0.0.1:8288`, set `H3EDIT_COMFY`; `H3EDIT_INPUT`/`H3EDIT_OUTPUT` point at your ComfyUI
`input/` and `output/` dirs (defaults assume `~/ComfyUI-h3/`).

## Usage and dials

```sh
h3edit "Task: Reference-guided generation. ..." -r scene.png -r artwork.png \
  --ar 16:9 --seed 42 -o out.png --wait
```

| flag | default | why |
|---|---|---|
| `--lora` | `H3-PK-Parasyte-Turbo.safetensors` @ 1.5 | the turbo lane; `--lora off` = base model, then use `--steps 20 --sampler euler --scheduler simple` |
| `--steps` | 8 | with the turbo LoRA; 20 for the base model (14–50 measured flat on a pod) |
| `--sampler` / `--scheduler` | `er_sde` / `beta57` | the LoRA author's recipe; measured equal to the 20-step base at 46 % of the time |
| `--mp` | 4.0 | see below — this also caps the references |
| `-r` | up to 5 | slots 3–5 are added at queue time by cloning the exported LoadImage entry, the pattern the pod driver validated |
| `--ref-size` | `match` | `max`: 72:48 total (8:44 of it sampling, the rest reference encode); `match`: 7:30 total. No visible gain from `max` in this test |
| `--ar` | `21:9` | output aspect; set it to match your scene (all included demos use `16:9`) |

**Megapixels is secretly the reference-resolution dial — and the lettering dial.** With `match`,
references are scaled *down* to the generation's pixel area (never up) — at `--mp 1.0` a 1024px
wordmark was squashed to ~650px before H3 saw it and came back airbrushed with a halo. 2.0 is
clean for large marks, but H3's VAE is **16 px per latent cell**: a ~50-px door plate spans three
cells and comes back as mush at 2.0 whatever the references show (a 2x close-up of the plate as
its own reference changed nothing, same seed). At **4.0** the same plate, fleet number and phone
digits read (measured 2026-09-09, five references, seed-matched). Timing on the M5: **~13 min**
at 4 MP / 8 turbo steps / 5 refs (28 min for the 20-step base model); ~7–10 min at 2 MP with 2 refs.

**Judge results at 100%, never at thumbnail size.** A render that looks clean at feed size can
have deformed letterforms and a soft halo. Crop the edited region and compare it against the
reference before calling it good.

**What did not help (same seed, 2026-09-09):** the fl2va/ref2va `b25-49` hybrid DiT
(smhfacct's merge, the technique author's own checkpoint, GGUF via hoidhxd) rendered
indistinguishably from plain FL2VA — and its GGUF carries the arch tag `minimax_h3`, which
city96's loader rejects (Abiray's files are tagged `wan`). More steps on the base model
(20 → 50) and a dedicated close-up reference for the small text were also flat. Output pixels
per detail is the lever.

## Prompt grammar

Cite images as `<Picture 1>` / `<Picture 2>`. The skeleton that works:

1. `Task: Reference-guided generation.`
2. **A negative role assignment** for the asset image — the anti-bleed device:
   > `<Picture 2>` is the label artwork reference and supplies nothing else. It does not supply
   > a scene, a container, a background, or lighting.
3. The edit itself, with **"exactly one"** stated in words — "the front and rear doors, spanning
   both" was rendered as one copy *per door*.
4. An exhaustive **preserve list** naming everything from `<Picture 1>` that must survive.
5. A closing sentence naming the output form: *"A single coherent photograph shows …"*

Every demo prompt in `demos/prompts/` follows this shape. The original author's 11 worked
prompts are in [`prompts/reference_prompts.txt`](prompts/reference_prompts.txt) — note some use
three or four reference pictures, and the CLI wires two; adapt in the GUI and re-export for more.

**Making reference artwork:** H3 transfers reference art literally — including unwanted borders,
barcodes and garbled micro-text that image models love to add. Generate clean, asset-only
artwork; avoid category nouns that summon packaging furniture (*poster* added a border and a
date line, *label* added an ingredient panel and a barcode — *mural painting* and *banner* were
clean); and keep any wordmark near the artwork's centre if it must survive a cylinder wrap — a
cylinder only shows ~40% of a full-width strip.

## The graph

The CLI queues the bundled `api_graph.json` — the ComfyUI frontend's own `graphToPrompt()`
export, never hand-written, because H3's reference images only serialize correctly through the
frontend (as dotted `ref_images.ref_image_N` keys; a hand-assembled graph drops them silently).
The CLI sets values into it and refuses to queue if those keys are missing. To modify the
wiring: load `h3_image_edit_mac.json` in the GUI, edit, then re-export with
`H3EDIT_CDP=<port> h3edit --export <tab-id>`.

## Credits

- **Technique** (one-frame R2V as an image editor): [Patient_Ratio4177](https://www.reddit.com/r/StableDiffusion/comments/1vo1ab3/h3_as_a_singleimage_edit_model/)
- **Single-image VAE**: [Mamad8/MiniMax-H3-Image-VAE](https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE)
- **Parasyte turbo LoRA**: [Plaguekind/H3-Lora](https://huggingface.co/Plaguekind/H3-Lora)
- **Pruned FL2VA GGUF**: [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF)
- **ClipProj projection + loader node**: [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3), [nicolab28/ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj)
- **Qwen3-VL text encoder**: [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL)
- **MiniMax H3** itself: MiniMax, via the [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) repack

This repository's contribution is the Apple Silicon port, the compatibility node, the CLI, and
the measured dial table — not the technique.

## License

MIT. The models above carry their own licenses — check them before commercial use.
