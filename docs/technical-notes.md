# Technical notes

The measurements and internals behind the README's advice. Everything here was measured on the setups
named, with pinned seeds; the scripted versions are in [`benchmark/`](../benchmark/README.md).

## Megapixels is secretly the reference-resolution dial

With `--ref-size match` (the default), references are scaled down to the generation's pixel area, never
up. So `--mp` also caps how sharp your reference arrives.

- At `--mp 1.0`, a 1024 px wordmark was squashed to about 650 px and came back airbrushed. 2.0 is clean
  for large marks.
- The VAE is 16 px per latent cell. A ~50 px door plate spans three cells: mush at 2.0, readable at 4.0
  (plate, fleet number and phone digits). Measured 2026-09-09, seed-matched.
- `--ref-size max` pins references to a 2048 px short edge: about 10x slower (72:48 vs 7:30) with no
  visible gain.

**Did not help** (same seed, 2026-09-09): the fl2va/ref2va `b25-49` hybrid DiT (indistinguishable from
FL2VA; its GGUF carries arch tag `minimax_h3`, which city96's loader rejects), going from 20 to 50 base
steps, and close-up references for small text. Output pixels per detail is the lever.

## Detail pass numbers

Truck cab test: a 760 px box of a 4 MP frame. All lettering legible, seam invisible, +6.5 min on the M5.
Two rules from the [benchmark](../benchmark/README.md#3-detail-pass---detail--works-with-two-caveats):
the box must include a physical edge of the object (a crop of flat panel and text gets a whole new plate
hallucinated inside it), and a detail pass sharpens but does not correct: a wrong digit survives it.

## Inpaint pass numbers

On the [benchmark](../benchmark/README.md#5-second-pass-candidates-what-actually-fixes-lettering),
`--inpaint` at 0.85 corrected a wrong digit, kept the plate's size and position, and left the wall
pixel-identical: 27 s on a 5090, base model, 20 steps. With the source as `<Picture 1>` instead, the
model reproduced its own mistake at denoise 0.6, 0.85 and 1.0. That's why the source enters only as an
encoded latent.

The effective boundary is the box grown by `--grow` (32 px), with the paste-back feathered by
`--feather` (48 px). On the 2026-09-23 acceptance run the changed pixels reached ~90 px past the box;
zero pixels changed beyond box + grow + 2 x feather.

## The 16 px grid: cause and fix

**Video VAE vs image VAE.** A 2-D FFT grid score on matched seeds: the image VAE leaves a faint 16 px grid
(harmonic 33–84x background); the video VAE scores 3.6–12x, the clean range. Verified on CUDA and Apple
Silicon; on the can demo the 16 px harmonic went from 9.8 to 4.0. The bundled graphs decode with the
video VAE.

**`--inpaint` carries the grid; `--detail` doesn't.** Measured 2026-09-10/11
(`benchmark/bangkok/out/grid/`). Every frame sampled with an encoded latent as context decodes with a
16 px harmonic at 50–200x background plus a 1–3 level per-cell mosaic in smooth areas, on the pod and
locally, on the turbo and the base lane, with fp16, fp32 and CPU decode alike. The encoded latent itself
is clean. A plain R2V frame is clean, and so is `--detail`.

**Workflow rule:** compose with `--inpaint`, then re-render the box with `--detail` for the final pixels.
`--inpaint` also notches the exact harmonics, deblocks cell boundaries and tone-matches its render
before the paste, as a partial mitigation (`--no-notch`, `--no-tonematch` turn those off).

## Benchmarks (summary)

Six scripted stress tests with pinned seeds and shipped inputs: [`benchmark/`](../benchmark/README.md).

- **Lettering** (5090, 4 MP): 8 of 8 seeds render every line of a five-tier plate; caps down to 23 px at
  ≥0.96 character accuracy, holding to about 17 px; invented characters start at 12 px. The one miss in
  40 lines was a single digit swap.
- **Outside the edit:** large structure survives; brick texture doesn't.
- **Lighting:** the neon sign's pavement reflection is ΔE 21; its glow on brick is barely measurable.
- **Second-pass candidates** against one wrong digit: tiling fails; `--inpaint` and a latent-upscale
  refine fix it.
- **Stress test:** the 43-pass Nachtwacht canvas ([canvas builds](canvas-builds.md)).

![lettering ladder](../benchmark/out/sheet_ladder.png)

## CUDA specifics

- The `cuda` profile drives the shipped `int8_convrot` single-image edit graph: base model, euler /
  simple, 20 steps. The `mac` profile drives the MacMax R2V graph on the Parasyte turbo lane.
- Tested on Blackwell cards on driver ≥ 580 (cu130): RTX PRO 6000, 5090 and RTX PRO 4500 (full
  acceptance run 2026-09-23, plus a plain `pip install` on Linux).
- **One-shot resolution ceiling** (PRO 6000, 2026-09-13): the `ResolutionSelector` clamps at **16.88 MP**
  (5024x3360 at 3:2). Asking for 20–32 MP silently returns the same frame (on Mac, 24 MP is an HTTP 400).
  Warm times: 8 MP 76 s, 12 MP 109 s, 16.88 MP ~150 s.
- **One-shot 16 MP is soft at 1:1** (RTX PRO 4500 and M5, 2026-09-24, same prompt and seed): 4 MP is
  sharp, 8 MP already softer, 16 MP melts glass and ironwork on both the turbo lane and the base model;
  40 base steps instead of 20 barely changes it. The fix is the `*_16mp` workflows: 4 MP base, LBH 3D
  latent upscaler 2x, then an er_sde refine on ManualSigmas starting at 0.9035. Measured on the PRO 4500
  (seconds, including load): base 20 / refine 4 = 194, 8 = 258, 16 = 353; base 40 / refine 4 = 244,
  8 = 447, 16 = 607. Refine 4 → 8 is a clear gain, 8 → 16 a small one; base 40 + refine 16 is the
  shipped default. Re-rendering the soft one-shot with `--detail` tiles does not fix it: the tiles copy
  the softness of their reference.
- The Mac-only diagnostic flags (`--te/--dit/--encode-*/--save-latent/--decode-crop/--frames`) are
  rejected under `--profile cuda`.
- CUDA `--doctor` requires the `H3SingleFrameEnabled` marker from this repo's compatibility node
  and checks the diffusion model, text encoder, video VAE, and audio VAE. Install the node and restart
  ComfyUI before checking; a linked length of 1 alone does not verify one-frame execution.

## How the CLI talks to ComfyUI

- `H3EDIT_COMFY` is the server URL (default `http://127.0.0.1:8288`). `H3EDIT_PROFILE` sets the default
  profile.
- **File transport follows the host:** localhost means a shared filesystem (`H3EDIT_INPUT` /
  `H3EDIT_OUTPUT`, default `~/ComfyUI-h3/input` and `output`); anything else uploads and downloads over
  HTTP. An SSH tunnel to a server whose files live elsewhere looks like localhost, so force it with
  `H3EDIT_TRANSPORT=upload` (or `=shared` for a remote host on a shared mount).
- Staged images get a content-hash prefix (`h3e_<hash>_name.png`), so two references that share a file
  name never overwrite each other.
- Masks, crops and cards the CLI generates go to a per-process temp folder, never straight into
  ComfyUI's input folder.

## The graphs

The CLI queues `h3edit_graphs/api_graph.json` (mac) or `h3edit_graphs/api_graph.cuda.json` (cuda). Both
are ComfyUI's own `graphToPrompt()` exports, used as templates: the prompt, model names and seed stored
in them are placeholders the CLI overwrites. H3's reference inputs only serialize correctly through the
frontend, as dotted `ref_images.ref_image_N` keys; a hand-assembled graph drops them silently, and the
CLI refuses to queue if they're missing. The masked modes add three nodes (`VAEEncode`,
`LoadImageMask`, `H3V2VInit`) at fixed ids; nothing else is constructed.

To change the wiring, edit the workflow in the GUI and re-export with
`H3EDIT_CDP=<port> h3edit --export <tab-id>`.

The GUI workflows in `workflows/` were each loaded in the ComfyUI frontend (Mac 1.51.9, CUDA 1.52.7),
passed server validation and rendered on 2026-09-23. The inpaint ones do the CLI's pixel paste-back;
the Mac one also pulls the render's tone back to the source with core `ColorTransfer`. Neither does the
grid notch.

## Full flag reference

| flag | default | notes |
|---|---|---|
| `--profile` | `mac` | `mac` or `cuda` (also `$H3EDIT_PROFILE`) |
| `-r` / `--ref` | | reference image, repeatable, up to 5 (slots 3–5 are cloned from the exported LoadImage) |
| `-o` / `--out` | | write the result here (implies `--wait`) |
| `--ar` | `21:9` | aspect ratio: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9 |
| `--mp` | 4.0 (2.0 with `--detail`) | megapixels; also caps the references |
| `--seed` / `--n` / `--seeds` | random / 1 / | seed-select: N variations or exact seeds, plus a contact sheet |
| `--lora` | `H3-PK-Parasyte-Turbo.safetensors` @ 1.5 | `off` = base model (then 20 steps, euler / simple) |
| `--steps` / `--sampler` / `--scheduler` | 8 / er_sde / beta57 | cuda and `--lora off`: 20 / euler / simple. 14–50 base steps is flat |
| `--ref-size` | `match` | `max` is ~10x slower with no visible gain |
| `--detail X0,Y0,X1,Y1` + `--source` | | reference-only re-render of a box, pasted back feathered |
| `--inpaint X0,Y0,X1,Y1` + `--source` | | masked re-denoise; `--denoise` 0.85, `--grow` 32, `--feather` 48 |
| `--generate` | | text-to-image; a neutral card is injected when there's no `-r` |
| `--autofix IMAGE` | | faces via YuNet (`[autofix]` extra); `--regions`, `--min-size` 64, `--pad` 0.35 |
| `--outpaint L,T,R,B` / `--reframe W:H` | | extend the canvas; `--anchor` for reframe |
| `--dry-run` | | plan image only; `--autofix` / `--outpaint` / `--reframe` |
| `--doctor` | | check the setup and exit |
| `--export TAB` | | re-export the API graph from a GUI tab over CDP (`$H3EDIT_CDP`) |
