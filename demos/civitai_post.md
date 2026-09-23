# CivitAI post draft — h3image v0.3.0 (DO NOT POST WITHOUT MIKE'S REVIEW)

Title: h3image — MiniMax H3 as an image editor + generator (ComfyUI workflows + CLI, Mac & CUDA)

---

MiniMax H3 is a 33B video model. Run it in reference-to-video mode for exactly **one frame**, decode
through the video VAE, and it becomes a strong instruction-based image editor — and a large-format
text-to-image generator (one shot up to ~16 MP). Credit for the technique: Patient_Ratio4177
(r/StableDiffusion 1vo1ab3). h3image packages it for Apple Silicon and CUDA.

Repo: https://github.com/Bambushu/h3image (MIT; the models carry their own licenses)

**What's in it**

- **4 ComfyUI workflows** — edit and mask-editor inpaint, each for Apple Silicon and CUDA. Laid out
  in four numbered groups (your inputs / models / engine / output) with a how-to note inside, and
  they open with a demo preloaded.
- **A CLI** (`h3edit`, also installed as `h3image`) that drives the same graphs headlessly: edit,
  generate, masked inpaint, detail re-render, seed-select contact sheets, outpaint/reframe, face
  auto-fix, and `h3-inpaint` for building a big image one masked pass at a time.
- A tiny custom node that allows 1-frame H3 renders without patching ComfyUI.

**Two stacks**

- Apple Silicon (MPS): pruned FL2VA GGUF (Q5_K_M) + ClipProj text encoder + Parasyte turbo LoRA,
  er_sde / beta57, 8 steps. A 4 MP edit takes ~11 min on an M5.
- CUDA (Blackwell, driver ≥ 580): the int8_convrot models from the Comfy-Org/MiniMax-H3 repack, base
  model, euler / simple, 20 steps. A 4 MP edit takes ~5 min on an RTX PRO 4500.

Both were run end to end on 2026-09-23: every supported CLI mode and all four workflows.

**What holds up**

Type and flat graphic marks transfer letter for letter, and H3 applies real surface physics: a
label compresses around a can under its condensation, a mural takes on the brick with the mortar
reading through, a neon sign lights the wall and reflects in wet pavement, a book cover warps to
the cover's perspective. Masked inpaint pastes the render back in pixel space: in the CLI test,
nothing beyond the box plus its grown, feathered margin changed by a single pixel.

**What doesn't**

- A plain edit is a regeneration conditioned on the references, not a masked edit: scene and light
  carry over, fine detail gets redrawn. Use the inpaint workflow when the rest must stay pixel-exact.
- A tattoo reads as a sticker (two attempts), and detailed illustration deforms where letterforms don't.
- Masked passes carry a faint 16-px decoder grid on smooth surfaces; the CLI reduces it with a
  notch/deblock filter, the GUI workflows don't.
- Likeness from a reference face is weak at small face sizes.

It's a workflow toolkit around an existing video model — not a new trained image model and not an
upscaler (Qwen-Image-Edit 2.1 handles upscaling well). All demo brands are fictional and AI-generated.

Attached: the four workflow JSONs (`workflows/`), the demo before/after sheets (`demos/sheets/`),
and the tattoo failure.
