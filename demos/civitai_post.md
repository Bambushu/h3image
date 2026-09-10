# CivitAI post draft — h3edit (DO NOT POST WITHOUT MIKE'S REVIEW)

Title: MiniMax H3 Image Editor — Mac / Apple Silicon (1-frame R2V workflow + CLI)

---

H3 is a video model, but render exactly ONE frame of R2V through Mamad8's single-image VAE and
it becomes a serious instruction-based image editor. Credit for the technique goes to
Patient_Ratio4177 (post 1vo1ab3) — this is the Apple Silicon port of it, since the original
recipe is CUDA-only (int8_convrot + nvfp4 + comfy-kitchen).

What's in the repo (github.com/Bambushu/h3edit-mac):
- the ComfyUI workflow (GGUF Ref2VA + lightx2v turbo LoRA + ClipProj, sa_solver/8, fully MPS)
- a `h3edit` CLI: one command, two reference images in, one edit out
- a tiny custom node that unlocks 1-frame renders WITHOUT patching ComfyUI's source (survives updates)
- 4 worked demos with full prompts + 1 honest failure

What holds up: type and flat graphic marks transfer letter-for-letter, and H3 applies real
surface physics — a label compresses around a can under its condensation, a mural absorbs into
brick with the mortar joints reading through, a neon sign casts its glow onto the wall AND into
the wet pavement, a book cover warps to the cover's own perspective.

What doesn't: a tattoo reads as a sticker (twice), and detailed illustration deforms where
letterforms never did. It's a regeneration conditioned on the refs, not a masked edit — fine
trim gets redrawn.

~7-10 min per image at 2208x960 on an M5. All demo brands are fictional and AI-generated.

Attached: the workflow JSON, the 4 demo images + the failure, before/after sheets.
