# CivitAI + Reddit posts — h3image v0.3.0

Live on CivitAI since 2026-09-23 as version v2.0: https://civitai.com/models/2866161
Title: MiniMax H3 Image Editor: edit + inpaint workflows for Mac and NVIDIA

## CivitAI description (as published)

MiniMax H3 is a video model, but if you ask it for a single frame it turns out to be a really good image editor. Give it a photo plus the thing you want in it (a logo, a label, a poster, a mural) and it puts it in properly: wrapped around a can, painted into brick, glowing on a wall with the reflection in the wet street. It also does plain text-to-image at big sizes, up to about 16 MP in one go.

The trick comes from Patient_Ratio4177 on r/StableDiffusion. I packaged it into ComfyUI workflows for Mac and for NVIDIA.

**What you get**

Four workflows:
- **Edit (Mac)** and **Edit (CUDA)**: two images in, one edited image out
- **Inpaint (Mac)** and **Inpaint (CUDA)**: paint over the part you want changed. Everything outside it stays untouched, apart from a soft blend at the edge.

Each one opens with an example already loaded (the demo images are in the zip). The inputs you actually touch are grouped at the top left, and there's a how-to note right there.

**Getting it running**

1. Copy the images from the zip's `input/` folder into ComfyUI's `input/` folder
2. Install the small `h3_single_frame` node from the GitHub repo and restart ComfyUI. It's what lets H3 render one frame instead of a video.
3. Mac: you also need ComfyUI-GGUF, ComfyUI-ClipProj and comfyui-obvpm. For inpainting (Mac or CUDA): ComfyUI-MAINodes.
4. Grab the models. The repo README lists every file, with links and where it goes.

Speed: about 11 minutes for a 4 MP edit on an M5 Mac, about 5 minutes on an RTX PRO 4500. CUDA needs a Blackwell card (RTX 50xx / RTX PRO) on driver 580 or newer.

**Writing prompts that work**

- Start with `Task: Reference-guided generation.`
- Refer to your images as `<Picture 1>`, `<Picture 2>` (first loaded image = Picture 1)
- Say what each picture is for, and what it is NOT for: "<Picture 2> is the label artwork only. It does not supply a background or lighting."
- Say "exactly one" when you want one of something
- List what has to stay the same in Picture 1
- Go to 4 MP if there's small lettering. At 2 MP small text turns to mush.

**What it's not good at**

- A normal edit redraws the whole picture, so fine details can shift a little. If the rest has to stay pixel-perfect, use the inpaint workflow.
- Tattoos look like stickers (last image). Detailed illustrations get distorted, but text and logos hold up.
- Inpainting can leave a very faint grid on smooth areas like sky or plain walls
- Faces from a reference photo don't hold a real likeness when they're small in the frame
- It's not an upscaler. Qwen-Image-Edit 2.1 does that job well.

**Want to script it?**

The repo also has a command-line tool that runs the same workflows: batch edits, several seeds at once with a contact sheet, extending an image past its edges, automatic face fixes, and building huge images one region at a time. `pip install git+https://github.com/Bambushu/h3image`

Repo, full docs and model list: https://github.com/Bambushu/h3image

All brands in the demos are made up and every demo image is AI-generated.

## Reddit post (draft, not posted)

TITLE
MiniMax H3 as an image editor: edit + inpaint ComfyUI workflows for Apple Silicon and NVIDIA Blackwell

BODY
A while back u/Patient_Ratio4177 showed that MiniMax H3, the 33B video model, makes a great image editor if you render just one frame. I've been building on that. It now runs on both Apple Silicon and NVIDIA, and I cleaned it up into four ComfyUI workflows you can just drag in.

What it's good at is putting a flat asset into a scene *properly*: a label that wraps around a can and sits under the condensation, a mural that takes on the brick texture, a neon sign that reflects in wet pavement, a book cover that follows the book's perspective. Text and logos transfer well in these examples, though small lettering and occasional character substitutions still need checking. It also does plain text-to-image, up to about 16 MP in one shot.

**The workflows**
- Edit (Mac / CUDA): two images in, one edit out
- Inpaint (Mac / CUDA): paint a mask in ComfyUI's mask editor; pixels outside the expanded, feathered edit region are preserved

They open with a demo loaded and a how-to note next to the inputs.

**Hardware and speed:** tested on an Apple M5 with 48 GB unified memory (~11 min for a 4 MP edit), and on NVIDIA Blackwell, including an RTX PRO 4500 (~5 min). The CUDA recipe needs Blackwell (50xx / RTX PRO) and driver 580+. These are tested configurations, not established memory minima.

**Prompt tips that made the biggest difference**
- Start with "Task: Reference-guided generation." and refer to images as <Picture 1>, <Picture 2>
- Tell it what each picture is NOT for ("<Picture 2> is the label artwork only, it does not supply a background")
- Say "exactly one" if you want one of something
- Use 4 MP for small lettering. 2 MP turns it to mush.

**Where it falls short:** a normal edit redraws the whole image, so use inpaint if the rest must stay put. Tattoos look like stickers. Inpainting can leave a faint grid on smooth surfaces. Small faces don't keep the likeness from a reference. It's not an upscaler either (Qwen-Image-Edit 2.1 is better for that).

There's also a CLI for scripting edits, seed contact sheets, and advanced canvas builds one region at a time. Outpainting and automatic face fixes are experimental. This is an early community release; feedback and reproducible bug reports are welcome.

- Workflows + demo images: https://civitai.com/models/2866161
- Repo, setup and model list: https://github.com/Bambushu/h3image

Happy to answer setup questions. Mac setup in particular has a few gotchas.
