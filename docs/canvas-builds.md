# Building big images with `h3-inpaint`

Some images are too much for one prompt: a 16 MP group painting with twenty named figures, a street
scene where every sign has to read. `h3-inpaint` builds those on **one canvas**, one masked pass at a
time, back to front. It also works for repairing a single wrong region in a finished image.

Each pass runs `h3edit --inpaint` on a ~4 MP window cut around the box you give it, then pastes only
that box (grown and feathered) back onto the canvas in pixel space. Everything beyond the box + grow +
~feather keeps its exact pixels, so pass 40 can't slowly degrade what pass 3 painted.

It runs wherever `h3edit` runs. For a CUDA server, set `H3EDIT_PROFILE=cuda
H3EDIT_COMFY=http://host:port` before `h3-inpaint run`. (`--pod NAME` is the author's own whole-canvas
pod backend; it needs a private render kit that isn't in this repo, and it only does masked inpaint
passes.)

## Commands

```sh
h3-inpaint init   DIR --canvas start.png                  # new project: plan.json, refs/, prompts/, out/ (sides snap to /32)
h3-inpaint add    DIR NAME --box X0,Y0,X1,Y1 --prompt FILE|TEXT [-r ref.png] [--denoise 1.0]
h3-inpaint run    DIR [--only NAME] [--redo NAME]         # every pass not yet done, in plan order
h3-inpaint show   DIR [NAME]                              # 1:1 crop of a pass's box, or 12 audit tiles
h3-inpaint revert DIR NAME [--to PASS]                    # bit-exact pixel revert of a bad pass, logged
h3-inpaint add    DIR NAME --box ... --prompt fin.txt --kind detail   # reference-only re-render (grid-free, sharper)
h3-inpaint degrid DIR [--cells 32,16]                     # whole-canvas cell deblock (once, on an upscaled base)
h3-inpaint score  DIR                                     # outside/seam SSIM per pass + a making-of sheet
```

No `-r` means the pass gets a palette card sampled from the canvas (colours only). Prompt files live in
`DIR/prompts/`. Projects store paths relative to the project folder, so they open from anywhere and can
be moved. `init` refuses to overwrite an existing project unless you pass `--force`. `score` needs the
`[score]` extra.

`score` reports outside SSIM, a quarter-resolution perceptual check; anything below 1.000 means a box
leaked. It read 1.000 pass after pass on the builds below.

## Worked example: the Nachtwacht build

The [Nachtwacht build](../benchmark/nachtwacht/README.md) is a 5440x3072 militia group portrait,
painted from a blank canvas in 43 masked passes (15 on a pod, 28 local). It's a documented case study:
the persona references it used aren't in the repo, so the commands show the workflow rather than
reproduce the image byte for byte. Every prompt is in `benchmark/nachtwacht/prompts/`; the full pass
list with boxes, refs and denoise is in its `build.py` PLAN.

```sh
cd benchmark/nachtwacht
h3edit "$(cat prompts/canvas.txt)" -r refs/palette.png --ar 16:9 --mp 4 -o out/canvas.png   # the empty hall, palette card only
h3-inpaint init  . --canvas out/start.png                                      # 5440x3072, sides /32
h3-inpaint add   . banner  --box 1952,0,3488,928     --prompt banner.txt                  # back to front: banner, steps, pikes first
h3-inpaint add   . shield  --box 4352,64,5216,672    --prompt shield.txt  -r shield_art.png --denoise 0.85   # exact lettering from an artwork card
h3-inpaint add   . girl    --box 768,992,1696,2720   --prompt girl.txt    -r rosalie.png   # a persona still as <Picture 1>
h3-inpaint add   . captain --box 1632,672,2720,3072  --prompt captain.txt -r lakem_b.jpg
h3-inpaint add   . dog     --box 3584,2208,4672,3072 --prompt dog.txt                     # no -r: a palette card from the canvas
h3-inpaint run   .                                       # every pass not yet done, in order (~4 MP window, 6-7 min/pass on an M5)
h3-inpaint show  . dog                                   # 1:1 crop of that box: LOOK before the next pass
h3-inpaint revert . pikes2                               # bad pass (a gallery of militiamen in an empty box): revert, add again under a new name
h3-inpaint add   . fin_r0c0 --box 0,0,1600,1280 --prompt fin.txt --kind detail   # finish: reference-only tiles, grid-free, sharper
h3-inpaint degrid .                                      # whole-canvas cell deblock (a 2x-upscaled base)
h3-inpaint score .                                       # outside/seam SSIM per pass + making-of sheet
```

![nachtwacht](../benchmark/nachtwacht/out/final_4k.jpg)

More builds: [Bangkok Chinatown](../benchmark/bangkok/README.md) (photoreal, 24 passes) and
[Patong Beach](../benchmark/phuket/README.md), a failure analysis of what large flat regions (sky,
water, sand) do to masked composition.

## Finish with `--kind detail` tiles

A masked pass always carries the decoder's faint 16 px cell grid. A reference-only re-render of the
same box doesn't, and comes out about 2x sharper
([why](../benchmark/bangkok/README.md#the-faint-grid-what-it-was-and-what-fixed-it)). So: compose with
inpaint passes, then finish the whole canvas with detail tiles of ~1.3 MP, each with one "reproduce
exactly, sharper" prompt. Keep those boxes small and give each crop a real edge (a pure-sky crop invents
a skyline).

## Rules that each cost a pass to learn

- **Back to front.** A later box overwrites whatever it covers, references or not. If a box has to cover
  a finished neighbour's head, repaint that head right after.
- **Box = everything you're composing plus a strip of finished ground on every side.** Too small clips a
  body or decapitates a neighbour. Too big only costs the neighbours inside it.
- **An empty box is a blank page.** At `--denoise 1.0`, bare wall becomes a fresh picture (a whole
  gallery of militiamen appeared once). Props on bare ground: 0.85, and the ground survives. 1.0 needs
  existing content on at least two sides. 0.8 won't paint over lit, detailed ground.
- **Things on a detailed facade:** 0.9 keeps the architecture and adds the sign or balcony.
- **Give an animal or prop a solid surface in the prompt.** "A cat on the cart's roof edge" floated;
  "sitting on the flat steel lid" sat.
- **Vehicles before the stalls and people they pass.**
- **Ask for five fingers and audit at 1:1.** `h3-inpaint show` with no pass name writes 12 tiles. A
  six-fingered hand survived thirty passes unnoticed at half size.
- **Identity from a persona still is weak at ~150 px faces.** Costume, pose and light carry over; the
  likeness mostly doesn't. With no reference, the model copies the figures already on the canvas.
- **Never mention people you don't want.** The word "ranks" summoned a crowd.

## Prompt shape per box

1. What `<Picture 1>` supplies (identity only / colours only / lettering exactly).
2. "The frame shows …", naming what's already there and must stay.
3. The one thing to add, and where it sits relative to those anchors.
4. "The frame is a crop of a much larger finished painting: continue its light, palette, scale and
   brushwork exactly. Paint only inside the frame."

See `benchmark/nachtwacht/prompts/` for 43 real examples.

## Timings

Local turbo lane: 6–7 minutes per pass on an M5 at 4 MP windows. On a pod the whole 16 MP canvas runs
in about 65 seconds per pass.
