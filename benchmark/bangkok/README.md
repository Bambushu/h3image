# Bangkok Chinatown: a 16 MP photoreal street built by H3 alone, one masked pass at a time

The second canvas build, and the first photoreal one: Yaowarat Road at dusk, 5440x3072, painted
from an empty street in 24 masked passes with `h3-inpaint`, every one of them on the M5 through the
local turbo lane. Nothing outside a pass's box ever moves (outside SSIM 1.000 on all 24).

![final](out/final_4k.jpg)

**Numbers (M5, Parasyte turbo lane, 8 steps, 2026-09-10):** canvas 550 s at 4 MP, then 24 passes at
310–690 s each in ~4 MP windows cut from the 16 MP canvas; seam SSIM 0.95–0.98. Two rounds, one
repair, two misses. Every prompt is in `prompts/`, the plan in `plan.json`, every box and seed in
`state.json`.

![making of](out/sheet_makingof.png)

## How it was built

```sh
h3edit "$(cat prompts/canvas.txt)" -r refs/palette.png --ar 16:9 --mp 4 --seed 8800 -o out/canvas_4mp.png --wait
# Lanczos x2 -> out/start.png (5440x3072)
h3-inpaint init . --canvas out/start.png
h3-inpaint add  . neon_left --box 688,1088,2496,2448 --prompt neon_left.txt --denoise 0.9   # ... 23 more
h3-inpaint run  .
h3-inpaint show . neon_left      # 1:1 after every pass
h3-inpaint score .
```

The canvas prompt asked for the empty setting only: shophouses, cables, wet road, dusk, "no
vehicles, no people, no stalls, no lit neon yet". Everything else is a box.

## Round 1: fifteen boxes, back to front

| pass | box (px) | denoise | result |
|---|---|---|---|
| neon_left, neon_right | 1808x1360, 1768x1360 | 0.9 | lit vertical signs on both arcades: Chinese, Thai, a green pharmacy sign; the facades untouched |
| far_street | 912x592 | 1.0 | evening traffic at the vanishing point: an orange bus, a tuk-tuk, motorbikes, tail lights |
| lanterns | 2400x992 | 0.85 | two strings of red lanterns across the street; sky and cables kept |
| gold_shop | 928x976 | 1.0 | two shutters became a lit gold shop, red fascia, glass counters, two staff |
| seafood_stall | 1184x736 | 1.0 | cart, ice bed, charcoal grill, red stools, a cook turning skewers |
| noodle_cart | 912x704 | 1.0 | glass-sided cart, hanging roast duck, a woman ladling broth |
| tuk_tuk | 768x608 | 1.0 | head-on, driver and two passengers, reflected in the wet road |
| cat | 376x272 | 1.0 | **miss**: "on the cart's roof edge" put the cat in mid-air beside the awning post |
| taxi | 1408x672 | 1.0 | pink Toyota, lit roof sign, plate; **its box covered most of the seafood stall** |
| scooter | 816x576 | 1.0 | a green delivery rider with a food box |
| crowd_right | 864x1056 | 1.0 | four pedestrians as listed in the prompt, every hand five-fingered |
| tourists | 1008x1120 | 1.0 | a couple photographing the street, backpack, sunhat, phone |
| monk | 544x768 | 1.0 | saffron robes, alms bowl in its sling, barefoot |
| soi_dog | 544x400 | 1.0 | asleep against the column base |

## Round 2: a repair and eight additions

| pass | box (px) | denoise | result |
|---|---|---|---|
| paifang | 672x624 | 1.0 | a lit Chinese gate over the far end of the road; a touch large and saturated for the distance |
| balcony_right | 1216x1056 | 0.9 | a woman watering plants, laundry on the line, a bird cage; facade re-composed a little, coherent |
| balcony_left | 1184x1088 | 0.9 | an old man smoking at the rail, a bamboo cage, bougainvillea; the arched windows intact |
| stall_fix | 432x544 | 1.0 | the floating cat and the bare table frame gone; a glass cart with fish on ice and the cat on its lid, though it stands in the lane behind the taxi rather than on the pavement |
| shrine | 416x756 | 1.0 | red and gold altar, Buddha, joss sticks, oranges, couplets |
| child | 288x552 | 1.0 | a girl with a red balloon holding the floral-dress woman's hand; the dress got repainted lighter and the shopping bag went |
| moto_taxi | 850x352 | 1.0 | two riders in orange vests on a parked scooter, cut by the bottom edge |
| puddle | 800x224 | 0.85 | a mirror sheet of standing water reflecting the tuk-tuk and the neon |
| street_sign | 432x480 | 0.85 | blue Bangkok sign on the lamp post reading exactly ถนนเยาวราช / YAOWARAT RD from an artwork card |

## The faint grid, found and fixed

Mike had seen a faint grid on both canvas builds. Measured here: a horizontal 16 px cell pattern,
50–200x background at the 16 px FFT harmonic, on every inpaint render, inside the regenerated box as
much as in the frozen area, pod base lane and local turbo lane alike. The plain R2V canvas scores
1–5x. Tested on a 1 MP inpaint that reproduces it in five minutes (`out/grid/`):

| variable | result |
|---|---|
| decode VAE: image VAE instead of video VAE | worse (150–175x); the two files share the encoder tensor for tensor |
| five frames instead of one on the sampling side | worse (60x) |
| base lane, 20 steps, instead of turbo | halves it inside the box (36x), round trip unchanged |
| scheduler leak | none: sigma starts at exactly 1.0, the box begins as pure noise |
| decode only the box crop | no change |
| the encoded latent itself | clean: Nyquist power under 2.5x in every channel, both axes |
| five-frame encode | not testable on the M5: the encode alone claims ~30 GB of MPS memory |

So the decoder lays the grid over any latent that carries encoded context, generated cells
included. The fix is pixel-space and now the default in `h3edit --inpaint`: the exact 16 px and
8 px row and column harmonics are zeroed in the render before the box is pasted (`--no-notch`
disables). Mean pixel change ~2 levels; inside-box harmonic 30–230x → 0–14x on every round-1 box,
untouched pixels bit-identical. Round 1 was notched retroactively per box; round 2 came out notched.

## Rules this build added

- **Onto a detailed facade: 0.9.** Signs and balcony life on shophouse fronts, four for four; the
  architecture stays, the thing appears. 0.85 is for props on bare ground.
- **Give an animal a surface.** "On the cart's roof edge" floated; "sitting on the flat steel lid" sat.
- **Vehicles before what they pass.** The taxi box wiped the stall painted before it. Order by
  depth, but when a foreground vehicle will cover a mid-ground stall, paint the vehicle first and
  compose the stall around it.
- **Lettering at 0.85 with an artwork card** works for street signs as it did for the shield:
  Thai and Latin exact.
- **Don't load anything else on the GPU while a build runs.** A diagnostic that loaded a second
  5 GB VAE pushed the next pass into MPS out-of-memory. The build resumed from state; the pass
  re-rendered.

## What is still soft

The sky and the upper facades outside any box are the 2x-upscaled 4 MP canvas, visibly softer
than the painted boxes at 1:1. A third round of 0.85–0.9 "re-detail" boxes over those areas is the
obvious next step and the remaining stress test: whether the whole 16 MP can be brought to box
sharpness pass by pass.
