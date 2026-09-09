#!/usr/bin/env python3
"""Generate the benchmark inputs deterministically. Everything is PIL-drawn or a shipped demo ref;
the generated PNGs are committed too, so a reproducer never needs these fonts.

  sign_plate.png   five-tier lettering plate (the lettering ladder / digit-stability input)
  neon_unlit.png   the neon demo sign with its tubes switched off (light-as-source control)
  sign.txt / sign_detail.txt / neon_unlit.txt   the prompts
"""
import json, os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
DEMOS = os.path.join(HERE, "..", "demos", "refs")
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]

# Tier text, cap height as a fraction of plate height. Ratios are what matter: the plate is
# prompted to fill the wall box SIGN_BOX of the 1280x720 scene, so cap height in OUTPUT pixels is
# ratio * box_height * sqrt(mp / 0.92). Mixed letters/digits/punctuation on purpose.
TIERS = [("L",   "HARBOUR",                          0.24),
         ("M",   "KXT-4821",                         0.12),
         ("S",   "UNIT 7 - 0141 496 0721",           0.065),
         ("XS",  "EST 1962 - NO PARKING - REF 88-3105", 0.038),
         ("XXS", "WWW.HARBOURWORKS.EXAMPLE - OPEN 08-18", 0.024)]
SIGN_BOX = (130, 110, 1050, 640)      # where the plate lands in gable_scene-s77 (x0,y0,x1,y1)
PLATE = (1280, 736)                    # plate ref size; aspect ~= SIGN_BOX aspect
BG, FG = (28, 36, 58), (245, 240, 225)


def font(px):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    raise SystemExit("no TTF font found; install fonts-dejavu or use the committed PNGs")


def cap_font(cap_px):
    """Font whose capital letters are cap_px tall."""
    f = font(max(8, int(cap_px * 1.4)))
    h = f.getbbox("H")[3] - f.getbbox("H")[1]
    return font(max(8, int(cap_px * 1.4 * cap_px / h)))


def sign_plate():
    W, H = PLATE
    im = Image.new("RGB", PLATE, BG)
    d = ImageDraw.Draw(im)
    d.rectangle((14, 14, W - 15, H - 15), outline=FG, width=6)
    y = 0.07 * H
    layout = []
    for name, text, ratio in TIERS:
        f = cap_font(ratio * H)
        bb = d.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x = (W - tw) / 2
        d.text((x - bb[0], y - bb[1]), text, font=f, fill=FG)
        layout.append({"tier": name, "text": text, "cap_ratio": ratio,
                       "band": [round(y / H, 4), round((y + th) / H, 4)]})
        y += th * 1.55
    assert y < H, f"plate overflow: {y} > {H}"
    im.save(os.path.join(INP, "sign_plate.png"))
    json.dump({"tiers": layout, "sign_box": SIGN_BOX, "plate": PLATE, "scene": "gable_scene-s77.png"},
              open(os.path.join(INP, "sign_layout.json"), "w"), indent=1)


def neon_unlit():
    im = Image.open(os.path.join(DEMOS, "neon_sign-s42.png")).convert("RGB")
    # kill the glow: desaturate, darken, and flatten the halo so the tubes read as off glass
    im = ImageEnhance.Color(im).enhance(0.15)
    im = ImageEnhance.Brightness(im).enhance(0.45)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    im.save(os.path.join(INP, "neon_unlit.png"))


PROMPTS = {
"sign.txt": """Task: Reference-guided generation.

<Picture 1> supplies the entire scene: the building, the brickwork, the window, the pavement, the sky, the lighting and the photograph's look. <Picture 2> is a painted sign plate reference and supplies nothing else. It does not supply a location, a background, a sky, or lighting.

Mount <Picture 2> onto the bare brick wall in <Picture 1> as exactly one flat rectangular painted metal sign plate, filling the brickwork between the stone plinth course at the base and the roof coping above, from near the left edge of the wall to the window reveal, so the plate is as large as the wall allows. The plate is a rigid flat panel screwed to the wall and lit by the same low sun, with a thin shadow along its lower and right edges. Copy every line of lettering from <Picture 2> exactly, letter for letter and digit for digit, in the same layout, the same typeface and the same sizes relative to the plate: the large word, the code beneath it, and the three smaller lines under that. Do not add, remove, reorder or respell any characters. No other text anywhere.

Preserve from <Picture 1>: the camera position and framing; the wall's geometry, its corner edge and roof line; the tall window, its dark frame, its deep reveal and its stone sill; the stone plinth course and the pavement below; brick colour, mortar tone and surface wear around the plate; the sky; the direction and warmth of the sunlight and all cast shadows; the exposure and photographic grain.

A single coherent photograph shows the same brick wall on the same street, with the <Picture 2> sign plate mounted on the brickwork.
""",
"sign_detail.txt": """Task: Reference-guided generation.

<Picture 1> is a tight crop from a photograph of a painted metal sign plate mounted on a red-brick wall. The crop shows only the middle of the plate: a dark navy painted surface with four lines of pale lettering, and the plate's edges, border and the brick wall all lie OUTSIDE the frame. <Picture 1> supplies the entire image: the framing, the navy paint, the position and size of every line of lettering, the colours and the sunlight. <Picture 2> is the flat artwork of the same sign plate and is the authority for the lettering only: copy the four lines exactly as they appear there, letter for letter and digit for digit.

Reproduce <Picture 1> exactly, at higher sharpness: the same tight crop, the same navy surface filling the whole frame edge to edge, the same four lines of lettering in the same positions and sizes, with every character corrected to match <Picture 2>. Do not draw a plate border, a plate edge, screws, brick, or a wall. Do not add, move, resize or remove any line. No other text anywhere.

A single coherent photograph identical to <Picture 1> in composition: navy painted metal filling the frame, four lines of lettering, nothing else.
""",
"neon_unlit.txt": """Task: Reference-guided generation.

<Picture 1> supplies the entire scene: the brick shopfront, the doorway, the plate-glass window, the bare wooden signboard, the wet pavement, the dusk sky, the existing warm interior light and the photograph's look. <Picture 2> is the sign reference and supplies nothing else. It does not supply a scene, a building, a background, or a dark surround.

Mount <Picture 2> onto the bare wooden signboard in <Picture 1> as exactly one neon sign that is switched off, sized to sit inside the board with a margin of bare timber around it. The sign is unpowered: its glass tubing is dark and unlit, it emits no light, it casts no glow onto the brickwork, adds no coloured spill to the doorway, and lays no coloured reflection into the wet pavement. The only light in the scene is the existing warm interior light and the dusk sky.

Preserve from <Picture 1>: the camera position and framing; the brickwork, the doorway and its dark frame, the plate-glass window and the warm interior visible through it; the shape and position of the wooden signboard; the wet pavement and its existing warm reflection; the dusk sky and its colour; the exposure, white balance and photographic grain.

A single coherent photograph shows the same shopfront at dusk, with the <Picture 2> sign mounted above its door and switched off.
""",
}

if __name__ == "__main__":
    os.makedirs(INP, exist_ok=True)
    sign_plate()
    neon_unlit()
    for n, t in PROMPTS.items():
        open(os.path.join(INP, n), "w").write(t)
    print("inputs written to", INP)
