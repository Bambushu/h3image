# h3edit benchmark

## 1. Lettering ladder (seed 1001)

Character accuracy per tier (1.0 = every character right). Cap height in output px in brackets.

| MP | output | L | M | S | XS | XXS | wall s |
|---|---|---|---|---|---|---|---|
| 1 | 1376x768 | 1.00 (116px) | 1.00 (58px) | 1.00 (31px) | 1.00 (18px) | 0.85 (12px) | 139 |
| 2 | 1920x1088 | 1.00 (166px) | 1.00 (83px) | 1.00 (45px) | 1.00 (26px) | 0.93 (17px) | 132 |
| 4 | 2720x1536 | 1.00 (231px) | 1.00 (116px) | 1.00 (63px) | 0.96 (37px) | 1.00 (23px) | 144 |

## 2. Seed stability (4 MP, 8 seeds)

| seed | L | M | S | XS | XXS | S+XS+XXS pass (≥0.9) |
|---|---|---|---|---|---|---|
| 1001 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 3/3 |
| 1002 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1003 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1004 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1005 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1006 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1007 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| 1008 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3/3 |
| **mean** | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | |
| **worst** | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | |

## 3. Detail-pass rescue (`--detail` on sign_mp4_s1001, the weakest seed)

| pass | L | M | S | XS | XXS | seam SSIM |
|---|---|---|---|---|---|---|
| single pass | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | |
| + --detail (2 MP) | 1.00 | 0.46 | 0.40 | 0.58 | 1.00 | 0.930 |
| + --detail --mp 4 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 0.929 |
| + --detail --mp 4, box incl. plate edge | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 0.987 |

## 4. Preservation outside the edit (what survives whole-frame regeneration)

Outside-plate SSIM vs source, and fiducial patches re-found by NCC (≥0.6 = kept).

| render | SSIM outside | kept /8 | window_top | window_sill | coping_right | coping_left | plinth_left | pavement | sky | wall_right |
|---|---|---|---|---|---|---|---|---|---|---|
| sign_mp1_s1001 | 0.44 | 7 | 0.94 | 0.89 | 0.90 | 0.97 | 0.43 | 0.74 | 0.95 | 0.73 |
| sign_mp2_s1001 | 0.58 | 7 | 0.96 | 0.92 | 0.79 | 0.98 | 0.74 | 0.75 | 0.96 | 0.38 |
| sign_mp4_s1001 | 0.58 | 8 | 0.97 | 0.94 | 0.90 | 0.98 | 0.73 | 0.74 | 0.96 | 0.69 |
| sign_mp4_s1002 | 0.50 | 8 | 0.96 | 0.94 | 0.91 | 0.98 | 0.73 | 0.66 | 0.96 | 0.72 |
| sign_mp4_s1003 | 0.50 | 8 | 0.97 | 0.95 | 0.91 | 0.98 | 0.77 | 0.75 | 0.96 | 0.71 |
| sign_mp4_s1004 | 0.51 | 8 | 0.97 | 0.95 | 0.89 | 0.98 | 0.75 | 0.74 | 0.95 | 0.60 |
| sign_mp4_s1005 | 0.55 | 7 | 0.97 | 0.95 | 0.86 | 0.98 | 0.74 | 0.76 | 0.96 | 0.49 |
| sign_mp4_s1006 | 0.57 | 7 | 0.96 | 0.95 | 0.88 | 0.97 | 0.75 | 0.73 | 0.95 | 0.51 |
| sign_mp4_s1007 | 0.50 | 7 | 0.95 | 0.96 | 0.86 | 0.97 | 0.77 | 0.75 | 0.97 | 0.51 |
| sign_mp4_s1008 | 0.51 | 8 | 0.97 | 0.95 | 0.91 | 0.98 | 0.75 | 0.68 | 0.95 | 0.69 |

## 5. Light as a source (neon lit vs switched-off, same seed)

Mean Lab shift of the lit render relative to the unlit one. `toward_pink` > 0 = region moved toward the sign's colour; `far_control` should stay ~0.

| region | ΔE00 | toward pink | ΔL |
|---|---|---|---|
| brick_left | 0.6 | +0.2 | -0.7 |
| brick_right | 0.7 | +0.6 | -0.5 |
| pavement | 20.8 | +18.2 | +10.9 |
| far_control | 0.2 | +0.0 | -0.3 |

## Speed

| render | MP | size | wall s |
|---|---|---|---|
| sign_mp1_s1001 | 1.0 | 1376x768 | 139 |
| sign_mp2_s1001 | 2.0 | 1920x1088 | 132 |
| sign_mp4_s1001 | 4.0 | 2720x1536 | 144 |
| sign_mp4_s1002 | 4.0 | 2720x1536 | 149 |
| sign_mp4_s1003 | 4.0 | 2720x1536 | 154 |
| sign_mp4_s1004 | 4.0 | 2720x1536 | 155 |
| sign_mp4_s1005 | 4.0 | 2720x1536 | 144 |
| sign_mp4_s1006 | 4.0 | 2720x1536 | 144 |
| sign_mp4_s1007 | 4.0 | 2720x1536 | 145 |
| sign_mp4_s1008 | 4.0 | 2720x1536 | 150 |
| neon_lit | 4.0 | 2720x1536 | 145 |
| neon_unlit | 4.0 | 2720x1536 | 146 |
| sign_mp4_s1001_detail | None | 2720x1536 | 136 |
| sign_mp4_s1001_detail4 | 4.0 | 2720x1536 | 142 |
| sign_mp4_s1001_detail4edge | 4.0 | 2720x1536 | 138 |
