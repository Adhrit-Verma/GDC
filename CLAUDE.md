# GDC - Gradient Dense Code

Goal (owner: Adhrit Verma): raise QR data density by giving every data module a
multi-level value (gray or RGB levels) instead of one black/white bit, while the
symbol still looks like an everyday QR code. NOT a hidden layer under a dummy
QR - that was v10, deliberately abandoned.

## Layout
- `gdc.py` - whole format: layout, stream, encode, decode, CLI (`encode|decode|capacity`).
- `test_gdc.py` - unittest suite (~3 s): exact round trips, density vs QR, camera sim, damage.
- `tools/benchmark.py` - print+phone robustness sweep (SER per profile/resolution).
- `tools/make_readme_assets.py` - regenerates `assets/readme/*.png` from the real encoder.
- `CHANGELOG.md` - format history v1..v11.

## Format v11 in one breath
QR geometry from `qrcode` internals (`function_pattern`), 5-bit profile
(mode, levels, ecc) BCH-coded in the QR format area. Payload -> optional zlib ->
header `>BHH` (version<<1|compressed, len, CRC-16) -> RS blocks (<=255, parity
share per ECC level mirrors QR) -> interleave -> XOR shake_256 keystream ->
Gray-coded levels in QR zigzag order. Decode: contour-hierarchy finders ->
version estimate -> affine + alignment-pattern homography -> sample centers ->
fit black/white lighting surfaces on function modules -> 1-D k-means per channel.

## Conventions
- Run: `.venv/Scripts/python.exe -m unittest test_gdc` (Windows venv).
- Keep QR function patterns pure black/white; data modules only change.
- Wire-format change => bump `VERSION`, update CHANGELOG + README.
- Ponytail style: minimal code, `# ponytail:` marks known ceilings.
- Commit + push after each tested milestone; don't rerun tests for doc-only edits.

## Measured envelope (tools/benchmark.py, QR version-10 size, harsh print sim)
4 gray: 0% SER down to 5.8 px/module. 8 gray: <=3%. 16 gray / RGB need >=8.5-12 px/module.
Open ideas: RS erasures from level confidence, webcam scanner, CIELAB for RGB, per-region warps.
