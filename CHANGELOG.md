# Changelog

This file records the known GDC format history. Versions without surviving
source in this repository are marked explicitly instead of reconstructing
details that cannot be verified.

## Version 10 — QR carrier with GDC color data

- Renamed the implementation from `gdc_v4.py` to `gdc_v10.py`.
- Uses a standards-generated QR Version 10 matrix as the visual carrier.
- Preserves the carrier's exact light/dark module pattern after colorization.
- Keeps standard QR finder, separator, timing, alignment, format, version, and
  quiet-zone geometry.
- Keeps every light carrier module pure white and applies GDC color only to
  dark carrier modules, matching conventional colored QR styling.
- Encodes GDC values as four calibrated RGB levels per channel while keeping
  all colored modules below the dark/light threshold.
- Distributes dark calibration samples through ordinary dark carrier modules,
  so there is no exposed palette banner.
- Uses QR-style two-column zigzag data placement.
- Retains compression, CRC32, Reed-Solomon protection, and byte interleaving.
- Added exact QR-carrier, standard QR-scanner, and camera-distortion tests.

## Version 9 — Real-world field profile

- Reduced the logical grid to 73 × 73 modules.
- Increased rendered modules to 49 × 49 pixels.
- Reduced color quantization to four levels per RGB channel.
- Increased Reed-Solomon parity to 32 bytes per 255-byte block.
- Added four finder targets and stronger camera stress tests.

## Version 8 — QR-like large-module profile

- Reduced the logical grid to 121 × 121 modules.
- Increased rendered modules to 31 × 31 pixels.
- Added a four-module quiet zone.
- Prioritized visual module size over the earlier density target.

## Version 7 — Dense protected prototype

- Expanded the grid to 251 × 251 modules.
- Added optional zlib compression.
- Added Reed-Solomon error correction and byte interleaving.
- Added four-point perspective correction and larger calibration samples.
- Used 16 RGB levels per channel for a protected capacity above 72 KiB.

## Version 6 — Fixed-grid calibrated prototype

- Earliest surviving implementation in this repository.
- Used a 99 × 99 content grid and 16-pixel modules.
- Used eight levels per RGB channel, providing nine bits per data cell.
- Added three finder markers, an RGB reference palette, CRC32 validation,
  exact-image decoding, and experimental photo decoding.

## Version 5

- Historical version referenced by the project numbering.
- No source snapshot or verified change description is available.

## Version 4

- Historical version referenced by the original filename.
- No source snapshot or verified change description is available.

## Version 3

- Historical prototype; no source snapshot or verified change description is
  available.

## Version 2

- Historical prototype; no source snapshot or verified change description is
  available.

## Version 1 — Initial GDC concept

- Introduced the Gradient Dense Code concept.
- Proposed color-gradient encoding, camera calibration, and higher density
  than binary QR codes.
- Documented in the project technical paper dated March 8, 2025.
