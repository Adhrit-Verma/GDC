"""Symbol-error and decode-success sweep under simulated print + phone capture.

Run from the repo root: python -m tools.benchmark
"""

import os

import cv2
import numpy as np

import gdc
from test_gdc import camera

PROFILES = [(10, 4, "gray", "M"), (10, 8, "gray", "M"), (10, 16, "gray", "L"),
            (10, 4, "rgb", "M"), (10, 8, "rgb", "L")]
WIDTHS = (380, 550, 800)  # capture width in px; version 10 spans 65 modules
NOISE = (4, 12)           # sensor noise sigma
RNG = np.random.default_rng(1)


def printed(image, width, noise):
    """Paper/ink range, dot-gain tone curve and warm light, then a phone capture."""
    rgb = np.asarray(image).astype(np.float32) / 255
    rgb = (0.12 + 0.7 * rgb) ** 1.6 * np.array([1.0, 0.93, 0.82])
    photo = camera(np.uint8(rgb * 255), width=width, blur=1.2, quality=70).astype(np.float32)
    return np.clip(photo + RNG.normal(0, noise, photo.shape), 0, 255).astype(np.uint8)


def trial(profile, width, noise):
    """(decoded?, symbol error rate) for one random full-capacity payload."""
    version, levels, mode, ecc = profile
    payload = os.urandom(gdc.capacity(*profile))
    rows, _ = gdc.data_cells(version)
    channels = gdc.MODES[mode]
    truth = gdc.stream_to_levels(gdc.build_stream(payload, *profile), rows.size * channels,
                                 gdc.bits_per_level(levels)).reshape(rows.size, channels)
    photo = printed(gdc.encode(payload, *profile), width, noise)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    try:
        grid = gdc.sample_modules(photo, gdc.rectify(gray, gdc.find_finders(gray), version), 17 + 4 * version)
        ser = float((gdc.read_levels(grid, version, gdc.read_meta(grid, version)) != truth).mean())
    except ValueError:
        return False, 1.0
    try:
        return gdc.decode(photo) == payload, ser
    except ValueError:
        return False, ser


def main():
    print("profile (version, levels, mode, ecc) | bytes | per capture width/noise: decoded, symbol error rate")
    for profile in PROFILES:
        cells = []
        for width in WIDTHS:
            for noise in NOISE:
                ok, ser = trial(profile, width, noise)
                cells.append(f"{width}/{noise}: {'OK ' if ok else 'ERR'} {ser:.3f}")
        print(f"{profile} | {gdc.capacity(*profile)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
