"""Generate README animations from the real GDC encoder and decoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gdc_v10 as gdc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "readme"
REFERENCE = ROOT / "references" / "qr_reference.png"
CANVAS_SIZE = 760
HEADER_HEIGHT = 88
FRAME_MS = 950


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE_FONT = font(34, bold=True)
BODY_FONT = font(22)
SMALL_FONT = font(18)


def contain(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    return result


def frame(title: str, subtitle: str, image: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CANVAS_SIZE, HEADER_HEIGHT), fill="#111827")
    draw.text((24, 15), title, font=TITLE_FONT, fill="white")
    draw.text((25, 55), subtitle, font=SMALL_FONT, fill="#cbd5e1")

    visual = contain(image, CANVAS_SIZE - 56, CANVAS_SIZE - HEADER_HEIGHT - 48)
    x = (CANVAS_SIZE - visual.width) // 2
    y = HEADER_HEIGHT + (CANVAS_SIZE - HEADER_HEIGHT - visual.height) // 2
    canvas.paste(visual, (x, y))
    return canvas


def message_frame(title: str, lines: list[str], accent: str = "#2563eb") -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, CANVAS_SIZE, HEADER_HEIGHT), fill="#111827")
    draw.text((24, 15), title, font=TITLE_FONT, fill="white")
    draw.rounded_rectangle(
        (60, 150, CANVAS_SIZE - 60, CANVAS_SIZE - 110),
        radius=28,
        fill="white",
        outline=accent,
        width=5,
    )
    y = 210
    for index, line in enumerate(lines):
        draw.text(
            (100, y),
            line,
            font=TITLE_FONT if index == 0 else BODY_FONT,
            fill=accent if index == 0 else "#1f2937",
        )
        y += 68 if index == 0 else 48
    return canvas


def save_gif(frames: list[Image.Image], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed = [
        item.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        for item in frames
    ]
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )


def threshold_carrier(image: Image.Image) -> Image.Image:
    logical = gdc.sample_grid_colors(
        image,
        gdc.fixed_side_blocks(),
        gdc.fixed_side_blocks(),
    )
    start = gdc.QUIET_ZONE_BLOCKS
    core = logical.crop(
        (
            start,
            start,
            start + gdc.CONTENT_SIDE,
            start + gdc.CONTENT_SIDE,
        )
    )
    array = np.asarray(core.convert("RGB"), dtype=np.uint8)
    dark = array.mean(axis=2) < 128
    binary = np.where(dark[:, :, None], 0, 255).astype(np.uint8)
    binary = np.repeat(binary, 3, axis=2)
    return Image.fromarray(binary).resize(
        (gdc.fixed_core_size_px(), gdc.fixed_core_size_px()),
        Image.Resampling.NEAREST,
    )


def make_qr_to_gdc():
    payload = (
        b"Gradient Dense Code: QR geometry with calibrated color data, "
        b"compression, CRC32, and Reed-Solomon protection."
    )
    colored, *_ = gdc.stream_to_image(gdc.build_stream(payload))

    carrier = Image.new(
        "RGB",
        (gdc.CONTENT_SIDE, gdc.CONTENT_SIDE),
        "white",
    )
    pixels = carrier.load()
    for row, matrix_row in enumerate(gdc.qr_carrier_matrix()):
        for col, dark in enumerate(matrix_row):
            pixels[col, row] = (0, 0, 0) if dark else (255, 255, 255)
    carrier = carrier.resize(
        (gdc.fixed_core_size_px(), gdc.fixed_core_size_px()),
        Image.Resampling.NEAREST,
    )

    frames = [
        frame(
            "1. Standard QR reference",
            "The familiar visual language: white background and three finders",
            Image.open(REFERENCE),
        ),
        frame(
            "2. Standards-generated carrier",
            "GDC starts with a real QR Version 10 module matrix",
            carrier,
        ),
        frame(
            "3. GDC color payload",
            "Only dark QR modules receive calibrated RGB values",
            colored,
        ),
        frame(
            "4. Threshold proof",
            "Converted back to black and white, the exact QR carrier remains",
            threshold_carrier(colored),
        ),
        message_frame(
            "Two compatible layers",
            [
                "QR layer",
                "Standard scanners read: GDC-V10-COLOR-CARRIER",
                "GDC layer",
                "The custom decoder recovers the color payload.",
            ],
            accent="#0f766e",
        ),
    ]
    save_gif(frames, OUTPUT / "qr-to-gdc.gif")


def perspective_photo(image: Image.Image) -> np.ndarray:
    source = np.float32(
        [
            [0, 0],
            [image.width - 1, 0],
            [image.width - 1, image.height - 1],
            [0, image.height - 1],
        ]
    )
    size = image.width
    destination = np.float32(
        [
            [size * 0.08, size * 0.06],
            [size * 0.92, size * 0.02],
            [size * 0.97, size * 0.93],
            [size * 0.04, size * 0.98],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.warpPerspective(
        bgr,
        transform,
        (size, size),
        borderValue=(225, 225, 225),
    )


def make_camera_recovery():
    payload = b"GDC camera recovery demo: perspective corrected successfully."
    encoded, *_ = gdc.stream_to_image(gdc.build_stream(payload))
    photo = perspective_photo(encoded)

    with tempfile.TemporaryDirectory() as directory:
        photo_path = Path(directory) / "camera.jpg"
        cv2.imwrite(str(photo_path), photo, [cv2.IMWRITE_JPEG_QUALITY, 82])
        recovered_core = gdc.warp_photo_to_core(photo_path)
        decoded = gdc.photo_to_payload(photo_path)

    photo_rgb = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    recovered_large = recovered_core.resize(
        (gdc.fixed_core_size_px(), gdc.fixed_core_size_px()),
        Image.Resampling.NEAREST,
    )
    decoded_text = decoded.decode("utf-8")

    frames = [
        frame(
            "1. Encoded symbol",
            "A GDC payload inside an exact QR visual carrier",
            encoded,
        ),
        frame(
            "2. Simulated camera capture",
            "Perspective, resampling, and JPEG compression are introduced",
            photo_rgb,
        ),
        frame(
            "3. Geometric recovery",
            "Finder and alignment patterns restore the module grid",
            recovered_large,
        ),
        message_frame(
            "4. Payload recovered",
            [
                "Decode successful",
                decoded_text,
                f"Protected capacity: {gdc.max_payload_bytes()} bytes",
                "CRC32 and Reed-Solomon checks passed.",
            ],
            accent="#7c3aed",
        ),
    ]
    save_gif(frames, OUTPUT / "camera-recovery.gif")


def main():
    make_qr_to_gdc()
    make_camera_recovery()
    print(f"Generated README assets in {OUTPUT}")


if __name__ == "__main__":
    main()
