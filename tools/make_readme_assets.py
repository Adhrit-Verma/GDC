"""Regenerate README images from the real encoder: python -m tools.make_readme_assets"""

import random
from pathlib import Path

import qrcode
from PIL import Image, ImageDraw, ImageFont

import gdc
from test_gdc import camera

OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "readme"
TILE = 360
fill = random.Random(0).randbytes  # incompressible, reproducible


def label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    try:
        big, small = ImageFont.truetype("arialbd.ttf", 22), ImageFont.truetype("arial.ttf", 17)
    except OSError:
        big = small = ImageFont.load_default()
    card = Image.new("RGB", (TILE, TILE + 70), "white")
    card.paste(image.convert("RGB").resize((TILE, TILE), Image.Resampling.NEAREST))
    draw = ImageDraw.Draw(card)
    draw.text((TILE // 2, TILE + 18), title, font=big, fill="#111827", anchor="mm")
    draw.text((TILE // 2, TILE + 48), subtitle, font=small, fill="#4b5563", anchor="mm")
    return card


def strip(cards, path: Path):
    out = Image.new("RGB", (len(cards) * (TILE + 20) + 20, TILE + 90), "white")
    for i, card in enumerate(cards):
        out.paste(card, (20 + i * (TILE + 20), 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, optimize=True)


def make_comparison():
    """Same 57x57 version-10 footprint, M error correction, full capacity each."""
    qr = qrcode.QRCode(version=10, error_correction=gdc.QR_ECC["M"], border=gdc.QUIET)
    qr.add_data(fill(gdc.qr_capacity(10)), optimize=0)
    qr.make(fit=False)
    cards = [label(qr.make_image().get_image(), "Standard QR v10-M", f"{gdc.qr_capacity(10)} bytes")]
    for levels in (4, 8):
        payload = fill(gdc.capacity(10, levels))
        cards.append(label(gdc.encode(payload, 10, levels, module_px=4),
                           f"GDC v11, {levels} gray levels", f"{gdc.capacity(10, levels)} bytes"))
    payload = fill(gdc.capacity(10, 4, "rgb"))
    cards.append(label(gdc.encode(payload, 10, 4, "rgb", module_px=4), "GDC v11, RGB 4 levels",
                       f"{gdc.capacity(10, 4, 'rgb')} bytes"))
    strip(cards, OUTPUT / "qr-vs-gdc.png")


def make_camera_recovery():
    payload = b"GDC v11 camera recovery: perspective, blur, shading and JPEG corrected."
    encoded = gdc.encode(payload, version=6)
    photo = camera(encoded, width=700, rotate=1)
    gray = gdc.cv2.cvtColor(photo, gdc.cv2.COLOR_BGR2GRAY)
    grid = gdc.sample_modules(photo, gdc.rectify(gray, gdc.find_finders(gray), 6), 41)
    assert gdc.decode(photo) == payload
    strip([
        label(encoded, "1. Encoded", "41x41 modules, 4 gray levels"),
        label(Image.fromarray(photo[..., ::-1]), "2. Simulated capture", "rotation, perspective, JPEG"),
        label(Image.fromarray(grid.clip(0, 255).astype("uint8")), "3. Rectified samples", "decoded + CRC verified"),
    ], OUTPUT / "camera-recovery.png")


if __name__ == "__main__":
    make_comparison()
    make_camera_recovery()
    print(f"Wrote README assets to {OUTPUT}")
