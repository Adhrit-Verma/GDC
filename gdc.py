"""GDC v11 - Gradient Dense Code: a QR code whose data modules carry
multi-level values instead of single black/white bits.

The geometry is standard QR (finders, separators, timing, alignment, format and
version areas, quiet zone), built by the `qrcode` package, so the symbol looks
like an everyday QR code. Every data module holds one of `levels` intensities
per channel: gray mode uses 1 channel, rgb mode 3. Four gray levels store
2 bits per module, roughly twice a same-size binary QR code.
"""

import argparse
import binascii
import hashlib
import itertools
import struct
import zlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import qrcode
from PIL import Image
from qrcode.base import rs_blocks
from qrcode.util import pattern_position
from reedsolo import RSCodec, ReedSolomonError

VERSION = 11
QUIET = 4                       # quiet zone, modules
MODES = {"gray": 1, "rgb": 3}   # channels per module
LEVELS = (2, 4, 8, 16)          # intensity levels per channel
# Parity share of every RS block, mirroring QR's EC codeword share per level.
ECC = {"L": 0.20, "M": 0.38, "Q": 0.55, "H": 0.65}
QR_ECC = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}
HEADER = struct.Struct(">BHH")  # format version << 1 | compressed, stored length, CRC-16
SAMPLE_PX = 8                   # pixels per module in the rectified image


# ---------------------------------------------------------------------------
# Symbol layout
# ---------------------------------------------------------------------------

def pack_meta(levels: int, mode: str, ecc: str) -> int:
    """5-bit profile id, stored in the QR format-information area."""
    return (list(MODES).index(mode) << 4) | (LEVELS.index(levels) << 2) | list(ECC).index(ecc)


def unpack_meta(meta: int) -> tuple[int, str, str]:
    return LEVELS[(meta >> 2) & 3], list(MODES)[meta >> 4], list(ECC)[meta & 3]


@lru_cache(maxsize=None)
def function_pattern(version: int, meta: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(is_function, is_dark) matrices of the standard QR function patterns.

    The 15-bit format area carries `meta` with QR's own BCH code, split into
    the 2-bit "error correction" and 3-bit "mask" fields qrcode writes there.
    """
    qr = qrcode.QRCode(version=version, error_correction=meta >> 3)
    n = qr.modules_count = 17 + 4 * version
    qr.modules = [[None] * n for _ in range(n)]
    for row, col in ((0, 0), (n - 7, 0), (0, n - 7)):
        qr.setup_position_probe_pattern(row, col)
    qr.setup_position_adjust_pattern()
    qr.setup_timing_pattern()
    qr.setup_type_info(False, meta & 7)
    if version >= 7:
        qr.setup_type_number(False)
    is_function = np.array([[m is not None for m in row] for row in qr.modules])
    is_dark = np.array([[bool(m) for m in row] for row in qr.modules])
    return is_function, is_dark


@lru_cache(maxsize=None)
def data_cells(version: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and cols of the data modules in QR's two-column zigzag order."""
    is_function, _ = function_pattern(version)
    n = len(is_function)
    rows, cols = [], []
    right, upward = n - 1, True
    while right > 0:
        if right == 6:
            right -= 1
        for row in range(n - 1, -1, -1) if upward else range(n):
            for col in (right, right - 1):
                if not is_function[row, col]:
                    rows.append(row)
                    cols.append(col)
        upward = not upward
        right -= 2
    return np.array(rows), np.array(cols)


def alignment_centers(version: int) -> list[tuple[float, float]]:
    """Alignment-pattern centers as (x, y) module coordinates, nearest the finders first."""
    pos = pattern_position(version)
    finder_corners = {(pos[0], pos[0]), (pos[0], pos[-1]), (pos[-1], pos[0])} if pos else set()
    centers = [(c + 0.5, r + 0.5) for r in pos for c in pos if (r, c) not in finder_corners]
    return sorted(centers, key=lambda p: p[0] + p[1])


def bits_per_level(levels: int) -> int:
    return levels.bit_length() - 1


def rs_layout(version: int, levels: int, mode: str, ecc: str) -> tuple[int, int, int]:
    """(block count, block size, parity bytes per block) filling the symbol."""
    raw = len(data_cells(version)[0]) * MODES[mode] * bits_per_level(levels) // 8
    blocks = -(-raw // 255)
    size = raw // blocks
    parity = max(2, 2 * round(size * ECC[ecc] / 2))
    return blocks, size, parity


def capacity(version: int, levels: int = 4, mode: str = "gray", ecc: str = "M") -> int:
    """Largest incompressible payload in bytes."""
    blocks, size, parity = rs_layout(version, levels, mode, ecc)
    return blocks * (size - parity) - HEADER.size


def qr_capacity(version: int, ecc: str = "M") -> int:
    """Binary-mode byte capacity of a standard QR code of the same version."""
    data_bytes = sum(block.data_count for block in rs_blocks(version, QR_ECC[ecc]))
    return (data_bytes * 8 - 4 - (8 if version < 10 else 16)) // 8


# ---------------------------------------------------------------------------
# Byte stream: compression, header, Reed-Solomon, interleaving, whitening
# ---------------------------------------------------------------------------

def keystream(size: int) -> np.ndarray:
    # Whitening spreads levels evenly (needed for level clustering) and breaks
    # up long runs, the job QR's mask patterns do for binary modules.
    return np.frombuffer(hashlib.shake_256(b"GDC-v11").digest(size), np.uint8)


@lru_cache(maxsize=None)
def rs_codec(parity: int) -> RSCodec:
    return RSCodec(parity)


def stored_payload(payload: bytes) -> tuple[int, bytes]:
    compressed = zlib.compress(payload, 9)
    return (1, compressed) if len(compressed) < len(payload) else (0, payload)


def build_stream(payload: bytes, version: int, levels: int, mode: str, ecc: str) -> np.ndarray:
    blocks, size, parity = rs_layout(version, levels, mode, ecc)
    flags, stored = stored_payload(payload)
    limit = capacity(version, levels, mode, ecc)
    if len(stored) > limit:
        raise ValueError(f"Payload needs {len(stored)} bytes; version {version} holds {limit}.")
    k = size - parity
    data = HEADER.pack(VERSION << 1 | flags, len(stored), binascii.crc_hqx(payload, 0)) + stored
    data = data.ljust(blocks * k, b"\0")
    coded = np.array(
        [list(rs_codec(parity).encode(data[i * k:(i + 1) * k])) for i in range(blocks)],
        np.uint8,
    )
    stream = coded.T.reshape(-1)  # interleave: byte i of every block, then i + 1
    return stream ^ keystream(stream.size)


def parse_stream(stream: np.ndarray, version: int, levels: int, mode: str, ecc: str) -> bytes:
    blocks, size, parity = rs_layout(version, levels, mode, ecc)
    coded = (stream[:blocks * size] ^ keystream(blocks * size)).reshape(size, blocks).T
    data = b""
    for block in coded:
        try:
            data += bytes(rs_codec(parity).decode(bytearray(block))[0])
        except ReedSolomonError as exc:
            raise ValueError("Too much damage for Reed-Solomon correction.") from exc
    flags, length, crc = HEADER.unpack_from(data)
    if flags >> 1 != VERSION:
        raise ValueError(f"Not a GDC v{VERSION} symbol.")
    stored = data[HEADER.size:HEADER.size + length]
    try:
        payload = zlib.decompress(stored) if flags & 1 else stored
    except zlib.error as exc:
        raise ValueError("Compressed payload is damaged.") from exc
    if len(stored) != length or binascii.crc_hqx(payload, 0) != crc:
        raise ValueError("CRC mismatch after error correction.")
    return payload


# ---------------------------------------------------------------------------
# Bits <-> levels (Gray-coded so a one-level misread flips one bit)
# ---------------------------------------------------------------------------

def gray_code(bits: int) -> np.ndarray:
    index = np.arange(1 << bits)
    return index ^ (index >> 1)


def stream_to_levels(stream: np.ndarray, count: int, bits: int) -> np.ndarray:
    total_bits = count * bits
    bit_array = np.unpackbits(keystream(-(-total_bits // 8)))[:total_bits]  # filler past the stream
    bit_array[:stream.size * 8] = np.unpackbits(stream)
    values = bit_array.reshape(count, bits) @ (1 << np.arange(bits)[::-1])
    return np.argsort(gray_code(bits))[values]


def levels_to_stream(levels_index: np.ndarray, bits: int, byte_count: int) -> np.ndarray:
    values = gray_code(bits)[levels_index]
    bit_array = ((values[:, None] >> np.arange(bits)[::-1]) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bit_array[:byte_count * 8])


def level_values(levels: int) -> np.ndarray:
    return np.round(np.linspace(0, 255, levels)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def smallest_version(payload: bytes, levels: int, mode: str, ecc: str) -> int:
    need = len(stored_payload(payload)[1])
    for version in range(1, 41):
        if capacity(version, levels, mode, ecc) >= need:
            return version
    raise ValueError(f"Payload needs {need} bytes; version 40 holds {capacity(40, levels, mode, ecc)}.")


def encode(
    payload: bytes,
    version: int | None = None,
    levels: int = 4,
    mode: str = "gray",
    ecc: str = "M",
    module_px: int = 16,
) -> Image.Image:
    """Render `payload` as a multilevel QR image. version=None picks the smallest fit."""
    version = version or smallest_version(payload, levels, mode, ecc)
    stream = build_stream(payload, version, levels, mode, ecc)
    _, is_dark = function_pattern(version, pack_meta(levels, mode, ecc))
    matrix = np.repeat(np.where(is_dark, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    rows, cols = data_cells(version)
    channels = MODES[mode]
    index = stream_to_levels(stream, rows.size * channels, bits_per_level(levels))
    matrix[rows, cols] = level_values(levels)[index.reshape(rows.size, channels)]
    matrix = np.pad(matrix, ((QUIET, QUIET), (QUIET, QUIET), (0, 0)), constant_values=255)
    return Image.fromarray(matrix.repeat(module_px, 0).repeat(module_px, 1))


# ---------------------------------------------------------------------------
# Decoding: locate, rectify, sample, normalize, quantize
# ---------------------------------------------------------------------------

def load_bgr(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image {image}.")
        return bgr
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def find_finders(gray: np.ndarray):
    """[(center_xy, module_px)] for the top-left, top-right and bottom-left finders."""
    scale = min(1.0, 1600 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    # A finder is a dark ring holding a dark core: contour -> hole -> core.
    for outer, (_, _, hole, _) in zip(contours, hierarchy[0] if hierarchy is not None else []):
        core = hierarchy[0][hole][2] if hole >= 0 else -1
        if core < 0:
            continue
        outer_area, core_area = cv2.contourArea(outer), cv2.contourArea(contours[core])
        _, (w, h), _ = cv2.minAreaRect(outer)
        if core_area <= 0 or not 3 < outer_area / core_area < 12 or min(w, h) < 7 or max(w, h) > 1.6 * min(w, h):
            continue
        m = cv2.moments(contours[core])
        center = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) / scale
        found.append((center, np.sqrt(outer_area) / 7 / scale))

    best = None
    for trio in itertools.combinations(sorted(found, key=lambda f: -f[1])[:12], 3):
        sizes = [f[1] for f in trio]
        for corner in range(3):
            tl = trio[corner]
            a, b = (trio[i] for i in range(3) if i != corner)
            va, vb = a[0] - tl[0], b[0] - tl[0]
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            score = abs(va @ vb) / (la * lb) + abs(la - lb) / max(la, lb) + (max(sizes) - min(sizes)) / max(sizes)
            if best is None or score < best[0]:
                if va[0] * vb[1] - va[1] * vb[0] < 0:  # make a the top-right finder
                    a, b = b, a
                best = (score, (tl, a, b))
    if best is None or best[0] > 0.6:
        raise ValueError("Could not find three QR finder patterns.")
    return best[1]


def candidate_versions(finders) -> list[int]:
    tl, tr, bl = finders
    module = np.mean([f[1] for f in finders])
    side = (np.linalg.norm(tr[0] - tl[0]) + np.linalg.norm(bl[0] - tl[0])) / 2 / module + 7
    estimate = (side - 17) / 4
    return sorted(range(1, 41), key=lambda v: abs(v - estimate))[:5]


@lru_cache(maxsize=1)
def alignment_template() -> np.ndarray:
    ring = np.full((5, 5), 255, np.uint8)
    ring[[0, -1], :] = ring[:, [0, -1]] = ring[2, 2] = 0
    return ring.repeat(SAMPLE_PX, 0).repeat(SAMPLE_PX, 1)


def locate_alignment(gray: np.ndarray, H: np.ndarray, x: float, y: float, radius: float):
    """Image point of the alignment pattern expected near module coords (x, y)."""
    half = radius + 2.5
    size = round(2 * half * SAMPLE_PX)
    to_patch = np.array([[SAMPLE_PX, 0, -(x - half) * SAMPLE_PX], [0, SAMPLE_PX, -(y - half) * SAMPLE_PX], [0, 0, 1]])
    patch = cv2.warpPerspective(gray, to_patch @ H, (size, size), flags=cv2.INTER_LINEAR)
    scores = cv2.matchTemplate(patch, alignment_template(), cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(scores)
    if score < 0.45:
        return None
    found = np.array([[[x - half + bx / SAMPLE_PX + 2.5, y - half + by / SAMPLE_PX + 2.5]]])
    return cv2.perspectiveTransform(found, np.linalg.inv(H))[0, 0]


def rectify(gray: np.ndarray, finders, version: int) -> np.ndarray:
    """Homography from image pixels to module coordinates."""
    n = 17 + 4 * version
    src = [f[0] for f in finders]
    dst = [(3.5, 3.5), (n - 3.5, 3.5), (3.5, n - 3.5)]
    H = np.vstack([cv2.getAffineTransform(np.float32(src), np.float32(dst)), [0, 0, 1]])
    # ponytail: one global homography; per-region warps if lens distortion breaks large versions.
    # Walk outward from the finders so each search extrapolates the fit a short way.
    centers = alignment_centers(version)
    for x, y in centers:
        point = locate_alignment(gray, H, x, y, radius=4 if len(centers) == 1 else 2)
        if point is None:
            continue
        src.append(point)
        dst.append((x, y))
        if len(src) >= 4:
            method = cv2.RANSAC if len(src) >= 5 else 0
            found, _ = cv2.findHomography(np.float32(src), np.float32(dst), method, 0.7)
            H = found if found is not None else H
    return H


def sample_modules(bgr: np.ndarray, H: np.ndarray, n: int, margin: float = 0.3) -> np.ndarray:
    """Mean RGB of every module center, quiet zone included: (n + 2Q, n + 2Q, 3)."""
    side = n + 2 * QUIET
    to_grid = np.array([[SAMPLE_PX, 0, QUIET * SAMPLE_PX], [0, SAMPLE_PX, QUIET * SAMPLE_PX], [0, 0, 1]]) @ H
    warped = cv2.warpPerspective(bgr, to_grid, (side * SAMPLE_PX,) * 2, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    a, b = int(SAMPLE_PX * margin), SAMPLE_PX - int(SAMPLE_PX * margin)
    cells = warped[..., ::-1].astype(np.float32).reshape(side, SAMPLE_PX, side, SAMPLE_PX, 3)
    return cells[:, a:b, :, a:b].mean(axis=(1, 3))


def read_meta(grid: np.ndarray, version: int) -> int:
    """Profile id whose function pattern best matches the sampled symbol."""
    n = 17 + 4 * version
    luminance = grid.mean(axis=2)
    core = luminance[QUIET:QUIET + n, QUIET:QUIET + n]
    is_function, is_dark = function_pattern(version)
    threshold = (np.median(core[is_function & is_dark]) + np.median(core[is_function & ~is_dark])) / 2
    dark = core < threshold
    mismatches = [np.count_nonzero((dark != function_pattern(version, m)[1]) & is_function) for m in range(32)]
    meta = int(np.argmin(mismatches))
    if mismatches[meta] > 0.1 * is_function.sum():
        raise ValueError("Sampled modules do not match QR function patterns.")
    return meta


def normalize(grid: np.ndarray, version: int, meta: int) -> np.ndarray:
    """Per-channel 0 (black) .. 1 (white) scale, from surfaces fitted to known modules."""
    n = 17 + 4 * version
    side = n + 2 * QUIET
    is_function, is_dark = function_pattern(version, meta)
    known = np.zeros((side, side), bool)
    dark = np.zeros((side, side), bool)
    known[QUIET - 1:QUIET + n + 1, QUIET - 1:QUIET + n + 1] = True  # innermost quiet ring
    known[QUIET:QUIET + n, QUIET:QUIET + n] = is_function
    dark[QUIET:QUIET + n, QUIET:QUIET + n] = is_function & is_dark
    y, x = np.mgrid[0:side, 0:side] / side - 0.5
    terms = [np.ones_like(x), x, y] + ([x * y, x * x, y * y] if version >= 7 else [])
    design = np.stack(terms, axis=-1)
    # ponytail: low-order lighting surfaces; tile-local fits if shadows get sharp.

    def surface(mask, values):
        coef = np.linalg.lstsq(design[mask], values[mask], rcond=None)[0]
        residual = np.abs(design[mask] @ coef - values[mask])
        keep = residual <= 3 * np.median(residual) + 1  # drop outliers such as a cropped quiet zone
        return design @ np.linalg.lstsq(design[mask][keep], values[mask][keep], rcond=None)[0]

    out = np.empty_like(grid)
    for c in range(3):
        black, white = surface(dark, grid[..., c]), surface(known & ~dark, grid[..., c])
        out[..., c] = (grid[..., c] - black) / np.maximum(white - black, 1)
    return out[QUIET:QUIET + n, QUIET:QUIET + n]


def quantize(values: np.ndarray, levels: int) -> np.ndarray:
    """Nearest level after 1-D k-means, which absorbs print and camera tone curves."""
    centers = np.linspace(0, 1, levels)
    for _ in range(8):
        index = np.abs(values[:, None] - centers).argmin(axis=1)
        centers = np.array([values[index == k].mean() if np.any(index == k) else centers[k] for k in range(levels)])
    return np.abs(values[:, None] - centers).argmin(axis=1)


def decode_grid(grid: np.ndarray, version: int) -> bytes:
    meta = read_meta(grid, version)
    levels, mode, ecc = unpack_meta(meta)
    rows, cols = data_cells(version)
    values = normalize(grid, version, meta)[rows, cols]
    if mode == "gray":
        values = values.mean(axis=1, keepdims=True)
    index = np.stack([quantize(values[:, c], levels) for c in range(values.shape[1])], axis=1)
    blocks, size, _ = rs_layout(version, levels, mode, ecc)
    stream = levels_to_stream(index.reshape(-1), bits_per_level(levels), blocks * size)
    return parse_stream(stream, version, levels, mode, ecc)


def decode(image) -> bytes:
    """Decode a GDC symbol from a path, PIL image, or BGR array: exact render or photo."""
    bgr = load_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    finders = find_finders(gray)
    errors = []
    for version in candidate_versions(finders):
        try:
            H = rectify(gray, finders, version)
            return decode_grid(sample_modules(bgr, H, 17 + 4 * version), version)
        except ValueError as exc:
            errors.append(f"v{version}: {exc}")
    raise ValueError("Decoding failed. " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=f"GDC v{VERSION}: multilevel QR codes")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode text or a file into a PNG")
    enc.add_argument("output", type=Path)
    source = enc.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    enc.add_argument("--version", type=int, choices=range(1, 41), metavar="1-40")
    enc.add_argument("--module-px", type=int, default=16)

    dec = sub.add_parser("decode", help="decode a PNG or camera photo")
    dec.add_argument("image", type=Path)
    dec.add_argument("--output", type=Path, help="write bytes here instead of printing text")

    cap = sub.add_parser("capacity", help="compare capacity with standard QR")
    for command in (enc, cap):
        command.add_argument("--levels", type=int, choices=LEVELS, default=4)
        command.add_argument("--mode", choices=MODES, default="gray")
        command.add_argument("--ecc", choices=ECC, default="M")

    args = parser.parse_args(argv)
    if args.command == "encode":
        payload = args.text.encode() if args.text is not None else args.file.read_bytes()
        image = encode(payload, args.version, args.levels, args.mode, args.ecc, args.module_px)
        image.save(args.output)
        print(f"Saved {args.output}: {len(payload)} bytes, {image.width}x{image.height}px")
    elif args.command == "decode":
        payload = decode(args.image)
        if args.output:
            args.output.write_bytes(payload)
            print(f"Wrote {len(payload)} bytes to {args.output}")
        else:
            print(payload.decode("utf-8", errors="replace"))
    else:
        print(f"{'version':>7} {'modules':>8} {'QR bytes':>9} {'GDC bytes':>10} {'gain':>6}")
        for version in range(1, 41):
            qr, ours = qr_capacity(version, args.ecc), capacity(version, args.levels, args.mode, args.ecc)
            n = 17 + 4 * version
            print(f"{version:>7} {f'{n}x{n}':>8} {qr:>9} {ours:>10} {ours / qr:>5.2f}x")


if __name__ == "__main__":
    main()
