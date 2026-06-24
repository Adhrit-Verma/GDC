import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAGIC = b"GDC"
VERSION = 6
HEADER_SIZE = 3 + 1 + 4 + 4  # MAGIC + VER + LEN + CRC32

# ============================================================
# ---------------------- FIXED STANDARD ----------------------
# ============================================================

QUIET_ZONE_BLOCKS = 1
MARKER_SIZE_BLOCKS = 11
PALETTE_ROWS = MARKER_SIZE_BLOCKS
SEPARATOR_ROWS = 2
RESERVED_TOP_ROWS = PALETTE_ROWS + SEPARATOR_ROWS

CONTENT_SIDE = 99       # fixed core size
BLOCK_SIZE = 16        # fixed block size
MAX_PHYSICAL_SIZE = 2048

# 8 levels per channel = 512 states = 9 bits per cell
# These are "fraction-like" ordered levels from 0.0 to 1.0
LEVEL_FRACTIONS = [0.00, 0.14, 0.28, 0.42, 0.58, 0.72, 0.86, 1.00]
LEVEL_COUNT = len(LEVEL_FRACTIONS)
BITS_PER_CHANNEL = 3
BITS_PER_CELL = 9

FINDER_PATTERN_11 = [
    [1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1],
]


def clear_screen():
    print("\n" * 40)


def pause():
    input("\nPress ENTER to continue...")


# ============================================================
# ---------------------- BASIC UTILS -------------------------
# ============================================================

def fixed_side_blocks() -> int:
    return CONTENT_SIDE + 2 * QUIET_ZONE_BLOCKS


def fixed_image_size_px() -> int:
    return fixed_side_blocks() * BLOCK_SIZE


def validate_fixed_standard():
    final_px = fixed_image_size_px()
    if final_px > MAX_PHYSICAL_SIZE:
        raise ValueError(
            f"Fixed standard too large: {final_px}px exceeds {MAX_PHYSICAL_SIZE}px. "
            f"Reduce CONTENT_SIDE or BLOCK_SIZE."
        )


def bytes_to_bitstring(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def bitstring_to_bytes(bits: str) -> bytes:
    usable = len(bits) - (len(bits) % 8)
    bits = bits[:usable]
    out = bytearray()
    for i in range(0, usable, 8):
        out.append(int(bits[i:i + 8], 2))
    return bytes(out)


def build_stream(payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = bytearray()
    header.extend(MAGIC)
    header.append(VERSION)
    header.extend(len(payload).to_bytes(4, "big"))
    header.extend(crc.to_bytes(4, "big"))
    return bytes(header) + payload


def parse_stream(stream: bytes) -> bytes:
    if len(stream) < HEADER_SIZE:
        raise ValueError("Invalid or corrupted GDC image (too short).")

    if stream[:3] != MAGIC:
        raise ValueError("This image is NOT a valid GDC code (magic mismatch).")

    version = stream[3]
    if version != VERSION:
        raise ValueError(f"GDC version mismatch. Expected {VERSION}, got {version}")

    length = int.from_bytes(stream[4:8], "big")
    crc_expected = int.from_bytes(stream[8:12], "big")

    payload = stream[HEADER_SIZE:HEADER_SIZE + length]
    if len(payload) != length:
        raise ValueError("Corrupted GDC data: length exceeds available bytes.")

    crc_actual = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_actual != crc_expected:
        raise ValueError("CRC mismatch. Data corrupted.")

    return payload


# ============================================================
# ---------------------- LAYOUT ------------------------------
# ============================================================

def in_top_left_marker(row: int, col: int) -> bool:
    return row < MARKER_SIZE_BLOCKS and col < MARKER_SIZE_BLOCKS


def in_top_right_marker(row: int, col: int, side: int) -> bool:
    return row < MARKER_SIZE_BLOCKS and col >= side - MARKER_SIZE_BLOCKS


def in_bottom_left_marker(row: int, col: int, side: int) -> bool:
    return row >= side - MARKER_SIZE_BLOCKS and col < MARKER_SIZE_BLOCKS


def in_any_marker(row: int, col: int, side: int) -> bool:
    return (
        in_top_left_marker(row, col)
        or in_top_right_marker(row, col, side)
        or in_bottom_left_marker(row, col, side)
    )


def in_palette_band(row: int, col: int, side: int) -> bool:
    return row < PALETTE_ROWS and MARKER_SIZE_BLOCKS <= col < side - MARKER_SIZE_BLOCKS


def in_separator_rows(row: int) -> bool:
    return PALETTE_ROWS <= row < RESERVED_TOP_ROWS


def is_data_cell(row: int, col: int, side: int) -> bool:
    if in_any_marker(row, col, side):
        return False
    if in_palette_band(row, col, side):
        return False
    if in_separator_rows(row):
        return False
    if row < PALETTE_ROWS:
        return False
    return True


def iter_data_cells(side: int):
    for row in range(side):
        for col in range(side):
            if is_data_cell(row, col, side):
                yield row, col


def count_capacity_cells(side: int) -> int:
    return sum(1 for _ in iter_data_cells(side))


def max_payload_bytes() -> int:
    stream_capacity_bytes = (count_capacity_cells(CONTENT_SIDE) * BITS_PER_CELL) // 8
    return max(0, stream_capacity_bytes - HEADER_SIZE)


# ============================================================
# ---------------------- ENCODING STATES ---------------------
# ============================================================

def frac_to_u8(f: float) -> int:
    return int(round(max(0.0, min(1.0, f)) * 255.0))


LEVEL_U8 = [frac_to_u8(f) for f in LEVEL_FRACTIONS]


def symbol_to_rgb(symbol: int):
    # 9 bits => 3 bits per channel
    r_idx = (symbol >> 6) & 0b111
    g_idx = (symbol >> 3) & 0b111
    b_idx = symbol & 0b111
    return (LEVEL_U8[r_idx], LEVEL_U8[g_idx], LEVEL_U8[b_idx])


def pack_bits_to_symbols(bits: str):
    symbols = []
    for i in range(0, len(bits), BITS_PER_CELL):
        chunk = bits[i:i + BITS_PER_CELL]
        if len(chunk) < BITS_PER_CELL:
            chunk = chunk.ljust(BITS_PER_CELL, "0")
        symbols.append(int(chunk, 2))
    return symbols


def unpack_symbols_to_bits(symbols):
    return "".join(f"{s:09b}" for s in symbols)


# ============================================================
# ---------------------- DRAWING -----------------------------
# ============================================================

def draw_marker(px, top: int, left: int):
    for r in range(MARKER_SIZE_BLOCKS):
        for c in range(MARKER_SIZE_BLOCKS):
            px[left + c, top + r] = (0, 0, 0) if FINDER_PATTERN_11[r][c] else (255, 255, 255)


def write_palette_and_separators(core_px, side: int):

    palette_start = MARKER_SIZE_BLOCKS
    palette_end = side - MARKER_SIZE_BLOCKS
    palette_width = palette_end - palette_start

    # ---- ONLY 3 palette rows ----
    for channel in range(3):

        for x in range(palette_width):

            idx = min((x * LEVEL_COUNT) // palette_width, LEVEL_COUNT - 1)

            if channel == 0:
                core_px[palette_start + x, channel] = (LEVEL_U8[idx], 0, 0)

            elif channel == 1:
                core_px[palette_start + x, channel] = (0, LEVEL_U8[idx], 0)

            else:
                core_px[palette_start + x, channel] = (0, 0, LEVEL_U8[idx])

    # separator is BELOW the full finder/palette band
    sep_row_1 = PALETTE_ROWS
    sep_row_2 = PALETTE_ROWS + 1

    for col in range(side):
        core_px[col, sep_row_1] = (0, 0, 0) if col % 2 == 0 else (255, 255, 255)
        core_px[col, sep_row_2] = (255, 255, 255) if col % 2 == 0 else (0, 0, 0)


def filler_symbol(idx: int) -> int:
    # deterministic filler so no dead white space
    return ((idx * 73) ^ (idx >> 1) ^ 0x12D) & 0x1FF  # 9-bit symbol


def stream_to_image(stream: bytes):
    validate_fixed_standard()

    content_side = CONTENT_SIDE
    side_blocks = fixed_side_blocks()

    logical = Image.new("RGB", (side_blocks, side_blocks), color=(255, 255, 255))
    x0 = QUIET_ZONE_BLOCKS
    y0 = QUIET_ZONE_BLOCKS

    core = Image.new("RGB", (content_side, content_side), color=(255, 255, 255))
    core_px = core.load()

    draw_marker(core_px, 0, 0)
    draw_marker(core_px, 0, content_side - MARKER_SIZE_BLOCKS)
    draw_marker(core_px, content_side - MARKER_SIZE_BLOCKS, 0)

    write_palette_and_separators(core_px, content_side)

    bits = bytes_to_bitstring(stream)
    symbols = pack_bits_to_symbols(bits)

    cells = list(iter_data_cells(content_side))
    if len(symbols) > len(cells):
        raise ValueError(
            f"Payload too large for fixed GDC standard.\n"
            f"Max payload: {max_payload_bytes()} bytes\n"
            f"Given payload: {len(stream) - HEADER_SIZE} bytes"
        )

    for idx, (row, col) in enumerate(cells):
        sym = symbols[idx] if idx < len(symbols) else filler_symbol(idx)
        core_px[col, row] = symbol_to_rgb(sym)

    logical.paste(core, (x0, y0))
    img = logical.resize((side_blocks * BLOCK_SIZE, side_blocks * BLOCK_SIZE), Image.NEAREST)
    return img, content_side, BLOCK_SIZE, side_blocks


# ============================================================
# ---------------------- SAMPLING / CALIBRATION --------------
# ============================================================

def sample_logical_colors(img: Image.Image, side_blocks: int):
    px = img.load()
    sampled = Image.new("RGB", (side_blocks, side_blocks), color=(255, 255, 255))
    out = sampled.load()

    for row in range(side_blocks):
        for col in range(side_blocks):
            cx = col * BLOCK_SIZE + BLOCK_SIZE // 2
            cy = row * BLOCK_SIZE + BLOCK_SIZE // 2
            out[col, row] = px[cx, cy]

    return sampled


def validate_marker(core_px, top: int, left: int) -> bool:
    for r in range(MARKER_SIZE_BLOCKS):
        for c in range(MARKER_SIZE_BLOCKS):
            expected_black = FINDER_PATTERN_11[r][c] == 1
            actual = core_px[left + c, top + r]
            brightness = sum(actual) / 3
            actual_black = brightness < 128
            if actual_black != expected_black:
                return False
    return True


def read_palette_models(core_px, side: int):
    palette_start = MARKER_SIZE_BLOCKS
    palette_end = side - MARKER_SIZE_BLOCKS
    palette_width = max(0, palette_end - palette_start)

    refs_r = []
    refs_g = []
    refs_b = []

    for level_idx in range(LEVEL_COUNT):
        x_start = palette_start + (level_idx * palette_width) // LEVEL_COUNT
        x_end = palette_start + ((level_idx + 1) * palette_width) // LEVEL_COUNT
        if x_end <= x_start:
            x_end = x_start + 1

        # Red row
        vals = [core_px[x, 0][0] for x in range(x_start, min(x_end, side))]
        refs_r.append(float(sum(vals)) / max(1, len(vals)))

        # Green row
        vals = [core_px[x, 1][1] for x in range(x_start, min(x_end, side))]
        refs_g.append(float(sum(vals)) / max(1, len(vals)))

        # Blue row
        vals = [core_px[x, 2][2] for x in range(x_start, min(x_end, side))]
        refs_b.append(float(sum(vals)) / max(1, len(vals)))

    return refs_r, refs_g, refs_b


def nearest_level_index(value: int, refs):
    best_idx = 0
    best_dist = None
    for i, ref in enumerate(refs):
        dist = abs(float(value) - float(ref))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def decode_symbols_from_core(core):
    core_px = core.load()
    content_side = core.size[0]

    if not validate_marker(core_px, 0, 0):
        raise ValueError("Top-left marker mismatch.")
    if not validate_marker(core_px, 0, content_side - MARKER_SIZE_BLOCKS):
        raise ValueError("Top-right marker mismatch.")
    if not validate_marker(core_px, content_side - MARKER_SIZE_BLOCKS, 0):
        raise ValueError("Bottom-left marker mismatch.")

    refs_r, refs_g, refs_b = read_palette_models(core_px, content_side)

    symbols = []
    for row, col in iter_data_cells(content_side):
        r, g, b = core_px[col, row]
        r_idx = nearest_level_index(r, refs_r)
        g_idx = nearest_level_index(g, refs_g)
        b_idx = nearest_level_index(b, refs_b)
        symbol = (r_idx << 6) | (g_idx << 3) | b_idx
        symbols.append(symbol)

    return symbols


def decode_core_to_payload(core):
    symbols = decode_symbols_from_core(core)
    bits = unpack_symbols_to_bits(symbols)
    stream = bitstring_to_bytes(bits)
    return parse_stream(stream)


# ============================================================
# ---------------------- EXACT IMAGE DECODE ------------------
# ============================================================

def image_to_payload(path: Path) -> bytes:
    img = Image.open(path).convert("RGB")
    width, height = img.size

    if width != height:
        raise ValueError("GDC image must be square.")

    expected_size = fixed_image_size_px()
    if width != expected_size:
        raise ValueError(
            f"Unexpected image size. This fixed GDC standard expects {expected_size}x{expected_size}px."
        )

    side_blocks = fixed_side_blocks()
    logical = sample_logical_colors(img, side_blocks)

    x0 = QUIET_ZONE_BLOCKS
    y0 = QUIET_ZONE_BLOCKS
    core = logical.crop((x0, y0, x0 + CONTENT_SIDE, y0 + CONTENT_SIDE))
    return decode_core_to_payload(core)


# ============================================================
# ---------------------- PHOTO DECODE ------------------------
# ============================================================

def order_box_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def candidate_marker_boxes(gray):
    # dark threshold because markers are black-heavy
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        if len(approx) != 4:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue

        ratio = max(w, h) / min(w, h)
        if ratio > 1.35:
            continue

        box = cv2.boxPoints(rect)
        box = order_box_points(box)
        candidates.append({
            "area": area,
            "center": np.array([cx, cy], dtype=np.float32),
            "box": box,
        })

    candidates.sort(key=lambda x: x["area"], reverse=True)
    return candidates[:20]


def choose_three_markers(candidates):
    if len(candidates) < 3:
        raise ValueError("Could not find enough marker candidates.")

    best = None
    best_score = None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                trio = [candidates[i], candidates[j], candidates[k]]
                centers = np.array([t["center"] for t in trio], dtype=np.float32)

                # classify TL, TR, BL using sums/diffs
                s = centers.sum(axis=1)
                d = centers[:, 0] - centers[:, 1]

                tl_idx = int(np.argmin(s))
                tr_idx = int(np.argmax(d))

                remaining = [idx for idx in range(3) if idx not in (tl_idx, tr_idx)]
                if len(remaining) != 1:
                    continue
                bl_idx = remaining[0]

                tl = centers[tl_idx]
                tr = centers[tr_idx]
                bl = centers[bl_idx]

                # geometry sanity
                if tr[0] <= tl[0] or bl[1] <= tl[1]:
                    continue

                horizontal = np.linalg.norm(tr - tl)
                vertical = np.linalg.norm(bl - tl)
                score = abs(horizontal - vertical)

                if best_score is None or score < best_score:
                    best_score = score
                    best = (trio[tl_idx], trio[tr_idx], trio[bl_idx])

    if best is None:
        raise ValueError("Could not identify marker geometry.")
    return best


def marker_outer_corner(box, which):
    # box is ordered tl,tr,br,bl
    if which == "tl":
        return box[0]
    if which == "tr":
        return box[1]
    if which == "bl":
        return box[3]
    raise ValueError("Invalid corner name.")


def warp_photo_to_core(photo_path: Path):
    img = cv2.imread(str(photo_path))
    if img is None:
        raise ValueError("Could not read photo file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = candidate_marker_boxes(gray)
    tl_marker, tr_marker, bl_marker = choose_three_markers(candidates)

    tl = marker_outer_corner(tl_marker["box"], "tl")
    tr = marker_outer_corner(tr_marker["box"], "tr")
    bl = marker_outer_corner(bl_marker["box"], "bl")
    br = tr + bl - tl

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_size = fixed_image_size_px()
    dst = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (dst_size, dst_size))

    pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    side_blocks = fixed_side_blocks()
    logical = sample_logical_colors(pil, side_blocks)

    x0 = QUIET_ZONE_BLOCKS
    y0 = QUIET_ZONE_BLOCKS
    core = logical.crop((x0, y0, x0 + CONTENT_SIDE, y0 + CONTENT_SIDE))
    return core


def photo_to_payload(photo_path: Path) -> bytes:
    core = warp_photo_to_core(photo_path)
    return decode_core_to_payload(core)


# ============================================================
# ---------------------- UI ----------------------------------
# ============================================================

def encode_text():
    clear_screen()
    print("=== TEXT -> GDC v6 ENCODER ===\n")

    out_path = input("Enter output image filename (example: gdc_text_v6.png):\n> ").strip()
    if not out_path:
        print("Invalid file name.")
        pause()
        return

    text = input("\nEnter text to encode:\n> ")
    if not text.strip():
        print("Empty text. Cancelled.")
        pause()
        return

    payload = text.encode("utf-8")
    try:
        stream = build_stream(payload)
        img, content_side, block_size, side_blocks = stream_to_image(stream)
        img.save(out_path, "PNG")
    except Exception as e:
        print(f"\nEncoding denied: {e}")
        pause()
        return

    print(f"\nSaved: {out_path}")
    print(f"Payload bytes: {len(payload)}")
    print(f"Max payload allowed: {max_payload_bytes()} bytes")
    print(f"Content side: {content_side} blocks")
    print(f"Logical side incl. quiet zone: {side_blocks} blocks")
    print(f"Block size: {block_size}px")
    print(f"Final image size: {img.size[0]} x {img.size[1]} px")
    pause()


def encode_file():
    clear_screen()
    print("=== FILE -> GDC v6 ENCODER ===\n")

    filepath = input("Enter file path to encode:\n> ").strip()
    p = Path(filepath)
    if not p.is_file():
        print("File not found.")
        pause()
        return

    out_path = input("\nEnter output image filename (example: gdc_file_v6.png):\n> ").strip()
    if not out_path:
        print("Invalid file name.")
        pause()
        return

    payload = p.read_bytes()
    try:
        stream = build_stream(payload)
        img, content_side, block_size, side_blocks = stream_to_image(stream)
        img.save(out_path, "PNG")
    except Exception as e:
        print(f"\nEncoding denied: {e}")
        pause()
        return

    print(f"\nSaved: {out_path}")
    print(f"Original file size: {len(payload)} bytes")
    print(f"Max payload allowed: {max_payload_bytes()} bytes")
    print(f"Content side: {content_side} blocks")
    print(f"Logical side incl. quiet zone: {side_blocks} blocks")
    print(f"Block size: {block_size}px")
    print(f"Final image size: {img.size[0]} x {img.size[1]} px")
    pause()


def decode_exact_image():
    clear_screen()
    print("=== EXACT GDC IMAGE DECODER ===\n")

    filepath = input("Enter exact GDC image path:\n> ").strip()
    p = Path(filepath)
    if not p.is_file():
        print("File not found.")
        pause()
        return

    try:
        payload = image_to_payload(p)
    except Exception as e:
        print(f"\nFailed to decode: {e}")
        pause()
        return

    print("\nSelect output type:")
    print("1) Show as text")
    print("2) Save as file")
    choice = input("> ").strip()

    if choice == "1":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            print("Some characters were not valid UTF-8 and were replaced.")
        print("\n--- DECODED TEXT ---\n")
        print(text)
        print("\n--------------------")
    elif choice == "2":
        out_path = input("Enter output file path:\n> ").strip()
        Path(out_path).write_bytes(payload)
        print(f"\nPayload saved to: {out_path}")
    else:
        print("Invalid choice.")

    pause()


def decode_temp_photo():
    clear_screen()
    print("=== TEMP PHOTO -> GDC DECODER ===\n")
    print("Use this on a captured photo of the GDC symbol.\n")

    filepath = input("Enter photo path:\n> ").strip()
    p = Path(filepath)
    if not p.is_file():
        print("File not found.")
        pause()
        return

    try:
        payload = photo_to_payload(p)
    except Exception as e:
        print(f"\nPhoto decode failed: {e}")
        pause()
        return

    print("\nSelect output type:")
    print("1) Show as text")
    print("2) Save as file")
    choice = input("> ").strip()

    if choice == "1":
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            print("Some characters were not valid UTF-8 and were replaced.")
        print("\n--- DECODED TEXT ---\n")
        print(text)
        print("\n--------------------")
    elif choice == "2":
        out_path = input("Enter output file path:\n> ").strip()
        Path(out_path).write_bytes(payload)
        print(f"\nPayload saved to: {out_path}")
    else:
        print("Invalid choice.")

    pause()


def show_capacity():
    clear_screen()
    print("=== GDC v6 FIXED STANDARD ===\n")
    print(f"Content side: {CONTENT_SIDE} blocks")
    print(f"Marker size: {MARKER_SIZE_BLOCKS} blocks")
    print(f"Palette rows: {PALETTE_ROWS}")
    print(f"Separator rows: {SEPARATOR_ROWS}")
    print(f"Block size: {BLOCK_SIZE}px")
    print(f"Final size: {fixed_image_size_px()} x {fixed_image_size_px()} px")
    print(f"Bits per cell: {BITS_PER_CELL}")
    print(f"Max payload: ~{max_payload_bytes()} bytes")
    pause()


def main_menu():
    while True:
        clear_screen()
        print("==========================================")
        print("             GDC TOOL v6")
        print("  3-row palette + calibrated photo decode")
        print("==========================================")
        print("1) Encode text")
        print("2) Encode a file")
        print("3) Decode exact GDC image")
        print("4) Decode temp photo of GDC")
        print("5) Show fixed capacity")
        print("6) Exit")
        print("==========================================")

        choice = input("> ").strip()
        if choice == "1":
            encode_text()
        elif choice == "2":
            encode_file()
        elif choice == "3":
            decode_exact_image()
        elif choice == "4":
            decode_temp_photo()
        elif choice == "5":
            show_capacity()
        elif choice == "6":
            clear_screen()
            print("Goodbye")
            sys.exit(0)
        else:
            print("Invalid choice.")
            pause()


if __name__ == "__main__":
    main_menu()