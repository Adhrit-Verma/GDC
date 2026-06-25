import hashlib
import struct
import zlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import qrcode
from qrcode.constants import ERROR_CORRECT_L
from qrcode.util import pattern_position

try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'reedsolo'. Install project requirements with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc


# GDC v10 uses a QR-first field profile: a Version-10-sized 57x57 matrix with
# standard-looking three-corner finders, separators, timing tracks, alignment
# target and quiet zone. White carrier modules stay pure white; calibrated RGB
# values replace only dark carrier modules, matching conventional colored QR.
MAGIC = b"GDC"
VERSION = 10
FLAG_COMPRESSED = 0x01
HEADER_FORMAT = ">3sBBIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# ---------------------------------------------------------------------------
# Fixed camera-friendly dense profile
# ---------------------------------------------------------------------------

QR_VERSION = 10
CONTENT_SIDE = 17 + 4 * QR_VERSION
QUIET_ZONE_BLOCKS = 4
BLOCK_SIZE = 60
MAX_IMAGE_SIZE = 4096

MARKER_SIZE_BLOCKS = 7
ALIGNMENT_PATTERN_SIZE = 5
ALIGNMENT_CENTERS = tuple(pattern_position(QR_VERSION))

# 4 levels per channel = 64 RGB states = 6 bits per module. Wide 64-value
# spacing is substantially more tolerant of printing, focus, exposure, JPEG,
# and white-balance errors than the previous 16-level profile.
LEVEL_COUNT = 4
BITS_PER_CHANNEL = 2
BITS_PER_CELL = 6
DATA_LEVEL_U8 = [8, 44, 80, 116]

# RS(255, 223): 32 parity bytes correct up to 16 erroneous bytes per block.
# Interleaving spreads localized image damage across independent RS blocks.
RS_BLOCK_SIZE = 255
RS_PARITY_BYTES = 32
RS_DATA_BYTES = RS_BLOCK_SIZE - RS_PARITY_BYTES
RS_CODEC = RSCodec(RS_PARITY_BYTES, nsize=RS_BLOCK_SIZE)


def clear_screen():
    print("\n" * 40)


def pause():
    input("\nPress ENTER to continue...")


# ---------------------------------------------------------------------------
# Capacity and stream protection
# ---------------------------------------------------------------------------

def fixed_side_blocks() -> int:
    return CONTENT_SIDE + 2 * QUIET_ZONE_BLOCKS


def fixed_image_size_px() -> int:
    return fixed_side_blocks() * BLOCK_SIZE


def fixed_core_size_px() -> int:
    return CONTENT_SIDE * BLOCK_SIZE


def validate_fixed_standard():
    if fixed_image_size_px() > MAX_IMAGE_SIZE:
        raise ValueError(
            f"GDC image is {fixed_image_size_px()}px, above the "
            f"{MAX_IMAGE_SIZE}px profile limit."
        )


def finder_origins(side: int):
    return (
        (0, 0),
        (0, side - MARKER_SIZE_BLOCKS),
        (side - MARKER_SIZE_BLOCKS, 0),
    )


def in_finder_or_separator(row: int, col: int, side: int) -> bool:
    for top, left in finder_origins(side):
        if (
            max(0, top - 1) <= row <= min(side - 1, top + MARKER_SIZE_BLOCKS)
            and max(0, left - 1) <= col <= min(side - 1, left + MARKER_SIZE_BLOCKS)
        ):
            return True
    return False


@lru_cache(maxsize=None)
def alignment_pattern_centers(side: int):
    centers = []
    radius = ALIGNMENT_PATTERN_SIZE // 2
    for center_row in ALIGNMENT_CENTERS:
        for center_col in ALIGNMENT_CENTERS:
            overlaps_finder = any(
                in_finder_or_separator(row, col, side)
                for row in range(center_row - radius, center_row + radius + 1)
                for col in range(center_col - radius, center_col + radius + 1)
            )
            if not overlaps_finder:
                centers.append((center_row, center_col))
    return tuple(centers)


def in_alignment_pattern(row: int, col: int, side: int) -> bool:
    radius = ALIGNMENT_PATTERN_SIZE // 2
    return any(
        abs(row - center_row) <= radius and abs(col - center_col) <= radius
        for center_row, center_col in alignment_pattern_centers(side)
    )


def in_timing_pattern(row: int, col: int, side: int) -> bool:
    return (
        row == 6 and 8 <= col < side - 8
    ) or (
        col == 6 and 8 <= row < side - 8
    )


def in_format_information(row: int, col: int, side: int) -> bool:
    return (
        (row == 8 and (col <= 8 or col >= side - 8))
        or (col == 8 and (row <= 8 or row >= side - 7))
    )


def in_version_information(row: int, col: int, side: int) -> bool:
    if QR_VERSION < 7:
        return False
    return (
        row < 6 and side - 11 <= col <= side - 9
    ) or (
        col < 6 and side - 11 <= row <= side - 9
    )


def is_qr_function_cell(row: int, col: int, side: int) -> bool:
    return not (
        not in_finder_or_separator(row, col, side)
        and not in_alignment_pattern(row, col, side)
        and not in_timing_pattern(row, col, side)
        and not in_format_information(row, col, side)
        and not in_version_information(row, col, side)
        and not (row == side - 8 and col == 8)
    )


@lru_cache(maxsize=None)
def calibration_positions(side: int) -> tuple[tuple[int, int], ...]:
    candidates = [
        (row, col)
        for row in range(9, side - 8)
        for col in range(9, side - 8)
        if not is_qr_function_cell(row, col, side)
    ]
    candidates.sort(
        key=lambda cell: (
            ((cell[0] * 73) ^ (cell[1] * 151) ^ (cell[0] * cell[1] * 17))
            & 0xFFFF
        )
    )
    positions = []
    for _ in calibration_entries():
        match = next(
            cell
            for cell in candidates
            if cell not in positions
            and carrier_is_dark(*cell)
        )
        positions.append(match)
    return tuple(positions)


def is_data_cell(row: int, col: int, side: int) -> bool:
    return (
        not is_qr_function_cell(row, col, side)
        and (row, col) not in calibration_positions(side)
        and carrier_is_dark(row, col)
    )


def carrier_is_dark(row: int, col: int) -> bool:
    return qr_carrier_matrix()[row][col]


@lru_cache(maxsize=1)
def qr_carrier_matrix() -> tuple[tuple[bool, ...], ...]:
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    qr.add_data("GDC-V10-COLOR-CARRIER")
    qr.make(fit=False)
    matrix = tuple(tuple(bool(value) for value in row) for row in qr.get_matrix())
    if len(matrix) != CONTENT_SIDE or any(len(row) != CONTENT_SIDE for row in matrix):
        raise ValueError("Generated QR carrier has unexpected dimensions.")
    return matrix


@lru_cache(maxsize=None)
def data_cells(side: int) -> tuple[tuple[int, int], ...]:
    # QR-style two-column vertical zigzag, right to left.
    cells = []
    right = side - 1
    upward = True
    while right > 0:
        if right == 6:
            right -= 1
        rows = range(side - 1, -1, -1) if upward else range(side)
        for row in rows:
            for col in (right, right - 1):
                if is_data_cell(row, col, side):
                    cells.append((row, col))
        upward = not upward
        right -= 2
    return tuple(cells)


def iter_data_cells(side: int):
    yield from data_cells(side)


def count_capacity_cells(side: int = CONTENT_SIDE) -> int:
    return len(data_cells(side))


def raw_stream_capacity_bytes() -> int:
    return (count_capacity_cells() * BITS_PER_CELL) // 8


def rs_block_count() -> int:
    return raw_stream_capacity_bytes() // RS_BLOCK_SIZE


def protected_stream_capacity_bytes() -> int:
    return rs_block_count() * RS_BLOCK_SIZE


def protected_data_capacity_bytes() -> int:
    return rs_block_count() * RS_DATA_BYTES


def max_payload_bytes() -> int:
    """Maximum protected incompressible payload. Compression may allow more."""
    return protected_data_capacity_bytes() - HEADER_SIZE


def _stored_payload(payload: bytes) -> tuple[int, bytes]:
    compressed = zlib.compress(payload, level=9)
    if len(compressed) < len(payload):
        return FLAG_COMPRESSED, compressed
    return 0, payload


def _interleave_blocks(blocks: list[bytes]) -> bytes:
    return bytes(
        blocks[block_index][byte_index]
        for byte_index in range(RS_BLOCK_SIZE)
        for block_index in range(len(blocks))
    )


def _deinterleave_blocks(stream: bytes) -> list[bytearray]:
    block_count = rs_block_count()
    expected = block_count * RS_BLOCK_SIZE
    if len(stream) < expected:
        raise ValueError(f"Protected stream is truncated: expected {expected} bytes.")

    blocks = [bytearray(RS_BLOCK_SIZE) for _ in range(block_count)]
    offset = 0
    for byte_index in range(RS_BLOCK_SIZE):
        for block_index in range(block_count):
            blocks[block_index][byte_index] = stream[offset]
            offset += 1
    return blocks


def build_stream(payload: bytes) -> bytes:
    flags, stored = _stored_payload(payload)
    if len(stored) + HEADER_SIZE > protected_data_capacity_bytes():
        raise ValueError(
            "Payload is too large for this GDC profile.\n"
            f"Original bytes: {len(payload)}\n"
            f"Stored bytes after optional compression: {len(stored)}\n"
            f"Maximum stored payload: {max_payload_bytes()}"
        )

    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        flags,
        len(payload),
        len(stored),
        crc,
    )
    protected_data = header + stored

    # Whiten padding so short payloads do not create large uniform regions that
    # can be mistaken for finder patterns by a camera.
    padding_size = protected_data_capacity_bytes() - len(protected_data)
    padding = hashlib.shake_256(protected_data).digest(padding_size)
    protected_data += padding

    encoded_blocks = []
    for offset in range(0, len(protected_data), RS_DATA_BYTES):
        block = protected_data[offset:offset + RS_DATA_BYTES]
        encoded_blocks.append(bytes(RS_CODEC.encode(block)))

    return _interleave_blocks(encoded_blocks)


def parse_stream(stream: bytes) -> bytes:
    decoded_blocks = []
    for block_number, block in enumerate(_deinterleave_blocks(stream), start=1):
        try:
            decoded, _, _ = RS_CODEC.decode(block)
        except ReedSolomonError as exc:
            raise ValueError(
                f"Too much damage to recover Reed-Solomon block {block_number}."
            ) from exc
        decoded_blocks.append(bytes(decoded))

    protected_data = b"".join(decoded_blocks)
    if len(protected_data) < HEADER_SIZE:
        raise ValueError("Invalid GDC stream: protected header is missing.")

    magic, version, flags, original_length, stored_length, expected_crc = struct.unpack(
        HEADER_FORMAT, protected_data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError("This image is not a valid GDC code.")
    if version != VERSION:
        raise ValueError(f"GDC version mismatch. Expected {VERSION}, got {version}.")
    if flags & ~FLAG_COMPRESSED:
        raise ValueError("GDC header contains unsupported flags.")
    if stored_length > max_payload_bytes():
        raise ValueError("GDC header contains an invalid stored length.")

    stored = protected_data[HEADER_SIZE:HEADER_SIZE + stored_length]
    if len(stored) != stored_length:
        raise ValueError("GDC payload is truncated.")

    if flags & FLAG_COMPRESSED:
        try:
            payload = zlib.decompress(stored)
        except zlib.error as exc:
            raise ValueError("Compressed GDC payload is damaged.") from exc
    else:
        payload = stored

    if len(payload) != original_length:
        raise ValueError("Decoded payload length does not match its header.")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("CRC mismatch after error correction.")
    return payload


# ---------------------------------------------------------------------------
# Symbol conversion and drawing
# ---------------------------------------------------------------------------

def bytes_to_symbols(data: bytes) -> list[int]:
    accumulator = 0
    bit_count = 0
    symbols = []
    mask = (1 << BITS_PER_CELL) - 1

    for value in data:
        accumulator = (accumulator << 8) | value
        bit_count += 8
        while bit_count >= BITS_PER_CELL:
            bit_count -= BITS_PER_CELL
            symbols.append((accumulator >> bit_count) & mask)
            accumulator &= (1 << bit_count) - 1 if bit_count else 0

    if bit_count:
        symbols.append((accumulator << (BITS_PER_CELL - bit_count)) & mask)
    return symbols


def symbols_to_bytes(symbols: list[int], byte_length: int) -> bytes:
    accumulator = 0
    bit_count = 0
    output = bytearray()

    for symbol in symbols:
        accumulator = (accumulator << BITS_PER_CELL) | symbol
        bit_count += BITS_PER_CELL
        while bit_count >= 8 and len(output) < byte_length:
            bit_count -= 8
            output.append((accumulator >> bit_count) & 0xFF)
            accumulator &= (1 << bit_count) - 1 if bit_count else 0
        if len(output) == byte_length:
            break
    return bytes(output)


def symbol_to_rgb(symbol: int) -> tuple[int, int, int]:
    channel_mask = LEVEL_COUNT - 1
    r_index = (symbol >> (BITS_PER_CHANNEL * 2)) & channel_mask
    g_index = (symbol >> BITS_PER_CHANNEL) & channel_mask
    b_index = symbol & channel_mask
    return (
        DATA_LEVEL_U8[r_index],
        DATA_LEVEL_U8[g_index],
        DATA_LEVEL_U8[b_index],
    )


def filler_symbol(index: int) -> int:
    mask = (1 << BITS_PER_CELL) - 1
    return ((index * 2654435761) ^ (index >> 3) ^ 0x2B) & mask


def marker_is_black(row: int, col: int) -> bool:
    # Standard QR 7x7 finder: black border, white ring, black 3x3 center.
    distance_to_edge = min(
        row,
        col,
        MARKER_SIZE_BLOCKS - 1 - row,
        MARKER_SIZE_BLOCKS - 1 - col,
    )
    return distance_to_edge == 0 or distance_to_edge >= 2


def calibration_entries():
    for channel in range(3):
        for level_index in range(LEVEL_COUNT):
            yield channel, level_index


def write_calibration_samples(pixels, side: int):
    for (row, col), (channel, level_index) in zip(
        calibration_positions(side),
        calibration_entries(),
    ):
        color = [DATA_LEVEL_U8[0], DATA_LEVEL_U8[0], DATA_LEVEL_U8[0]]
        color[channel] = DATA_LEVEL_U8[level_index]
        pixels[col, row] = tuple(color)


def draw_qr_carrier(pixels):
    for row, matrix_row in enumerate(qr_carrier_matrix()):
        for col, dark in enumerate(matrix_row):
            pixels[col, row] = (0, 0, 0) if dark else (255, 255, 255)


def stream_to_image(stream: bytes):
    validate_fixed_standard()
    expected_stream_size = protected_stream_capacity_bytes()
    if len(stream) != expected_stream_size:
        raise ValueError(
            f"Protected stream must contain exactly {expected_stream_size} bytes."
        )

    cells = list(iter_data_cells(CONTENT_SIDE))
    symbols = bytes_to_symbols(stream)
    if len(symbols) > len(cells):
        raise ValueError("Protected stream exceeds the symbol's module capacity.")

    core = Image.new("RGB", (CONTENT_SIDE, CONTENT_SIDE), (255, 255, 255))
    pixels = core.load()
    draw_qr_carrier(pixels)
    write_calibration_samples(pixels, CONTENT_SIDE)

    for index, (row, col) in enumerate(cells):
        symbol = symbols[index] if index < len(symbols) else filler_symbol(index)
        pixels[col, row] = symbol_to_rgb(symbol)

    logical = Image.new(
        "RGB",
        (fixed_side_blocks(), fixed_side_blocks()),
        color=(255, 255, 255),
    )
    logical.paste(core, (QUIET_ZONE_BLOCKS, QUIET_ZONE_BLOCKS))
    image = logical.resize(
        (fixed_image_size_px(), fixed_image_size_px()),
        Image.Resampling.NEAREST,
    )
    return image, CONTENT_SIDE, BLOCK_SIZE, fixed_side_blocks()


# ---------------------------------------------------------------------------
# Sampling and calibration
# ---------------------------------------------------------------------------

def sample_grid_colors(
    image: Image.Image,
    columns: int,
    rows: int,
    center_margin_fraction: float = 0.25,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if width < columns or height < rows:
        raise ValueError("Image resolution is too low for this GDC grid.")

    cell_width = width // columns
    cell_height = height // rows
    usable_width = cell_width * columns
    usable_height = cell_height * rows
    rgb = rgb[:usable_height, :usable_width]
    grid = rgb.reshape(rows, cell_height, columns, cell_width, 3)

    x_margin = (
        max(1, int(cell_width * center_margin_fraction))
        if cell_width >= 4
        else 0
    )
    y_margin = (
        max(1, int(cell_height * center_margin_fraction))
        if cell_height >= 4
        else 0
    )
    x_end = cell_width - x_margin if x_margin else cell_width
    y_end = cell_height - y_margin if y_margin else cell_height
    centers = grid[:, y_margin:y_end, :, x_margin:x_end, :]
    sampled = np.rint(centers.mean(axis=(1, 3))).astype(np.uint8)
    return Image.fromarray(sampled, mode="RGB")


def validate_marker(pixels, top: int, left: int, max_mismatch_ratio: float = 0.08) -> bool:
    mismatches = 0
    total = MARKER_SIZE_BLOCKS * MARKER_SIZE_BLOCKS
    for row in range(MARKER_SIZE_BLOCKS):
        for col in range(MARKER_SIZE_BLOCKS):
            expected_black = marker_is_black(row, col)
            actual = pixels[left + col, top + row]
            actual_black = (sum(actual) / 3) < 128
            mismatches += actual_black != expected_black
    return (mismatches / total) <= max_mismatch_ratio


def read_calibration_models(pixels):
    models = [[0.0] * LEVEL_COUNT for _ in range(3)]
    for (row, col), (channel, level_index) in zip(
        calibration_positions(CONTENT_SIDE),
        calibration_entries(),
    ):
        models[channel][level_index] = float(pixels[col, row][channel])
    return models


def nearest_level_index(value: int, references: list[float]) -> int:
    return min(range(len(references)), key=lambda index: abs(value - references[index]))


def decode_symbols_from_core(core: Image.Image) -> list[int]:
    pixels = core.load()
    side = core.size[0]
    if side != CONTENT_SIDE or core.size[1] != CONTENT_SIDE:
        raise ValueError("Sampled GDC core has the wrong dimensions.")

    checks = (
        (0, 0, "top-left"),
        (0, side - MARKER_SIZE_BLOCKS, "top-right"),
        (side - MARKER_SIZE_BLOCKS, 0, "bottom-left"),
    )
    for top, left, name in checks:
        if not validate_marker(pixels, top, left):
            raise ValueError(f"{name.capitalize()} finder marker mismatch.")

    red_refs, green_refs, blue_refs = read_calibration_models(pixels)
    symbols = []
    for row, col in iter_data_cells(side):
        red, green, blue = pixels[col, row]
        symbol = (
            (nearest_level_index(red, red_refs) << (BITS_PER_CHANNEL * 2))
            | (nearest_level_index(green, green_refs) << BITS_PER_CHANNEL)
            | nearest_level_index(blue, blue_refs)
        )
        symbols.append(symbol)
    return symbols


def decode_core_to_payload(core: Image.Image) -> bytes:
    symbols = decode_symbols_from_core(core)
    stream = symbols_to_bytes(symbols, protected_stream_capacity_bytes())
    return parse_stream(stream)


# ---------------------------------------------------------------------------
# Exact image and camera/photo decoding
# ---------------------------------------------------------------------------

def image_to_payload(path: Path) -> bytes:
    image = Image.open(path).convert("RGB")
    if image.width != image.height:
        raise ValueError("GDC image must be square.")
    if image.size != (fixed_image_size_px(), fixed_image_size_px()):
        raise ValueError(
            f"Expected a {fixed_image_size_px()}x{fixed_image_size_px()} GDC v{VERSION} image."
        )

    logical = sample_grid_colors(image, fixed_side_blocks(), fixed_side_blocks())
    start = QUIET_ZONE_BLOCKS
    core = logical.crop((start, start, start + CONTENT_SIDE, start + CONTENT_SIDE))
    return decode_core_to_payload(core)


def order_box_points(points):
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def candidate_marker_boxes(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    minimum_area = max(100.0, gray.shape[0] * gray.shape[1] * 0.00005)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        if area > gray.shape[0] * gray.shape[1] * 0.1:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if len(approximation) != 4:
            continue

        rectangle = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), _ = rectangle
        if min(width, height) <= 0:
            continue
        if max(width, height) / min(width, height) > 1.35:
            continue

        candidates.append(
            {
                "area": float(area),
                "center": np.array([center_x, center_y], dtype=np.float32),
                "box": order_box_points(approximation.reshape(-1, 2)),
                "size": float(min(width, height)),
            }
        )

    # Finder rings create nested contours. Keep only the largest contour at
    # each center so the geometry solver sees one candidate per marker.
    candidates.sort(key=lambda item: item["area"], reverse=True)
    unique = []
    for candidate in candidates:
        duplicate = any(
            np.linalg.norm(candidate["center"] - kept["center"])
            < 0.25 * min(candidate["size"], kept["size"])
            for kept in unique
        )
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= 24:
            break
    return unique


def choose_three_markers(candidates):
    if len(candidates) < 3:
        raise ValueError("Could not find three QR-style finder patterns.")

    best = None
    best_score = None
    largest_area = max(item["area"] for item in candidates)
    finder_sized = [
        item for item in candidates
        if item["area"] >= largest_area * 0.45
    ]
    search = finder_sized[:8] if len(finder_sized) >= 3 else candidates[:12]
    for first in range(len(search)):
        for second in range(first + 1, len(search)):
            for third in range(second + 1, len(search)):
                group = [search[first], search[second], search[third]]
                centers = np.array(
                    [item["center"] for item in group],
                    dtype=np.float32,
                )
                sums = centers.sum(axis=1)
                differences = centers[:, 0] - centers[:, 1]
                top_left_index = int(np.argmin(sums))
                top_right_index = int(np.argmax(differences))
                remaining = [
                    index
                    for index in range(3)
                    if index not in (top_left_index, top_right_index)
                ]
                if len(remaining) != 1:
                    continue
                bottom_left_index = remaining[0]

                top_left = centers[top_left_index]
                top_right = centers[top_right_index]
                bottom_left = centers[bottom_left_index]
                top_vector = top_right - top_left
                left_vector = bottom_left - top_left
                top = np.linalg.norm(top_vector)
                left = np.linalg.norm(left_vector)
                if min(top, left) <= 0:
                    continue
                if top_right[0] <= top_left[0] or bottom_left[1] <= top_left[1]:
                    continue

                orthogonality = abs(float(np.dot(top_vector, left_vector)) / (top * left))
                side_difference = abs(top - left) / max(top, left)
                areas = [item["area"] for item in group]
                area_difference = (max(areas) - min(areas)) / max(areas)
                score = orthogonality + side_difference + area_difference
                if best_score is None or score < best_score:
                    best_score = score
                    best = (
                        group[top_left_index],
                        group[top_right_index],
                        group[bottom_left_index],
                    )

    if best is None:
        raise ValueError("Could not identify GDC finder-marker geometry.")
    return best


def alignment_template() -> np.ndarray:
    logical = np.full(
        (ALIGNMENT_PATTERN_SIZE, ALIGNMENT_PATTERN_SIZE),
        255,
        dtype=np.uint8,
    )
    radius = ALIGNMENT_PATTERN_SIZE // 2
    for row in range(ALIGNMENT_PATTERN_SIZE):
        for col in range(ALIGNMENT_PATTERN_SIZE):
            distance = max(abs(row - radius), abs(col - radius))
            if distance in (0, radius):
                logical[row, col] = 0
    return cv2.resize(
        logical,
        (
            ALIGNMENT_PATTERN_SIZE * BLOCK_SIZE,
            ALIGNMENT_PATTERN_SIZE * BLOCK_SIZE,
        ),
        interpolation=cv2.INTER_NEAREST,
    )


def find_bottom_right_alignment(prewarped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(prewarped, cv2.COLOR_BGR2GRAY)
    expected_center = (ALIGNMENT_CENTERS[-1] + 0.5) * BLOCK_SIZE
    radius = 7 * BLOCK_SIZE
    template = alignment_template()
    half_template = template.shape[0] // 2

    left = max(0, int(expected_center - radius - half_template))
    top = max(0, int(expected_center - radius - half_template))
    right = min(gray.shape[1], int(expected_center + radius + half_template))
    bottom = min(gray.shape[0], int(expected_center + radius + half_template))
    search = gray[top:bottom, left:right]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        raise ValueError("Alignment-pattern search area is too small.")

    # Match at about 12 pixels/module. Full-resolution correlation with a
    # 245x245 template is needlessly expensive and does not improve location.
    match_scale = min(1.0, 12.0 / BLOCK_SIZE)
    search_small = cv2.resize(
        search,
        (
            max(1, round(search.shape[1] * match_scale)),
            max(1, round(search.shape[0] * match_scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    template_small = cv2.resize(
        template,
        (
            max(1, round(template.shape[1] * match_scale)),
            max(1, round(template.shape[0] * match_scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    scores = cv2.matchTemplate(
        search_small,
        template_small,
        cv2.TM_CCOEFF_NORMED,
    )
    _, best_score, _, best_location = cv2.minMaxLoc(scores)
    if best_score < 0.35:
        raise ValueError("Could not locate the bottom-right QR alignment pattern.")

    scale_x = search_small.shape[1] / search.shape[1]
    scale_y = search_small.shape[0] / search.shape[0]
    return np.array(
        [
            left + (best_location[0] + template_small.shape[1] / 2.0) / scale_x,
            top + (best_location[1] + template_small.shape[0] / 2.0) / scale_y,
        ],
        dtype=np.float32,
    )


def warp_photo_to_core(
    photo_path: Path,
    center_margin_fraction: float = 0.25,
) -> Image.Image:
    image = cv2.imread(str(photo_path))
    if image is None:
        raise ValueError("Could not read the photo file.")

    # Finder contour detection does not need the full 4K render. Limiting the
    # search image materially improves camera decoding latency.
    detection_scale = min(1.0, 1200.0 / max(image.shape[:2]))
    if detection_scale < 1.0:
        detection_image = cv2.resize(
            image,
            (
                round(image.shape[1] * detection_scale),
                round(image.shape[0] * detection_scale),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = image
    gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
    top_left_marker, top_right_marker, bottom_left_marker = choose_three_markers(
        candidate_marker_boxes(gray)
    )

    core_size = fixed_core_size_px()
    finder_center = (MARKER_SIZE_BLOCKS / 2.0) * BLOCK_SIZE
    opposite_finder_center = (
        CONTENT_SIDE - MARKER_SIZE_BLOCKS / 2.0
    ) * BLOCK_SIZE
    finder_source = np.array(
        [
            top_left_marker["center"] / detection_scale,
            top_right_marker["center"] / detection_scale,
            bottom_left_marker["center"] / detection_scale,
        ],
        dtype=np.float32,
    )
    finder_destination = np.array(
        [
            [finder_center, finder_center],
            [opposite_finder_center, finder_center],
            [finder_center, opposite_finder_center],
        ],
        dtype=np.float32,
    )

    # First normalize rotation/scale with the three QR finder patterns.
    affine = cv2.getAffineTransform(finder_source, finder_destination)
    prewarped = cv2.warpAffine(
        image,
        affine,
        (core_size, core_size),
        flags=cv2.INTER_LINEAR,
    )

    # Then use the standard bottom-right alignment target as the fourth
    # correspondence for full perspective correction.
    alignment_in_prewarped = find_bottom_right_alignment(prewarped)
    inverse_affine = cv2.invertAffineTransform(affine)
    alignment_source = cv2.transform(
        alignment_in_prewarped.reshape(1, 1, 2),
        inverse_affine,
    ).reshape(2)
    alignment_destination = np.array(
        [
            (ALIGNMENT_CENTERS[-1] + 0.5) * BLOCK_SIZE,
            (ALIGNMENT_CENTERS[-1] + 0.5) * BLOCK_SIZE,
        ],
        dtype=np.float32,
    )

    source = np.vstack(
        [
            finder_source[0],
            finder_source[1],
            alignment_source,
            finder_source[2],
        ]
    ).astype(np.float32)
    destination = np.vstack(
        [
            finder_destination[0],
            finder_destination[1],
            alignment_destination,
            finder_destination[2],
        ]
    ).astype(np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        transform,
        (core_size, core_size),
        flags=cv2.INTER_LINEAR,
    )
    pil_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return sample_grid_colors(
        pil_image,
        CONTENT_SIDE,
        CONTENT_SIDE,
        center_margin_fraction=center_margin_fraction,
    )


def photo_to_payload(photo_path: Path) -> bytes:
    errors = []
    for margin_fraction in (0.25, 0.375):
        try:
            core = warp_photo_to_core(
                photo_path,
                center_margin_fraction=margin_fraction,
            )
            return decode_core_to_payload(core)
        except ValueError as exc:
            errors.append(exc)
    raise ValueError(
        "Photo decoding failed with both normal and blur-resistant sampling: "
        f"{errors[-1]}"
    ) from errors[-1]


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------

def encode_payload(payload: bytes, output_path: str):
    stream = build_stream(payload)
    image, content_side, block_size, side_blocks = stream_to_image(stream)
    image.save(output_path, "PNG")
    flags, stored = _stored_payload(payload)
    return {
        "image": image,
        "content_side": content_side,
        "block_size": block_size,
        "side_blocks": side_blocks,
        "stored_bytes": len(stored),
        "compressed": bool(flags & FLAG_COMPRESSED),
    }


def encode_text():
    clear_screen()
    print(f"=== TEXT -> GDC v{VERSION} ENCODER ===\n")
    output_path = input(f"Output PNG filename (example: gdc_text_v{VERSION}.png):\n> ").strip()
    if not output_path:
        print("Invalid file name.")
        pause()
        return
    text = input("\nText to encode:\n> ")
    if not text:
        print("Empty text. Cancelled.")
        pause()
        return

    payload = text.encode("utf-8")
    try:
        result = encode_payload(payload, output_path)
    except Exception as exc:
        print(f"\nEncoding denied: {exc}")
        pause()
        return
    show_encode_result(output_path, payload, result)


def encode_file():
    clear_screen()
    print(f"=== FILE -> GDC v{VERSION} ENCODER ===\n")
    source_path = Path(input("File path to encode:\n> ").strip())
    if not source_path.is_file():
        print("File not found.")
        pause()
        return
    output_path = input(f"\nOutput PNG filename (example: gdc_file_v{VERSION}.png):\n> ").strip()
    if not output_path:
        print("Invalid file name.")
        pause()
        return

    payload = source_path.read_bytes()
    try:
        result = encode_payload(payload, output_path)
    except Exception as exc:
        print(f"\nEncoding denied: {exc}")
        pause()
        return
    show_encode_result(output_path, payload, result)


def show_encode_result(output_path: str, payload: bytes, result: dict):
    image = result["image"]
    print(f"\nSaved: {output_path}")
    print(f"Original payload: {len(payload):,} bytes")
    print(f"Stored payload: {result['stored_bytes']:,} bytes")
    print(f"Compression used: {'yes' if result['compressed'] else 'no'}")
    print(f"Protected incompressible capacity: {max_payload_bytes():,} bytes")
    print(f"Content grid: {result['content_side']} x {result['content_side']} modules")
    print(f"Rendered module size: {result['block_size']}px")
    print(f"Final image: {image.width} x {image.height}px")
    pause()


def output_decoded_payload(payload: bytes):
    print("\nSelect output type:")
    print("1) Show as text")
    print("2) Save as file")
    choice = input("> ").strip()
    if choice == "1":
        print("\n--- DECODED TEXT ---\n")
        print(payload.decode("utf-8", errors="replace"))
        print("\n--------------------")
    elif choice == "2":
        output_path = input("Output file path:\n> ").strip()
        Path(output_path).write_bytes(payload)
        print(f"\nPayload saved to: {output_path}")
    else:
        print("Invalid choice.")


def decode_exact_image():
    clear_screen()
    print(f"=== EXACT GDC v{VERSION} IMAGE DECODER ===\n")
    path = Path(input("Exact GDC PNG path:\n> ").strip())
    if not path.is_file():
        print("File not found.")
        pause()
        return
    try:
        payload = image_to_payload(path)
    except Exception as exc:
        print(f"\nFailed to decode: {exc}")
        pause()
        return
    output_decoded_payload(payload)
    pause()


def decode_photo():
    clear_screen()
    print(f"=== CAMERA PHOTO -> GDC v{VERSION} DECODER ===\n")
    print("For best results, keep all three finder patterns visible and avoid glare.\n")
    path = Path(input("Photo path:\n> ").strip())
    if not path.is_file():
        print("File not found.")
        pause()
        return
    try:
        payload = photo_to_payload(path)
    except Exception as exc:
        print(f"\nPhoto decode failed: {exc}")
        pause()
        return
    output_decoded_payload(payload)
    pause()


def show_capacity():
    clear_screen()
    print(f"=== GDC v{VERSION} QR-FIRST FIELD PROFILE ===\n")
    print(f"QR carrier version: {QR_VERSION}")
    print(f"Content grid: {CONTENT_SIDE} x {CONTENT_SIDE} modules")
    print(f"Finder patterns: 3 x {MARKER_SIZE_BLOCKS} x {MARKER_SIZE_BLOCKS} modules")
    print(
        f"Hidden calibration samples: 3 channels x {LEVEL_COUNT} dark levels"
    )
    print(f"Alignment patterns: {len(alignment_pattern_centers(CONTENT_SIDE))}")
    print(f"Rendered module size: {BLOCK_SIZE}px")
    print(f"Final PNG: {fixed_image_size_px()} x {fixed_image_size_px()}px")
    print(f"Color states: {LEVEL_COUNT ** 3:,}")
    print(f"Raw bits per module: {BITS_PER_CELL}")
    print(f"Raw encoded stream: {protected_stream_capacity_bytes():,} bytes")
    print(f"Reed-Solomon parity: {RS_PARITY_BYTES} bytes per {RS_BLOCK_SIZE}-byte block")
    print(f"Protected incompressible payload: {max_payload_bytes():,} bytes")
    print("Compressible text and documents may substantially exceed this limit.")
    print(f"At 3 inches wide: each module is {(3 * 25.4 / fixed_side_blocks()):.2f} mm")
    print(f"At 4 inches wide: each module is {(4 * 25.4 / fixed_side_blocks()):.2f} mm")
    pause()


def main_menu():
    while True:
        clear_screen()
        print("==============================================")
        print(f"                 GDC TOOL v{VERSION}")
        print(" Field-size modules + color error correction")
        print("==============================================")
        print("1) Encode text")
        print("2) Encode a file")
        print("3) Decode exact GDC PNG")
        print("4) Decode camera photo")
        print("5) Show capacity")
        print("6) Exit")
        print("==============================================")

        choice = input("> ").strip()
        if choice == "1":
            encode_text()
        elif choice == "2":
            encode_file()
        elif choice == "3":
            decode_exact_image()
        elif choice == "4":
            decode_photo()
        elif choice == "5":
            show_capacity()
        elif choice == "6":
            clear_screen()
            print("Goodbye")
            return
        else:
            print("Invalid choice.")
            pause()


if __name__ == "__main__":
    main_menu()
