import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import gdc_v10 as gdc


class GDCTest(unittest.TestCase):
    def test_real_world_field_profile(self):
        self.assertGreaterEqual(gdc.max_payload_bytes(), 850)
        self.assertEqual(gdc.QR_VERSION, 10)
        self.assertEqual(gdc.CONTENT_SIDE, 57)
        self.assertGreaterEqual(gdc.BLOCK_SIZE, 60)
        self.assertEqual(gdc.LEVEL_COUNT, 4)
        self.assertGreaterEqual(gdc.RS_PARITY_BYTES, 32)

    def test_colored_symbol_preserves_exact_qr_carrier(self):
        image, *_ = gdc.stream_to_image(gdc.build_stream(b"QR carrier check"))
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
        pixels = core.load()
        thresholded = tuple(
            tuple((sum(pixels[col, row]) / 3) < 128 for col in range(gdc.CONTENT_SIDE))
            for row in range(gdc.CONTENT_SIDE)
        )
        self.assertEqual(thresholded, gdc.qr_carrier_matrix())
        for row in range(gdc.CONTENT_SIDE):
            for col in range(gdc.CONTENT_SIDE):
                if not gdc.qr_carrier_matrix()[row][col]:
                    self.assertEqual(pixels[col, row], (255, 255, 255))

    def test_standard_qr_decoder_reads_carrier(self):
        image, *_ = gdc.stream_to_image(gdc.build_stream(b"GDC payload"))
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        scanner_size = 1000
        scanner_image = cv2.resize(
            bgr,
            (scanner_size, scanner_size),
            interpolation=cv2.INTER_AREA,
        )
        value, points, _ = cv2.QRCodeDetector().detectAndDecode(scanner_image)
        self.assertIsNotNone(points)
        self.assertEqual(value, "GDC-V10-COLOR-CARRIER")

    def test_reed_solomon_recovers_interleaved_damage(self):
        payload = os.urandom(700)
        stream = bytearray(gdc.build_stream(payload))
        block_count = gdc.rs_block_count()

        # Damage eight transmitted positions belonging to one RS block. This
        # remains below the sixteen-byte correction limit without making every
        # block execute the expensive error-location path.
        damaged_block = 3
        for byte_index in range(8):
            stream[byte_index * block_count + damaged_block] ^= 0x5A

        self.assertEqual(gdc.parse_stream(bytes(stream)), payload)

    def test_exact_image_round_trip(self):
        payload = os.urandom(700)
        image, *_ = gdc.stream_to_image(gdc.build_stream(payload))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.png"
            image.save(path)
            self.assertEqual(gdc.image_to_payload(path), payload)

    def test_compression_exceeds_incompressible_limit(self):
        payload = (b"GDC camera-readable dense storage\n" * 5000)
        self.assertGreater(len(payload), gdc.max_payload_bytes())
        self.assertEqual(gdc.parse_stream(gdc.build_stream(payload)), payload)

    def test_camera_path_handles_perspective_and_jpeg(self):
        payload = b"camera perspective validation" * 200
        image, *_ = gdc.stream_to_image(gdc.build_stream(payload))
        source = np.float32(
            [
                [0, 0],
                [image.width - 1, 0],
                [image.width - 1, image.height - 1],
                [0, image.height - 1],
            ]
        )
        size = gdc.fixed_image_size_px()
        destination = np.float32(
            [
                [size * 0.06, size * 0.045],
                [size * 0.935, size * 0.02],
                [size * 0.968, size * 0.955],
                [size * 0.03, size * 0.982],
            ]
        )
        transform = cv2.getPerspectiveTransform(source, destination)
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        warped = cv2.warpPerspective(
            bgr,
            transform,
            (gdc.fixed_image_size_px(), gdc.fixed_image_size_px()),
            borderValue=(210, 210, 210),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perspective.jpg"
            cv2.imwrite(str(path), warped, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.assertEqual(gdc.photo_to_payload(path), payload)

    def test_field_profile_handles_small_blurred_camera_image(self):
        payload = os.urandom(700)
        image, *_ = gdc.stream_to_image(gdc.build_stream(payload))
        camera_image = image.resize((800, 800), Image.Resampling.LANCZOS)
        camera_image = camera_image.filter(ImageFilter.GaussianBlur(0.8))
        camera_image = ImageEnhance.Brightness(camera_image).enhance(1.15)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small-camera-image.jpg"
            camera_image.save(path, "JPEG", quality=70)
            self.assertEqual(gdc.photo_to_payload(path), payload)


if __name__ == "__main__":
    unittest.main()
