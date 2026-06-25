import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import gdc_v4 as gdc


class GDCTest(unittest.TestCase):
    def test_real_world_field_profile(self):
        # This profile deliberately prioritizes camera and print tolerance over
        # the documentation's theoretical maximum density.
        self.assertGreaterEqual(gdc.max_payload_bytes(), 2500)
        self.assertEqual(gdc.CONTENT_SIDE, 73)
        self.assertGreaterEqual(gdc.BLOCK_SIZE, 48)
        self.assertEqual(gdc.LEVEL_COUNT, 4)
        self.assertGreaterEqual(gdc.RS_PARITY_BYTES, 32)

    def test_reed_solomon_recovers_interleaved_damage(self):
        payload = os.urandom(2048)
        stream = bytearray(gdc.build_stream(payload))
        block_count = gdc.rs_block_count()

        # Damage eight transmitted positions belonging to one RS block. This
        # remains below the twelve-byte correction limit without making every
        # block execute the expensive error-location path.
        damaged_block = 3
        for byte_index in range(8):
            stream[byte_index * block_count + damaged_block] ^= 0x5A

        self.assertEqual(gdc.parse_stream(bytes(stream)), payload)

    def test_exact_image_round_trip(self):
        payload = os.urandom(2048)
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
        payload = os.urandom(1800)
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
