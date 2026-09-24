import os
import unittest

import cv2
import numpy as np

import gdc


def camera(image, width=900, blur=0.8, quality=80, shading=0.3, rotate=0):
    """Simulated phone capture: rotation, perspective, blur, uneven light, JPEG."""
    bgr = np.rot90(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), rotate)
    bgr = cv2.resize(bgr, (width, width), interpolation=cv2.INTER_AREA)
    w = width
    src = np.float32([[0, 0], [w, 0], [w, w], [0, w]])
    dst = np.float32([[.06 * w, .05 * w], [.93 * w, .02 * w], [.97 * w, .95 * w], [.03 * w, .98 * w]])
    bgr = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (w, w), borderValue=(190, 185, 180))
    bgr = cv2.GaussianBlur(bgr, (0, 0), blur)
    light = np.linspace(1 - shading, 1, w)[None, :, None] * np.linspace(0.9, 1, w)[:, None, None]
    bgr = np.clip(bgr * light, 0, 255).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)


class GDCTest(unittest.TestCase):
    def test_exact_round_trip_at_full_capacity(self):
        for profile in [(1, 4, "gray", "L"), (10, 8, "gray", "M"), (10, 4, "rgb", "Q"),
                        (27, 4, "gray", "M"), (40, 16, "gray", "H")]:
            with self.subTest(profile=profile):
                payload = os.urandom(gdc.capacity(*profile))
                self.assertEqual(gdc.decode(gdc.encode(payload, *profile)), payload)

    def test_denser_than_same_size_qr(self):
        for version in range(1, 41):
            for ecc in gdc.ECC:
                qr = gdc.qr_capacity(version, ecc)
                self.assertGreaterEqual(gdc.capacity(version, 4, "gray", ecc), 1.7 * qr)
                self.assertGreaterEqual(gdc.capacity(version, 8, "gray", ecc), 2.5 * qr)
        self.assertGreaterEqual(gdc.capacity(40, 4, "gray", "L"), 1.99 * gdc.qr_capacity(40, "L"))

    def test_auto_version_and_compression(self):
        self.assertEqual(gdc.smallest_version(os.urandom(gdc.capacity(5)), 4, "gray", "M"), 5)
        text = b"GDC multilevel QR\n" * 400
        self.assertGreater(len(text), gdc.capacity(10))
        self.assertEqual(gdc.decode(gdc.encode(text, version=10)), text)

    def test_standard_qr_detector_sees_a_qr_code(self):
        image = gdc.encode(b"looks like an everyday QR code", version=10)
        found, _ = cv2.QRCodeDetector().detect(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        self.assertTrue(found)

    def test_camera_capture(self):
        for profile, rotate in [((10, 4, "gray", "M"), 0), ((6, 4, "gray", "M"), 1),
                                ((20, 4, "gray", "M"), 2), ((10, 4, "rgb", "M"), 3)]:
            with self.subTest(profile=profile):
                payload = os.urandom(gdc.capacity(*profile))
                photo = camera(gdc.encode(payload, *profile), rotate=rotate)
                self.assertEqual(gdc.decode(photo), payload)

    def test_reed_solomon_survives_covered_patch(self):
        payload = os.urandom(gdc.capacity(10, ecc="H"))
        pixels = np.asarray(gdc.encode(payload, version=10, ecc="H", module_px=10)).copy()
        start = (gdc.QUIET + 20) * 10
        pixels[start:start + 60, start:start + 60] = 255  # 6x6-module sticker
        self.assertEqual(gdc.decode(pixels[..., ::-1].copy()), payload)


if __name__ == "__main__":
    unittest.main()
