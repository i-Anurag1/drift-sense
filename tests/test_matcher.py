import unittest
import numpy as np
import cv2

from drift_sense.matcher import locate_reference, locate_reference_naive, MatchResult


def make_synthetic_pair(site_x=400, site_y=300, search_size=800, ref_size=200):
    canvas = np.full((search_size, search_size), 30, dtype=np.uint8)
    for x in range(0, search_size, 40):
        cv2.line(canvas, (x, 0), (x, search_size), 200, 3)
    for y in range(0, search_size, 40):
        cv2.line(canvas, (0, y), (search_size, y), 200, 3)
    cv2.circle(canvas, (site_x, site_y), 12, 255, -1)

    half = ref_size // 2
    reference = canvas[site_y - half:site_y + half, site_x - half:site_x + half].copy()
    return reference, canvas


class TestMatcher(unittest.TestCase):
    def test_returns_match_result(self):
        ref, search = make_synthetic_pair()
        result = locate_reference(ref, search, scale_steps=4, angle_steps=1)
        self.assertIsInstance(result, MatchResult)

    def test_localizes_near_known_site_no_scale_change(self):
        ref, search = make_synthetic_pair(site_x=400, site_y=300)
        result = locate_reference(ref, search, scale_range=(0.9, 1.1), scale_steps=3, angle_steps=1)
        err = np.hypot(result.x - 400, result.y - 300)
        self.assertLess(err, 25, f"expected near (400,300), got ({result.x:.1f},{result.y:.1f})")

    def test_naive_matcher_also_returns_result(self):
        ref, search = make_synthetic_pair()
        result = locate_reference_naive(ref, search, scale_range=(0.9, 1.1), scale_steps=3, angle_steps=1)
        self.assertIsInstance(result, MatchResult)

    def test_handles_grayscale_and_bgr_input(self):
        ref, search = make_synthetic_pair()
        search_bgr = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        result = locate_reference(ref, search_bgr, scale_range=(0.9, 1.1), scale_steps=3, angle_steps=1)
        self.assertIsInstance(result, MatchResult)

    def test_no_crash_on_tiny_reference(self):
        ref = np.full((10, 10), 128, dtype=np.uint8)
        search = np.full((200, 200), 100, dtype=np.uint8)
        result = locate_reference(ref, search, scale_steps=3, angle_steps=1)
        self.assertIsInstance(result, MatchResult)

    def test_result_coordinates_within_search_bounds(self):
        ref, search = make_synthetic_pair()
        result = locate_reference(ref, search, scale_range=(0.9, 1.1), scale_steps=3, angle_steps=1)
        h, w = search.shape[:2]
        self.assertGreaterEqual(result.x, 0)
        self.assertGreaterEqual(result.y, 0)
        self.assertLessEqual(result.x, w)
        self.assertLessEqual(result.y, h)


if __name__ == "__main__":
    unittest.main()
