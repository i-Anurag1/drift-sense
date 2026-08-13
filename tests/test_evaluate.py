import unittest
import numpy as np

from evaluate import error_class


class TestEvaluateHelpers(unittest.TestCase):
    def test_error_class_good(self):
        self.assertEqual(error_class(5.0), "good")
        self.assertEqual(error_class(14.9), "good")

    def test_error_class_warn(self):
        self.assertEqual(error_class(15.0), "warn")
        self.assertEqual(error_class(39.9), "warn")

    def test_error_class_bad(self):
        self.assertEqual(error_class(40.0), "bad")
        self.assertEqual(error_class(500.0), "bad")

    def test_pixel_distance_matches_numpy_hypot(self):
        gt = (100.0, 200.0)
        pred = (103.0, 204.0)
        expected = 5.0  # 3-4-5 triangle
        actual = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
