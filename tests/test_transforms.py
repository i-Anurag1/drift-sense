import unittest
import numpy as np
import cv2

from drift_sense.degrade import apply_rotation, apply_blur, apply_edge_brightening, apply_sensor_noise


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.img = (np.random.default_rng(0).random((100, 100)) * 255).astype(np.uint8)

    def test_rotation_identity_zero_degrees(self):
        out = apply_rotation(self.img, 0)
        np.testing.assert_array_equal(out, self.img)

    def test_rotation_preserves_shape(self):
        out = apply_rotation(self.img, 5.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, self.img.dtype)

    def test_rotation_no_black_border_artifacts(self):
        out = apply_rotation(self.img, 4.0)
        corner = out[:5, :5]
        self.assertGreater(corner.mean(), 5, "corners should use reflected content, not black fill")

    def test_blur_zero_sigma_is_noop(self):
        out = apply_blur(self.img, 0)
        np.testing.assert_array_equal(out, self.img)

    def test_blur_reduces_variance(self):
        out = apply_blur(self.img, 2.0)
        self.assertLess(out.astype(float).var(), self.img.astype(float).var())

    def test_edge_brightening_output_valid_range(self):
        out = apply_edge_brightening(self.img)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue((out >= 0).all() and (out <= 255).all())

    def test_edge_brightening_no_crash_on_uniform_image(self):
        flat = np.full((50, 50), 128, dtype=np.uint8)
        out = apply_edge_brightening(flat)
        self.assertEqual(out.shape, flat.shape)

    def test_sensor_noise_changes_image(self):
        rng = np.random.default_rng(1)
        out = apply_sensor_noise(self.img, rng=rng)
        self.assertFalse(np.array_equal(out, self.img))
        self.assertEqual(out.dtype, np.uint8)

    def test_sensor_noise_independent_across_calls(self):
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        out1 = apply_sensor_noise(self.img, rng=rng1)
        out2 = apply_sensor_noise(self.img, rng=rng2)
        self.assertFalse(np.array_equal(out1, out2),
                          "independently-seeded noise should differ between reference/search")


if __name__ == "__main__":
    unittest.main()
