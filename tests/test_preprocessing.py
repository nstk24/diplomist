from __future__ import annotations

import unittest

import numpy as np

from raman_webapp.preprocessing import batch_smooth_savgol, smooth_savgol


class PreprocessingTest(unittest.TestCase):
    def test_savgol_preserves_shape(self) -> None:
        x = np.linspace(0.0, 1.0, 51)
        smoothed = smooth_savgol(x)
        self.assertEqual(smoothed.shape, x.shape)

    def test_batch_savgol_preserves_matrix_shape(self) -> None:
        X = np.vstack([np.linspace(0.0, 1.0, 51), np.linspace(1.0, 0.0, 51)])
        smoothed = batch_smooth_savgol(X)
        self.assertEqual(smoothed.shape, X.shape)

    def test_savgol_reduces_high_frequency_noise(self) -> None:
        x = np.linspace(0.0, 2.0 * np.pi, 201)
        clean = np.sin(x)
        noisy = clean + 0.15 * np.sin(25.0 * x)
        smoothed = smooth_savgol(noisy, window_length=11, polyorder=3)

        noisy_error = float(np.mean((noisy - clean) ** 2))
        smoothed_error = float(np.mean((smoothed - clean) ** 2))
        self.assertLess(smoothed_error, noisy_error)


if __name__ == "__main__":
    unittest.main()
