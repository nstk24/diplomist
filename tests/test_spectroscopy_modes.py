from __future__ import annotations

import unittest

import numpy as np

from raman_webapp.analysis import detect_spectral_peaks
from raman_webapp.data import load_reference_band_library
from raman_webapp.spectroscopy import (
    analyze_spectroscopy_peaks,
    get_spectroscopy_mode_label,
    normalize_spectroscopy_mode,
)


def _gaussian(x: np.ndarray, center: float, amplitude: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _build_spectrum(centers: list[float], amplitudes: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    wavenumber = np.arange(500.0, 1701.0, 1.0)
    intensity = np.zeros_like(wavenumber)
    amplitudes = amplitudes or [1.0] * len(centers)
    for center, amplitude in zip(centers, amplitudes):
        intensity += _gaussian(wavenumber, center=center, amplitude=amplitude, sigma=2.0)
    intensity += 0.01 * np.sin(wavenumber / 25.0)
    return wavenumber, intensity


def _detect_test_peaks(wavenumber: np.ndarray, intensity: np.ndarray):
    return detect_spectral_peaks(
        wavenumber,
        intensity,
        prominence=0.05,
        min_distance_cm=8.0,
        min_width_cm=2.0,
        top_k=30,
    ).peaks_df


class SpectroscopyModesTest(unittest.TestCase):
    def test_default_mode_is_raman(self) -> None:
        self.assertEqual(normalize_spectroscopy_mode(None), "raman")
        self.assertEqual(get_spectroscopy_mode_label(None), "Рамановская спектроскопия")

    def test_mode_specific_libraries_are_different(self) -> None:
        raman_library = load_reference_band_library(spectroscopy_mode="raman")
        sers_library = load_reference_band_library(spectroscopy_mode="sers")

        self.assertIn("priority", raman_library.columns)
        self.assertIn("peak_role", sers_library.columns)
        self.assertNotIn("peak_role", raman_library.columns)
        self.assertNotIn("priority", sers_library.columns)

    def test_raman_mode_uses_raman_library(self) -> None:
        wavenumber, intensity = _build_spectrum([1001.0, 1154.0, 1515.0], [1.2, 1.0, 0.9])
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="raman"),
            spectroscopy_mode="raman",
            tolerance_cm=5.0,
        )

        self.assertEqual(result.spectroscopy_mode, "raman")
        self.assertGreaterEqual(result.candidate_peak_count, 1)
        self.assertIn("Фенилаланин", " ".join(result.matched_peaks_df["label"].tolist()))

    def test_sers_candidate_peaks_are_detected(self) -> None:
        centers = [534.0, 697.0, 744.0, 835.0, 927.0, 988.0, 1221.0, 1330.0, 1481.0]
        wavenumber, intensity = _build_spectrum(centers)
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            tolerance_cm=10.0,
        )

        detected_ids = set(result.detected_candidate_peaks_df["band_id"].tolist())
        expected_ids = {
            "sers_534",
            "sers_697",
            "sers_744",
            "sers_835",
            "sers_927",
            "sers_988",
            "sers_1221",
            "sers_1303_1359",
            "sers_1481",
        }
        self.assertTrue(expected_ids.issubset(detected_ids))

    def test_sers_range_band_is_processed_as_range(self) -> None:
        wavenumber, intensity = _build_spectrum([1332.0], [1.4])
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            tolerance_cm=10.0,
        )

        range_row = result.matched_peaks_df.loc[result.matched_peaks_df["band_id"] == "sers_1303_1359"].iloc[0]
        self.assertTrue(bool(range_row["peak_present"]))
        self.assertAlmostEqual(float(range_row["measured_shift_cm"]), 1332.0, delta=2.0)

    def test_background_peaks_do_not_increase_cvd_score(self) -> None:
        wavenumber, intensity = _build_spectrum([638.0, 725.0, 1659.0], [1.0, 1.1, 0.9])
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            tolerance_cm=10.0,
        )

        self.assertEqual(result.candidate_peak_count, 0)
        self.assertEqual(result.background_peak_count, 3)
        self.assertEqual(result.sers_cvd_score, 0.0)

    def test_low_reliability_peaks_do_not_overstate_score(self) -> None:
        wavenumber, intensity = _build_spectrum([1541.0, 1588.0], [1.0, 0.95])
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            tolerance_cm=10.0,
        )

        self.assertEqual(result.candidate_peak_count, 2)
        self.assertLess(result.sers_cvd_score or 0.0, 1.0)
        self.assertEqual(result.sers_cvd_pattern_level, "слабовыраженный")

    def test_sers_report_contains_required_warning_in_russian(self) -> None:
        wavenumber, intensity = _build_spectrum([534.0, 697.0, 744.0], [1.0, 0.9, 0.95])
        peaks_df = _detect_test_peaks(wavenumber, intensity)
        result = analyze_spectroscopy_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            tolerance_cm=10.0,
        )

        self.assertTrue(any("не являются самостоятельными диагностическими маркерами" in warning for warning in result.warnings))
        self.assertIn("сердечно-сосудистой патологией", result.interpretation)
        self.assertIn("Полученный результат следует рассматривать", result.interpretation)


if __name__ == "__main__":
    unittest.main()
