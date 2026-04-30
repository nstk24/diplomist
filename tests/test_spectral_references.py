from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from raman_webapp.analysis import detect_spectral_peaks
from raman_webapp.data import ProcessedDataset, load_reference_band_library
from raman_webapp.spectral_references import (
    add_reference_comparison,
    compare_peak_to_reference,
    compute_reference_peak_statistics,
    evaluate_expected_sers_bands,
    extract_expected_peak_intensities,
)
from raman_webapp.spectroscopy import analyze_spectroscopy_peaks


def _gaussian(x: np.ndarray, center: float, amplitude: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


class SpectralReferencesTest(unittest.TestCase):
    def test_z_score_is_computed_correctly(self) -> None:
        comparison = compare_peak_to_reference(1.8, 1.0, 0.4)
        self.assertAlmostEqual(float(comparison["z_score"]), 2.0)
        self.assertIn("выше контрольной группы", str(comparison["deviation_label"]))

    def test_zero_std_returns_russian_warning(self) -> None:
        comparison = compare_peak_to_reference(1.0, 1.0, 0.0)
        self.assertTrue(np.isnan(comparison["z_score"]))
        self.assertIn("Недостаточно данных", str(comparison["deviation_label"]))
        self.assertIn("z-score", str(comparison["reference_warning"]))

    def test_range_band_is_processed_as_range(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        spectra = np.vstack(
            [
                _gaussian(wavenumber, 1330.0, amplitude=1.0, sigma=3.0),
                _gaussian(wavenumber, 1334.0, amplitude=1.2, sigma=3.0),
                _gaussian(wavenumber, 1336.0, amplitude=0.8, sigma=3.0),
                _gaussian(wavenumber, 1332.0, amplitude=1.4, sigma=3.0),
            ]
        )
        dataset = ProcessedDataset(
            X=spectra.copy(),
            y=np.array([0, 0, 1, 1], dtype=int),
            wavenumber=wavenumber,
            snr=np.ones(4, dtype=float),
            X_snr=spectra.copy(),
            sample_names=np.array(["h1", "h2", "d1", "d2"], dtype=object),
        )
        stats_df = compute_reference_peak_statistics(
            dataset=dataset,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            intensity_basis="baseline_corrected",
        )
        range_row = stats_df.loc[stats_df["band_id"] == "sers_1303_1359"].iloc[0]
        self.assertGreater(float(range_row["mean_healthy"]), 0.0)
        self.assertGreater(float(range_row["mean_disease"]), 0.0)

    def test_point_peak_uses_tolerance_window(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        spectra = np.vstack(
            [
                _gaussian(wavenumber, 536.0, amplitude=1.0, sigma=2.0),
                _gaussian(wavenumber, 538.0, amplitude=1.1, sigma=2.0),
                _gaussian(wavenumber, 533.0, amplitude=0.8, sigma=2.0),
                _gaussian(wavenumber, 540.0, amplitude=1.2, sigma=2.0),
            ]
        )
        dataset = ProcessedDataset(
            X=spectra.copy(),
            y=np.array([0, 0, 1, 1], dtype=int),
            wavenumber=wavenumber,
            snr=np.ones(4, dtype=float),
            X_snr=spectra.copy(),
            sample_names=np.array(["h1", "h2", "d1", "d2"], dtype=object),
        )
        stats_df = compute_reference_peak_statistics(
            dataset=dataset,
            reference_library_df=load_reference_band_library(spectroscopy_mode="sers"),
            spectroscopy_mode="sers",
            intensity_basis="baseline_corrected",
        )
        point_row = stats_df.loc[stats_df["band_id"] == "sers_534"].iloc[0]
        self.assertGreater(float(point_row["mean_healthy"]), 0.0)

    def test_mean_aggregation_differs_from_max_for_wide_range(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        spectra = np.vstack(
            [
                _gaussian(wavenumber, 1330.0, amplitude=2.0, sigma=2.0) + 0.2,
                _gaussian(wavenumber, 1331.0, amplitude=1.8, sigma=2.0) + 0.2,
            ]
        )
        reference_library_df = load_reference_band_library(spectroscopy_mode="sers")
        max_df = extract_expected_peak_intensities(
            X=spectra,
            wavenumber=wavenumber,
            reference_library_df=reference_library_df,
            spectroscopy_mode="sers",
            aggregation_mode="max",
        )
        mean_df = extract_expected_peak_intensities(
            X=spectra,
            wavenumber=wavenumber,
            reference_library_df=reference_library_df,
            spectroscopy_mode="sers",
            aggregation_mode="mean",
        )
        max_value = float(max_df.loc[max_df["band_id"] == "sers_1303_1359", "peak_intensity"].iloc[0])
        mean_value = float(mean_df.loc[mean_df["band_id"] == "sers_1303_1359", "peak_intensity"].iloc[0])
        self.assertGreater(max_value, mean_value)

    def test_deviation_label_is_russian_in_add_reference_comparison(self) -> None:
        peak_df = pd.DataFrame(
            [
                {
                    "band_id": "sers_534",
                    "group": "липидные признаки",
                    "assignment": "эфиры холестерина",
                    "expected_position_cm": "534",
                    "measured_shift_cm": 534.0,
                    "intensity": 1.5,
                    "reliability": "средняя",
                }
            ]
        )
        reference_stats_df = pd.DataFrame(
            [
                {
                    "band_id": "sers_534",
                    "reference_basis": "baseline_corrected",
                    "mean_healthy": 1.0,
                    "std_healthy": 0.2,
                    "median_healthy": 1.0,
                    "iqr_healthy": 0.1,
                    "mean_disease": 1.3,
                    "std_disease": 0.2,
                    "effect_size": 1.0,
                    "n_healthy": 4,
                    "n_disease": 4,
                }
            ]
        )
        compared_df = add_reference_comparison(peak_df, reference_stats_df)
        self.assertIn("контрольной группы", str(compared_df.iloc[0]["deviation_label"]))

    def test_background_sers_peaks_are_not_candidate_pathology(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        intensity = (
            _gaussian(wavenumber, 638.0, amplitude=1.0, sigma=2.0)
            + _gaussian(wavenumber, 725.0, amplitude=1.0, sigma=2.0)
            + _gaussian(wavenumber, 1659.0, amplitude=1.0, sigma=2.0)
        )
        peaks_df = pd.DataFrame(
            [
                {"rank_by_prominence": 1, "wavenumber_cm-1": 638.0, "intensity": 1.0, "prominence": 1.0, "width_cm-1": 4.0},
                {"rank_by_prominence": 2, "wavenumber_cm-1": 725.0, "intensity": 1.0, "prominence": 1.0, "width_cm-1": 4.0},
                {"rank_by_prominence": 3, "wavenumber_cm-1": 1659.0, "intensity": 1.0, "prominence": 1.0, "width_cm-1": 4.0},
            ]
        )
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

    def test_expected_sers_zones_do_not_depend_on_top_k(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        intensity = np.zeros_like(wavenumber)
        high_noise_centers = list(np.arange(510.0, 610.0, 10.0))
        for idx, center in enumerate(high_noise_centers):
            intensity += _gaussian(wavenumber, center, amplitude=2.0 - idx * 0.05, sigma=1.5)
        intensity += _gaussian(wavenumber, 1481.0, amplitude=0.7, sigma=2.0)

        peaks_df = detect_spectral_peaks(
            wavenumber,
            intensity,
            prominence=0.05,
            min_distance_cm=6.0,
            min_width_cm=2.0,
            top_k=3,
        ).peaks_df
        self.assertNotIn(1481.0, peaks_df["wavenumber_cm-1"].round(0).tolist())

        spectra = np.vstack(
            [
                _gaussian(wavenumber, 1481.0, amplitude=0.5, sigma=2.0),
                _gaussian(wavenumber, 1481.0, amplitude=0.6, sigma=2.0),
                _gaussian(wavenumber, 1481.0, amplitude=0.8, sigma=2.0),
                _gaussian(wavenumber, 1481.0, amplitude=0.9, sigma=2.0),
            ]
        )
        dataset = ProcessedDataset(
            X=spectra.copy(),
            y=np.array([0, 0, 1, 1], dtype=int),
            wavenumber=wavenumber,
            snr=np.ones(4, dtype=float),
            X_snr=spectra.copy(),
            sample_names=np.array(["h1", "h2", "d1", "d2"], dtype=object),
        )
        reference_library_df = load_reference_band_library(spectroscopy_mode="sers")
        stats_df = compute_reference_peak_statistics(
            dataset=dataset,
            reference_library_df=reference_library_df,
            spectroscopy_mode="sers",
            intensity_basis="baseline_corrected",
        )
        expected_df = evaluate_expected_sers_bands(
            patient_wavenumber=wavenumber,
            patient_intensity=intensity,
            reference_library_df=reference_library_df,
            reference_stats_df=stats_df,
            aggregation_mode="max",
        )
        target_row = expected_df.loc[expected_df["band_id"] == "sers_1481"].iloc[0]
        self.assertEqual(str(target_row["peak_role"]), "candidate")
        self.assertEqual(str(target_row["status"]), "обнаружен")
        self.assertAlmostEqual(float(target_row["measured_shift_cm"]), 1481.0, delta=2.0)

    def test_missing_healthy_group_returns_warning_without_crash(self) -> None:
        wavenumber = np.arange(500.0, 1701.0, 1.0)
        spectra = np.vstack(
            [
                _gaussian(wavenumber, 534.0, amplitude=1.1, sigma=2.0),
                _gaussian(wavenumber, 536.0, amplitude=1.2, sigma=2.0),
            ]
        )
        dataset = ProcessedDataset(
            X=spectra.copy(),
            y=np.array([1, 1], dtype=int),
            wavenumber=wavenumber,
            snr=np.ones(2, dtype=float),
            X_snr=spectra.copy(),
            sample_names=np.array(["d1", "d2"], dtype=object),
        )
        reference_library_df = load_reference_band_library(spectroscopy_mode="sers")
        stats_df = compute_reference_peak_statistics(
            dataset=dataset,
            reference_library_df=reference_library_df,
            spectroscopy_mode="sers",
            intensity_basis="baseline_corrected",
        )
        peak_df = pd.DataFrame(
            [
                {
                    "band_id": "sers_534",
                    "group": "липидные признаки",
                    "assignment": "эфиры холестерина",
                    "expected_position_cm": "534",
                    "measured_shift_cm": 534.0,
                    "intensity": 1.3,
                    "reliability": "средняя",
                }
            ]
        )
        compared_df = add_reference_comparison(peak_df, stats_df)
        self.assertTrue(np.isnan(compared_df.iloc[0]["z_score"]))
        self.assertIn("Недостаточно данных", str(compared_df.iloc[0]["deviation_label"]))


if __name__ == "__main__":
    unittest.main()
