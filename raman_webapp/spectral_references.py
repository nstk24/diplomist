from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import ProcessedDataset
from .spectroscopy import normalize_spectroscopy_mode


def extract_expected_peak_intensities(
    X: np.ndarray,
    wavenumber: np.ndarray,
    reference_library_df: pd.DataFrame,
    spectroscopy_mode: str = "raman",
    aggregation_mode: str = "max",
) -> pd.DataFrame:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    X = np.asarray(X, dtype=float)
    wavenumber = np.asarray(wavenumber, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix of spectra.")

    records: list[dict[str, Any]] = []
    for row_idx in range(X.shape[0]):
        spectrum = X[row_idx]
        for band in reference_library_df.to_dict(orient="records"):
            intensity = _extract_band_intensity(
                spectrum=spectrum,
                wavenumber=wavenumber,
                band=band,
                spectroscopy_mode=mode,
                aggregation_mode=aggregation_mode,
            )
            records.append(
                {
                    "sample_index": row_idx,
                    "band_id": band["band_id"],
                    "peak_intensity": intensity,
                }
            )
    return pd.DataFrame(records)


def compute_reference_peak_statistics(
    dataset: ProcessedDataset,
    reference_library_df: pd.DataFrame,
    spectroscopy_mode: str = "raman",
    intensity_basis: str = "baseline_corrected",
    aggregation_mode: str = "max",
) -> pd.DataFrame:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    X_source, basis_label = _select_reference_basis_matrix(dataset, intensity_basis)
    intensity_df = extract_expected_peak_intensities(
        X=X_source,
        wavenumber=dataset.wavenumber,
        reference_library_df=reference_library_df,
        spectroscopy_mode=mode,
        aggregation_mode=aggregation_mode,
    )
    intensity_df["label"] = dataset.y[intensity_df["sample_index"].to_numpy(dtype=int)]

    stats_rows: list[dict[str, Any]] = []
    for band in reference_library_df.to_dict(orient="records"):
        band_values = intensity_df[intensity_df["band_id"] == band["band_id"]]
        healthy_values = band_values.loc[band_values["label"] == 0, "peak_intensity"].to_numpy(dtype=float)
        disease_values = band_values.loc[band_values["label"] == 1, "peak_intensity"].to_numpy(dtype=float)
        stats_rows.append(
            {
                "band_id": band["band_id"],
                "reference_basis": basis_label,
                "aggregation_mode": normalize_aggregation_mode(aggregation_mode),
                "mean_healthy": _safe_mean(healthy_values),
                "std_healthy": _safe_std(healthy_values),
                "median_healthy": _safe_median(healthy_values),
                "iqr_healthy": _safe_iqr(healthy_values),
                "mean_disease": _safe_mean(disease_values),
                "std_disease": _safe_std(disease_values),
                "effect_size": _cohen_d(healthy_values, disease_values),
                "n_healthy": int(healthy_values.size),
                "n_disease": int(disease_values.size),
            }
        )
    return pd.DataFrame(stats_rows)


def compare_peak_to_reference(
    patient_reference_intensity: float | int | None,
    mean_healthy: float | int | None,
    std_healthy: float | int | None,
) -> dict[str, Any]:
    if patient_reference_intensity is None or pd.isna(patient_reference_intensity):
        return {
            "z_score": np.nan,
            "deviation_label": "Недостаточно данных для расчёта отклонения от контрольной группы",
            "reference_warning": "Не удалось определить интенсивность пика пациента для сравнения с контрольной группой.",
        }

    if mean_healthy is None or std_healthy is None or pd.isna(mean_healthy) or pd.isna(std_healthy) or float(std_healthy) <= 1e-12:
        return {
            "z_score": np.nan,
            "deviation_label": "Недостаточно данных для расчёта отклонения от контрольной группы",
            "reference_warning": "Недостаточно данных контрольной группы для устойчивого расчёта z-score.",
        }

    z_score = float((float(patient_reference_intensity) - float(mean_healthy)) / float(std_healthy))
    if -1.0 <= z_score <= 1.0:
        label = "в пределах условной спектральной нормы контрольной группы"
    elif 1.0 < z_score <= 2.0:
        label = "умеренно выше контрольной группы"
    elif z_score > 2.0:
        label = "выраженно выше контрольной группы"
    elif -2.0 <= z_score < -1.0:
        label = "умеренно ниже контрольной группы"
    else:
        label = "выраженно ниже контрольной группы"
    return {
        "z_score": z_score,
        "deviation_label": label,
        "reference_warning": "",
    }


def add_reference_comparison(
    peak_df: pd.DataFrame,
    reference_stats_df: pd.DataFrame,
    patient_intensity_col: str = "intensity",
    allow_reference_comparison: bool = True,
    patient_wavenumber: np.ndarray | None = None,
    patient_intensity: np.ndarray | None = None,
    reference_library_df: pd.DataFrame | None = None,
    spectroscopy_mode: str = "raman",
    aggregation_mode: str = "max",
) -> pd.DataFrame:
    if peak_df.empty:
        return peak_df.copy()

    merged_df = peak_df.merge(reference_stats_df, on="band_id", how="left")
    if (
        patient_wavenumber is not None
        and patient_intensity is not None
        and reference_library_df is not None
        and "band_id" in merged_df.columns
    ):
        reference_lookup = {
            str(row["band_id"]): row for row in reference_library_df.to_dict(orient="records") if "band_id" in row
        }
        merged_df["patient_reference_intensity"] = merged_df["band_id"].map(
            lambda band_id: _extract_band_intensity(
                spectrum=np.asarray(patient_intensity, dtype=float),
                wavenumber=np.asarray(patient_wavenumber, dtype=float),
                band=reference_lookup.get(str(band_id), {}),
                spectroscopy_mode=spectroscopy_mode,
                aggregation_mode=aggregation_mode,
            )
            if str(band_id) in reference_lookup
            else np.nan
        )
    else:
        merged_df["patient_reference_intensity"] = merged_df[patient_intensity_col]

    if not allow_reference_comparison:
        merged_df["z_score"] = np.nan
        merged_df["deviation_label"] = "Сравнение с контрольной группой недоступно для выбранной основы интенсивности"
        merged_df["reference_warning"] = (
            "Для исходной ненормализованной интенсивности сравнение с контрольной группой не выполняется. "
            "Используйте режим после baseline correction для сопоставимого расчёта."
        )
        return merged_df

    comparisons = [
        compare_peak_to_reference(
            patient_reference_intensity=row["patient_reference_intensity"],
            mean_healthy=row.get("mean_healthy"),
            std_healthy=row.get("std_healthy"),
        )
        for _, row in merged_df.iterrows()
    ]
    comparison_df = pd.DataFrame(comparisons)
    return pd.concat([merged_df.reset_index(drop=True), comparison_df.reset_index(drop=True)], axis=1)


def summarize_reference_comparison(
    candidate_df: pd.DataFrame,
    background_df: pd.DataFrame,
    spectroscopy_mode: str = "raman",
) -> str:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    candidate_count = int(len(candidate_df))
    background_count = int(len(background_df))
    if candidate_count == 0 and background_count == 0:
        return "Сопоставимые спектральные признаки для сравнения с контрольной группой не обнаружены."

    higher_count = int((candidate_df["deviation_label"].astype(str).str.contains("выше контрольной группы", na=False)).sum()) if not candidate_df.empty else 0
    lower_count = int((candidate_df["deviation_label"].astype(str).str.contains("ниже контрольной группы", na=False)).sum()) if not candidate_df.empty else 0
    normal_count = int((candidate_df["deviation_label"] == "в пределах условной спектральной нормы контрольной группы").sum()) if not candidate_df.empty else 0
    represented_groups = sorted(dict.fromkeys(candidate_df["group"].tolist())) if "group" in candidate_df.columns else []

    if mode == "sers":
        first_sentence = (
            f"Обнаружено {candidate_count} кандидатных SERS-признаков, ассоциированных с сердечно-сосудистой патологией."
        )
        if represented_groups:
            group_text = "Представлены группы: " + ", ".join(represented_groups) + "."
        else:
            group_text = "Кандидатные группы признаков не представлены."
        deviation_text = (
            f"Из них {higher_count} признака выше контрольной группы, {lower_count} ниже контрольной группы, "
            f"{normal_count} находятся в пределах условной спектральной нормы."
        )
        if background_count > 0:
            background_text = (
                "Также обнаружены фоновые признаки сыворотки, которые не являются самостоятельными маркерами "
                "сердечно-сосудистой патологии."
            )
        else:
            background_text = "Фоновые признаки сыворотки не выявлены."
        return " ".join([first_sentence, deviation_text, group_text, background_text])

    first_sentence = f"Для найденных Raman-признаков выполнено сравнение с контрольной группой текущего датасета."
    deviation_text = (
        f"Выше контрольной группы: {higher_count}, ниже контрольной группы: {lower_count}, "
        f"в пределах условной спектральной нормы: {normal_count}."
    )
    return " ".join([first_sentence, deviation_text])


def evaluate_expected_sers_bands(
    patient_wavenumber: np.ndarray,
    patient_intensity: np.ndarray,
    reference_library_df: pd.DataFrame,
    reference_stats_df: pd.DataFrame,
    aggregation_mode: str = "max",
) -> pd.DataFrame:
    mode = normalize_aggregation_mode(aggregation_mode)
    patient_wavenumber = np.asarray(patient_wavenumber, dtype=float)
    patient_intensity = np.asarray(patient_intensity, dtype=float)
    stats_lookup = {
        str(row["band_id"]): row for row in reference_stats_df.to_dict(orient="records") if "band_id" in row
    }
    rows: list[dict[str, Any]] = []
    for band in reference_library_df.to_dict(orient="records"):
        band_id = str(band["band_id"])
        low, high = _get_band_window(band, "sers")
        mask = (patient_wavenumber >= low) & (patient_wavenumber <= high)
        stats_row = stats_lookup.get(band_id, {})
        patient_reference_intensity = np.nan
        measured_shift_cm = np.nan
        peak_present = False
        if np.any(mask):
            region_values = np.asarray(patient_intensity[mask], dtype=float)
            region_wavenumber = np.asarray(patient_wavenumber[mask], dtype=float)
            patient_reference_intensity = _aggregate_region(region_values, region_wavenumber, mode)
            if region_values.size > 0:
                max_index = int(np.argmax(region_values))
                measured_shift_cm = float(region_wavenumber[max_index])
                peak_present = bool(np.nanmax(region_values) > 0)
        comparison = compare_peak_to_reference(
            patient_reference_intensity=patient_reference_intensity,
            mean_healthy=stats_row.get("mean_healthy"),
            std_healthy=stats_row.get("std_healthy"),
        )
        status_label = "обнаружен" if peak_present else "не обнаружен"
        if pd.isna(patient_reference_intensity):
            status_label = "недостаточно данных"
        rows.append(
            {
                "band_id": band_id,
                "peak_role": band.get("peak_role", "candidate"),
                "expected_position_cm": _format_expected_position(band),
                "expected_peak_cm": float(band["shift_cm"]) if pd.notna(band.get("shift_cm")) else np.nan,
                "measured_shift_cm": measured_shift_cm,
                "status": status_label,
                "patient_reference_intensity": patient_reference_intensity,
                "mean_healthy": stats_row.get("mean_healthy", np.nan),
                "std_healthy": stats_row.get("std_healthy", np.nan),
                "median_healthy": stats_row.get("median_healthy", np.nan),
                "iqr_healthy": stats_row.get("iqr_healthy", np.nan),
                "mean_disease": stats_row.get("mean_disease", np.nan),
                "std_disease": stats_row.get("std_disease", np.nan),
                "effect_size": stats_row.get("effect_size", np.nan),
                "deviation_label": comparison["deviation_label"],
                "z_score": comparison["z_score"],
                "reference_warning": comparison["reference_warning"],
                "group": band.get("group", ""),
                "assignment": band.get("assignment", ""),
                "reliability": band.get("reliability", ""),
                "role_label": "кандидатный признак" if str(band.get("peak_role")) == "candidate" else "фоновый признак",
                "interpretation": band.get("interpretation", ""),
                "notes": band.get("notes", ""),
                "brief_meaning": _brief_meaning_from_group(str(band.get("group", ""))),
                "effect_size_label": effect_size_to_russian(stats_row.get("effect_size", np.nan)),
                "limitations": _build_band_limitations(band),
            }
        )
    return pd.DataFrame(rows)


def effect_size_to_russian(effect_size: float | int | None) -> str:
    if effect_size is None or pd.isna(effect_size):
        return "недостаточно данных"
    value = abs(float(effect_size))
    if value < 0.2:
        return "различие между группами слабое"
    if value < 0.5:
        return "малое различие"
    if value < 0.8:
        return "умеренное различие"
    return "выраженное различие"


def build_peak_consistency_text(
    predicted_label: str,
    candidate_reference_df: pd.DataFrame,
) -> str:
    label = str(predicted_label)
    if candidate_reference_df.empty:
        candidate_higher = 0
        candidate_any = 0
        all_normal = True
    else:
        labels = candidate_reference_df["deviation_label"].astype(str)
        candidate_higher = int(labels.str.contains("выше контрольной группы", na=False).sum())
        candidate_any = int(len(candidate_reference_df))
        all_normal = bool(
            (labels == "в пределах условной спектральной нормы контрольной группы").all()
        )
    if label == "Disease":
        if candidate_higher >= 2:
            return "Интерпретация спектральных признаков согласуется с прогнозом модели."
        return (
            "Модель предсказывает патологический спектральный профиль, однако выраженных кандидатных признаков из текущего справочника не обнаружено. "
            "Вероятно, решение модели связано с общей формой спектра или признаками вне текущего справочника."
        )
    if candidate_higher >= 2:
        return (
            "Модель предсказывает профиль нормы, однако часть кандидатных спектральных признаков отклоняется от контрольной группы. "
            "Результат требует осторожной интерпретации."
        )
    if candidate_any == 0 or all_normal:
        return "Интерпретация спектральных признаков согласуется с прогнозом модели."
    return "Интерпретация спектральных признаков требует осторожности из-за смешанного характера отклонений."


def _select_reference_basis_matrix(dataset: ProcessedDataset, intensity_basis: str) -> tuple[np.ndarray, str]:
    normalized_basis = str(intensity_basis).strip().lower()
    if normalized_basis == "baseline_corrected" and dataset.X_snr is not None:
        return np.asarray(dataset.X_snr, dtype=float), "baseline_corrected"
    return np.asarray(dataset.X, dtype=float), "snv"


def _extract_band_intensity(
    spectrum: np.ndarray,
    wavenumber: np.ndarray,
    band: dict[str, Any],
    spectroscopy_mode: str,
    aggregation_mode: str = "max",
) -> float:
    agg_mode = normalize_aggregation_mode(aggregation_mode)
    low, high = _get_band_window(band, spectroscopy_mode)
    mask = (wavenumber >= low) & (wavenumber <= high)
    if not np.any(mask):
        return np.nan
    region_values = np.asarray(spectrum[mask], dtype=float)
    region_wavenumber = np.asarray(wavenumber[mask], dtype=float)
    return _aggregate_region(region_values, region_wavenumber, agg_mode)


def normalize_aggregation_mode(aggregation_mode: str | None) -> str:
    normalized = str(aggregation_mode or "max").strip().lower()
    if normalized not in {"max", "mean", "area"}:
        return "max"
    return normalized


def _get_band_window(band: dict[str, Any], spectroscopy_mode: str) -> tuple[float, float]:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    if mode == "sers":
        if pd.notna(band.get("shift_min_cm")) and pd.notna(band.get("shift_max_cm")):
            return float(band["shift_min_cm"]), float(band["shift_max_cm"])
        shift = float(band["shift_cm"])
        tolerance = float(band.get("tolerance_cm", 10.0))
        return shift - tolerance, shift + tolerance
    return float(band["low_cm1"]), float(band["high_cm1"])


def _aggregate_region(region_values: np.ndarray, region_wavenumber: np.ndarray, aggregation_mode: str) -> float:
    if region_values.size == 0:
        return np.nan
    if aggregation_mode == "mean":
        return float(np.mean(region_values))
    if aggregation_mode == "area":
        if region_values.size < 2:
            return float(region_values[0])
        return float(np.trapezoid(region_values, region_wavenumber))
    return float(np.max(region_values))


def _format_expected_position(band: dict[str, Any]) -> str:
    if pd.notna(band.get("shift_min_cm")) and pd.notna(band.get("shift_max_cm")):
        return f"{float(band['shift_min_cm']):.2f}–{float(band['shift_max_cm']):.2f}"
    if pd.notna(band.get("shift_cm")):
        return f"{float(band['shift_cm']):.2f}"
    return ""


def _brief_meaning_from_group(group: str) -> str:
    lowered = group.lower()
    if "липид" in lowered:
        return "липидный признак"
    if "белково-аминокислот" in lowered:
        return "белково-аминокислотный признак"
    if "белков" in lowered:
        return "белковый признак"
    if "нуклеинов" in lowered:
        return "нуклеиновокислотный признак"
    if "пуринов" in lowered or "фонов" in lowered:
        return "фоновый пуриновый признак"
    if "неопредел" in lowered:
        return "неопределённое отнесение"
    return "спектральный признак"


def _build_band_limitations(band: dict[str, Any]) -> str:
    note = str(band.get("notes", "")).strip()
    if note:
        return note
    if str(band.get("peak_role", "")) == "background":
        return "Фоновый признак сыворотки, не является самостоятельным маркером патологии."
    return "Признак интерпретируется только как часть совокупного спектрального паттерна."


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else np.nan


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else np.nan


def _safe_iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q3 - q1)


def _cohen_d(healthy_values: np.ndarray, disease_values: np.ndarray) -> float:
    if healthy_values.size < 2 or disease_values.size < 2:
        return np.nan
    healthy_std = np.std(healthy_values, ddof=1)
    disease_std = np.std(disease_values, ddof=1)
    pooled_var = ((healthy_values.size - 1) * healthy_std**2 + (disease_values.size - 1) * disease_std**2) / (
        healthy_values.size + disease_values.size - 2
    )
    pooled_std = float(np.sqrt(max(pooled_var, 1e-12)))
    return float((np.mean(disease_values) - np.mean(healthy_values)) / pooled_std)
