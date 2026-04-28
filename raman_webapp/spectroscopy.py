from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import PeakBandMatchResult, match_peaks_to_reference_bands


DEFAULT_SPECTROSCOPY_MODE = "raman"
SPECTROSCOPY_MODE_LABELS = {
    "raman": "Рамановская спектроскопия",
    "sers": "SERS-спектроскопия",
}
SERS_DEFAULT_TOLERANCE_CM = 10.0
SERS_RELIABILITY_WEIGHTS = {
    "высокая": 1.1,
    "средняя": 1.0,
    "низкая–средняя": 0.6,
    "низкая": 0.4,
}
DEFAULT_SERS_SCORE_CONFIG = {
    "слабый_максимум": 2.0,
    "умеренный_максимум": 5.0,
    "слабые_пики_максимум": 2,
    "умеренные_пики_максимум": 5,
    "групповой_бонус": 0.08,
    "максимальный_групповой_множитель": 1.5,
}
SERS_PATTERN_WARNINGS = [
    "Отдельные SERS-пики не являются самостоятельными диагностическими маркерами.",
    "SERS и обычную рамановскую спектроскопию не следует интерпретировать по одному справочнику.",
]


@dataclass
class SpectroscopyAnalysisResult:
    spectroscopy_mode: str
    spectroscopy_mode_label: str
    matched_peaks_df: pd.DataFrame
    summary_df: pd.DataFrame
    detected_candidate_peaks_df: pd.DataFrame
    detected_background_peaks_df: pd.DataFrame
    candidate_peak_count: int
    background_peak_count: int
    represented_groups: list[str]
    sers_cvd_pattern_level: str | None
    sers_cvd_score: float | None
    interpretation: str
    warnings: list[str]

    def to_display_payload(self) -> dict[str, Any]:
        return {
            "spectroscopy_mode": self.spectroscopy_mode,
            "spectroscopy_mode_label": self.spectroscopy_mode_label,
            "detected_candidate_peaks": self.detected_candidate_peaks_df.to_dict(orient="records"),
            "detected_background_peaks": self.detected_background_peaks_df.to_dict(orient="records"),
            "candidate_peak_count": self.candidate_peak_count,
            "background_peak_count": self.background_peak_count,
            "represented_groups": self.represented_groups,
            "sers_cvd_pattern_level": self.sers_cvd_pattern_level,
            "sers_cvd_score": self.sers_cvd_score,
            "interpretation": self.interpretation,
            "warnings": self.warnings,
        }


def normalize_spectroscopy_mode(mode: str | None) -> str:
    normalized = str(mode or DEFAULT_SPECTROSCOPY_MODE).strip().lower()
    if normalized not in SPECTROSCOPY_MODE_LABELS:
        return DEFAULT_SPECTROSCOPY_MODE
    return normalized


def get_spectroscopy_mode_label(mode: str | None) -> str:
    normalized = normalize_spectroscopy_mode(mode)
    return SPECTROSCOPY_MODE_LABELS[normalized]


def analyze_spectroscopy_peaks(
    peaks_df: pd.DataFrame,
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    reference_library_df: pd.DataFrame,
    spectroscopy_mode: str | None = None,
    matching_mode: str = "close",
    tolerance_cm: float | None = None,
    sers_score_config: dict[str, float] | None = None,
) -> SpectroscopyAnalysisResult:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    if mode == "sers":
        return analyze_sers_peaks(
            peaks_df=peaks_df,
            wavenumber=wavenumber,
            intensity=intensity,
            reference_library_df=reference_library_df,
            tolerance_cm=SERS_DEFAULT_TOLERANCE_CM if tolerance_cm is None else float(tolerance_cm),
            sers_score_config=sers_score_config,
        )
    return analyze_raman_peaks(
        peaks_df=peaks_df,
        reference_library_df=reference_library_df,
        matching_mode=matching_mode,
        tolerance_cm=5.0 if tolerance_cm is None else float(tolerance_cm),
    )


def analyze_raman_peaks(
    peaks_df: pd.DataFrame,
    reference_library_df: pd.DataFrame,
    matching_mode: str = "close",
    tolerance_cm: float = 5.0,
) -> SpectroscopyAnalysisResult:
    peak_match_result = match_peaks_to_reference_bands(
        peaks_df,
        reference_library_df,
        matching_mode=matching_mode,
        tolerance_cm=tolerance_cm,
    )
    matched_df = peak_match_result.matched_peaks_df.copy()
    interpretation = _build_raman_interpretation(peak_match_result)
    return SpectroscopyAnalysisResult(
        spectroscopy_mode="raman",
        spectroscopy_mode_label=get_spectroscopy_mode_label("raman"),
        matched_peaks_df=matched_df,
        summary_df=peak_match_result.summary_df.copy(),
        detected_candidate_peaks_df=matched_df.copy(),
        detected_background_peaks_df=pd.DataFrame(),
        candidate_peak_count=int(len(matched_df)),
        background_peak_count=0,
        represented_groups=[],
        sers_cvd_pattern_level=None,
        sers_cvd_score=None,
        interpretation=interpretation,
        warnings=["Обычную рамановскую спектроскопию и SERS не следует интерпретировать по одному справочнику."],
    )


def analyze_sers_peaks(
    peaks_df: pd.DataFrame,
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    reference_library_df: pd.DataFrame,
    tolerance_cm: float = SERS_DEFAULT_TOLERANCE_CM,
    sers_score_config: dict[str, float] | None = None,
) -> SpectroscopyAnalysisResult:
    wavenumber = np.asarray(wavenumber, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    peak_records = peaks_df.to_dict(orient="records")
    intensity_scale = float(np.max(np.abs(intensity))) if intensity.size else 0.0
    intensity_scale = max(intensity_scale, 1e-12)

    match_rows: list[dict[str, Any]] = []
    for band in reference_library_df.to_dict(orient="records"):
        match_rows.append(
            _analyze_single_sers_band(
                band=band,
                peak_records=peak_records,
                wavenumber=wavenumber,
                intensity=intensity,
                intensity_scale=intensity_scale,
                tolerance_cm=tolerance_cm,
            )
        )

    matched_df = pd.DataFrame(match_rows)
    candidate_df = matched_df[(matched_df["peak_present"]) & (matched_df["peak_role"] == "candidate")].copy()
    background_df = matched_df[(matched_df["peak_present"]) & (matched_df["peak_role"] == "background")].copy()
    represented_groups = sorted(dict.fromkeys(candidate_df["group"].tolist()))
    score, pattern_level = _compute_sers_cvd_score(candidate_df, represented_groups, sers_score_config)
    interpretation = _build_sers_interpretation(candidate_df, background_df, represented_groups, pattern_level)
    return SpectroscopyAnalysisResult(
        spectroscopy_mode="sers",
        spectroscopy_mode_label=get_spectroscopy_mode_label("sers"),
        matched_peaks_df=matched_df,
        summary_df=_build_sers_summary_df(candidate_df, background_df),
        detected_candidate_peaks_df=candidate_df,
        detected_background_peaks_df=background_df,
        candidate_peak_count=int(len(candidate_df)),
        background_peak_count=int(len(background_df)),
        represented_groups=represented_groups,
        sers_cvd_pattern_level=pattern_level,
        sers_cvd_score=score,
        interpretation=interpretation,
        warnings=SERS_PATTERN_WARNINGS.copy(),
    )


def build_reference_library_display_df(reference_library_df: pd.DataFrame, spectroscopy_mode: str | None) -> pd.DataFrame:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    display_df = reference_library_df.copy()
    if mode == "sers":
        if "shift_cm" in display_df.columns:
            display_df["Ожидаемый пик, см⁻¹"] = display_df["shift_cm"].map(lambda value: f"{float(value):.2f}")
        if {"shift_min_cm", "shift_max_cm"}.issubset(display_df.columns):
            display_df["Ожидаемый диапазон, см⁻¹"] = display_df.apply(
                lambda row: (
                    f"{float(row['shift_min_cm']):.2f}–{float(row['shift_max_cm']):.2f}"
                    if pd.notna(row.get("shift_min_cm")) and pd.notna(row.get("shift_max_cm"))
                    else ""
                ),
                axis=1,
            )
        column_map = {
            "peak_role": "Роль признака",
            "group": "Группа признаков",
            "assignment": "Молекулярное отнесение",
            "associated_condition": "Связь с состоянием",
            "interpretation": "Биохимическая интерпретация",
            "direction": "Направление изменения",
            "reliability": "Надёжность",
            "notes": "Примечания",
        }
        preferred_columns = [
            "Роль признака",
            "Ожидаемый пик, см⁻¹",
            "Ожидаемый диапазон, см⁻¹",
            "Группа признаков",
            "Молекулярное отнесение",
            "Связь с состоянием",
            "Биохимическая интерпретация",
            "Направление изменения",
            "Надёжность",
            "Примечания",
        ]
        display_df = display_df.rename(columns=column_map)
        return display_df[[column for column in preferred_columns if column in display_df.columns]]

    column_map = {
        "priority": "Приоритет",
        "label": "Полоса",
        "low_cm1": "Нижняя граница, см⁻¹",
        "high_cm1": "Верхняя граница, см⁻¹",
        "assignment": "Молекулярное отнесение",
        "clinical_hint": "Клиническая подсказка",
        "notes": "Примечания",
    }
    preferred_columns = [
        "Приоритет",
        "Полоса",
        "Нижняя граница, см⁻¹",
        "Верхняя граница, см⁻¹",
        "Молекулярное отнесение",
        "Клиническая подсказка",
        "Примечания",
    ]
    display_df = display_df.rename(columns=column_map)
    return display_df[[column for column in preferred_columns if column in display_df.columns]]


def _analyze_single_sers_band(
    band: dict[str, Any],
    peak_records: list[dict[str, Any]],
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    intensity_scale: float,
    tolerance_cm: float,
) -> dict[str, Any]:
    is_range = pd.notna(band.get("shift_min_cm")) and pd.notna(band.get("shift_max_cm"))
    if is_range:
        expected_low = float(band["shift_min_cm"])
        expected_high = float(band["shift_max_cm"])
        search_low = expected_low
        search_high = expected_high
    else:
        expected_shift = float(band["shift_cm"])
        expected_low = expected_shift - float(band.get("tolerance_cm", tolerance_cm))
        expected_high = expected_shift + float(band.get("tolerance_cm", tolerance_cm))
        search_low = expected_low
        search_high = expected_high

    local_peak_candidates = [
        row for row in peak_records if search_low <= float(row["wavenumber_cm-1"]) <= search_high
    ]
    local_peak_candidates.sort(key=lambda row: float(row["intensity"]), reverse=True)

    window_mask = (wavenumber >= search_low) & (wavenumber <= search_high)
    if np.any(window_mask):
        local_wavenumber = wavenumber[window_mask]
        local_intensity = intensity[window_mask]
        local_idx = int(np.argmax(local_intensity))
        measured_shift = float(local_wavenumber[local_idx])
        measured_intensity = float(local_intensity[local_idx])
    else:
        measured_shift = np.nan
        measured_intensity = np.nan

    peak_present = bool(local_peak_candidates)
    if peak_present:
        best_peak = local_peak_candidates[0]
        measured_shift = float(best_peak["wavenumber_cm-1"])
        measured_intensity = float(best_peak["intensity"])
        prominence = float(best_peak.get("prominence", 0.0))
    else:
        prominence = 0.0

    normalized_intensity = (
        float(measured_intensity / intensity_scale)
        if pd.notna(measured_intensity)
        else 0.0
    )
    if is_range:
        expected_position_text = f"{expected_low:.2f}–{expected_high:.2f}"
        if peak_present:
            match_score = 1.0
        else:
            match_score = 0.0
    else:
        expected_shift = float(band["shift_cm"])
        expected_position_text = f"{expected_shift:.2f} ± {float(band.get('tolerance_cm', tolerance_cm)):.2f}"
        if peak_present:
            match_score = max(0.0, 1.0 - abs(measured_shift - expected_shift) / max(float(band.get("tolerance_cm", tolerance_cm)), 1e-12))
        else:
            match_score = 0.0

    return {
        "band_id": band["band_id"],
        "peak_role": band["peak_role"],
        "group": band["group"],
        "assignment": band["assignment"],
        "associated_condition": band["associated_condition"],
        "interpretation": band["interpretation"],
        "direction": band["direction"],
        "reliability": band["reliability"],
        "is_diagnostic_alone": bool(band.get("is_diagnostic_alone", False)),
        "notes": band.get("notes", ""),
        "peak_present": peak_present,
        "presence_label": "обнаружен" if peak_present else "не обнаружен",
        "measured_shift_cm": measured_shift,
        "expected_position_cm": expected_position_text,
        "intensity": measured_intensity,
        "normalized_intensity": normalized_intensity,
        "match_score": float(match_score),
        "prominence": prominence,
    }


def _compute_sers_cvd_score(
    candidate_df: pd.DataFrame,
    represented_groups: list[str],
    sers_score_config: dict[str, float] | None,
) -> tuple[float, str]:
    config = DEFAULT_SERS_SCORE_CONFIG.copy()
    if sers_score_config:
        config.update(sers_score_config)

    if candidate_df.empty:
        return 0.0, "паттерн не выявлен"

    weights = candidate_df["reliability"].map(SERS_RELIABILITY_WEIGHTS).fillna(0.5)
    base_score = float(weights.sum())
    group_multiplier = min(
        float(config["максимальный_групповой_множитель"]),
        1.0 + max(0, len(represented_groups) - 1) * float(config["групповой_бонус"]),
    )
    final_score = base_score * group_multiplier

    candidate_count = int(len(candidate_df))
    if final_score <= float(config["слабый_максимум"]) or candidate_count <= int(config["слабые_пики_максимум"]):
        return round(final_score, 3), "слабовыраженный"
    if final_score <= float(config["умеренный_максимум"]) or candidate_count <= int(config["умеренные_пики_максимум"]):
        return round(final_score, 3), "умеренно выраженный"
    return round(final_score, 3), "выраженный"


def _build_sers_summary_df(candidate_df: pd.DataFrame, background_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not candidate_df.empty:
        rows.extend(
            candidate_df.groupby("group", sort=False)
            .agg(
                количество=("band_id", "count"),
                пики=(
                    "expected_position_cm",
                    lambda values: ", ".join(dict.fromkeys(str(value) for value in values)),
                ),
            )
            .reset_index()
            .assign(роль="кандидатные признаки")
            .to_dict(orient="records")
        )
    if not background_df.empty:
        rows.extend(
            background_df.groupby("group", sort=False)
            .agg(
                количество=("band_id", "count"),
                пики=(
                    "expected_position_cm",
                    lambda values: ", ".join(dict.fromkeys(str(value) for value in values)),
                ),
            )
            .reset_index()
            .assign(роль="фоновые признаки")
            .to_dict(orient="records")
        )
    return pd.DataFrame(rows)


def _build_sers_interpretation(
    candidate_df: pd.DataFrame,
    background_df: pd.DataFrame,
    represented_groups: list[str],
    pattern_level: str,
) -> str:
    candidate_count = int(len(candidate_df))
    background_count = int(len(background_df))
    if candidate_count == 0:
        first_sentence = (
            "В спектре не выявлен выраженный набор кандидатных SERS-признаков, ассоциированных "
            "с сердечно-сосудистой патологией."
        )
    else:
        first_sentence = (
            f"В спектре обнаружен {pattern_level} SERS-паттерн, ассоциированный с сердечно-сосудистой патологией. "
            f"Найдено {candidate_count} кандидатных SERS-пиков."
        )

    if represented_groups:
        group_text = "Представлены группы признаков: " + ", ".join(represented_groups) + "."
    else:
        group_text = "Кандидатные группы признаков в текущем спектре не представлены."

    if background_count > 0:
        background_labels = ", ".join(dict.fromkeys(background_df["expected_position_cm"].astype(str).tolist()))
        background_text = (
            f"Также обнаружены фоновые SERS-пики сыворотки ({background_labels}). "
            "Они учитываются при интерпретации, но не повышают напрямую вероятность сердечно-сосудистой патологии."
        )
    else:
        background_text = "Выраженные фоновые пуриновые SERS-пики сыворотки не обнаружены."

    caution_text = (
        "Полученный результат следует рассматривать как вклад в интегральную оценку риска, "
        "а не как самостоятельный диагноз. Итоговая вероятность состояния должна оцениваться "
        "совместно с моделью машинного обучения и клиническими данными."
    )
    return " ".join([first_sentence, group_text, background_text, caution_text])


def _build_raman_interpretation(peak_match_result: PeakBandMatchResult) -> str:
    if peak_match_result.matched_peaks_df.empty:
        return (
            "Для текущего режима рамановской спектроскопии совпадения с локальной библиотекой полос "
            "не обнаружены. Это не исключает наличие особенностей спектра, а только означает отсутствие "
            "совпадений с текущим справочником."
        )
    n_matches = int(len(peak_match_result.matched_peaks_df))
    labels = ", ".join(dict.fromkeys(peak_match_result.matched_peaks_df["label"].tolist()))
    return (
        f"Для режима рамановской спектроскопии найдено {n_matches} совпадений с локальной библиотекой полос. "
        f"Наиболее заметные совпадения: {labels}. Отдельные полосы не следует трактовать как самостоятельный диагноз."
    )
