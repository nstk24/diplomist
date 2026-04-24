from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class GeminiInterpretationResult:
    ok: bool
    text: str
    prompt_payload: dict[str, Any] | None = None
    model_used: str | None = None


def is_gemini_configured() -> tuple[bool, str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False, "Переменная окружения GEMINI_API_KEY не найдена."
    return True, "GEMINI_API_KEY найдена."


def build_peak_region_payload(
    peaks_df: pd.DataFrame,
    max_peaks: int = 12,
) -> list[dict[str, float | int]]:
    if peaks_df.empty:
        return []

    payload_rows: list[dict[str, float | int]] = []
    for row in peaks_df.head(max_peaks).to_dict(orient="records"):
        center = float(row["wavenumber_cm-1"])
        width = float(row["width_cm-1"])
        half_width = max(width / 2.0, 3.0)
        payload_rows.append(
            {
                "rank": int(row["rank_by_prominence"]),
                "center_cm1": round(center, 2),
                "region_low_cm1": round(center - half_width, 2),
                "region_high_cm1": round(center + half_width, 2),
                "prominence": round(float(row["prominence"]), 5),
                "width_cm1": round(width, 2),
                "class_importance_score": round(float(row.get("class_importance_score", 0.0)), 5),
                "class_importance_priority": row.get("class_importance_priority"),
            }
        )
    return payload_rows


def generate_gemini_hypotheses(
    peaks_df: pd.DataFrame,
    prediction_probability_disease: float | None,
    peak_basis_label: str,
    spectrum_quality_text: str,
    model_name: str,
) -> GeminiInterpretationResult:
    configured, message = is_gemini_configured()
    if not configured:
        return GeminiInterpretationResult(ok=False, text=message)

    peak_payload = build_peak_region_payload(peaks_df)
    if not peak_payload:
        return GeminiInterpretationResult(
            ok=False,
            text="Для Gemini нет данных: сначала нужно найти пики на загруженном спектре.",
        )

    prompt_payload = {
        "peak_basis": peak_basis_label,
        "prediction_probability_disease": (
            round(float(prediction_probability_disease), 4)
            if prediction_probability_disease is not None
            else None
        ),
        "spectrum_quality_comment": spectrum_quality_text,
        "model_name": model_name,
        "peak_regions": peak_payload,
    }

    prompt = f"""
Ты помогаешь врачу интерпретировать Raman-спектр сыворотки крови как систему поддержки принятия решений.

Важно:
- Не ставь диагноз.
- Не утверждай, что конкретное вещество точно обнаружено.
- Работай только как генератор гипотез по спектральным областям.
- Для каждой гипотезы указывай, что нужна биохимическая верификация.
- Учитывай возможную связь с сердечно-сосудистыми заболеваниями, если она правдоподобна.
- Пиши по-русски, кратко и профессионально.

Ниже агрегированные данные по спектру пациента. Это не сырые массивы и не готовые определения веществ:
{prompt_payload}

Сформируй ответ в 3 частях:
1. Краткий вывод.
2. Возможные биохимические гипотезы по спектральным областям.
3. Что врачу имеет смысл проверить лабораторно.
""".strip()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return GeminiInterpretationResult(
            ok=False,
            text="Пакет google-genai не установлен. Установи зависимости из requirements.txt.",
            prompt_payload=prompt_payload,
        )

    client = genai.Client()
    model_candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-09-2025",
    ]
    retry_delays_sec = [0.0, 2.0, 5.0]
    last_error: Exception | None = None

    for model_name in model_candidates:
        for delay in retry_delays_sec:
            if delay > 0:
                time.sleep(delay)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                text = getattr(response, "text", None) or "Gemini не вернул текстовый ответ."
                return GeminiInterpretationResult(
                    ok=True,
                    text=text,
                    prompt_payload=prompt_payload,
                    model_used=model_name,
                )
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                if "503" in error_text or "UNAVAILABLE" in error_text or "high demand" in error_text:
                    continue
                return GeminiInterpretationResult(
                    ok=False,
                    text=f"Не удалось получить ответ Gemini: {exc}",
                    prompt_payload=prompt_payload,
                    model_used=model_name,
                )

    return GeminiInterpretationResult(
        ok=False,
        text=(
            "Gemini временно перегружен или недоступен. "
            "Было выполнено несколько повторов и попытка резервной модели. "
            f"Последняя ошибка: {last_error}"
        ),
        prompt_payload=prompt_payload,
    )
