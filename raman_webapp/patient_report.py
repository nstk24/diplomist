from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from .spectral_references import effect_size_to_russian
from .spectroscopy import get_spectroscopy_mode_label, normalize_spectroscopy_mode


def build_patient_report_data(
    spectroscopy_mode: str,
    prediction: dict[str, Any],
    candidate_df: pd.DataFrame,
    background_df: pd.DataFrame,
    consistency_text: str,
) -> dict[str, Any]:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    return {
        "spectroscopy_mode": mode,
        "spectroscopy_mode_label": get_spectroscopy_mode_label(mode),
        "prediction": prediction,
        "candidate_peaks_df": candidate_df.copy(),
        "background_peaks_df": background_df.copy(),
        "consistency_text": consistency_text,
        "summary": _build_summary_counts(candidate_df),
    }


def build_compact_peak_table(peaks_df: pd.DataFrame) -> pd.DataFrame:
    if peaks_df.empty:
        return pd.DataFrame(
            columns=[
                "Пик / диапазон, см⁻¹",
                "Статус",
                "Отклонение",
                "Группа",
                "Краткий смысл",
            ]
        )

    display_df = peaks_df.copy()
    if "brief_meaning" not in display_df.columns:
        display_df["brief_meaning"] = display_df.get("group", pd.Series("", index=display_df.index)).map(_brief_meaning_from_group)
    if "status" not in display_df.columns:
        display_df["status"] = "обнаружен"
    if "expected_position_cm" not in display_df.columns:
        display_df["expected_position_cm"] = display_df.get("label", "")
    if "deviation_label" not in display_df.columns:
        display_df["deviation_label"] = "Недостаточно данных для расчёта отклонения от контрольной группы"

    display_df["Статус"] = display_df.apply(
        lambda row: _expected_peak_short_status(str(row.get("deviation_label", "")), str(row.get("status", ""))),
        axis=1,
    )
    display_df["Отклонение"] = display_df["deviation_label"]
    display_df["Группа"] = display_df.get("group", display_df.get("label", ""))
    display_df["Краткий смысл"] = display_df["brief_meaning"]
    display_df["Пик / диапазон, см⁻¹"] = display_df["expected_position_cm"]
    return display_df[["Пик / диапазон, см⁻¹", "Статус", "Отклонение", "Группа", "Краткий смысл"]]


def build_detailed_peak_table(peaks_df: pd.DataFrame) -> pd.DataFrame:
    if peaks_df.empty:
        return pd.DataFrame(
            columns=[
                "Пик / диапазон, см⁻¹",
                "Ожидаемый пик, см⁻¹",
                "Найденный максимум, см⁻¹",
                "Статус",
                "Интенсивность пациента",
                "Среднее у здоровых",
                "Стандартное отклонение у здоровых",
                "Z-score",
                "Отклонение от контроля",
                "Медиана у здоровых",
                "IQR у здоровых",
                "Среднее у пациентов с патологией",
                "Размер эффекта",
                "Группа признаков",
                "Молекулярное отнесение",
                "Роль признака",
                "Надёжность",
                "Интерпретация",
                "Ограничения",
            ]
        )

    display_df = peaks_df.copy()
    if "status" not in display_df.columns:
        display_df["status"] = "обнаружен"
    if "role_label" not in display_df.columns:
        display_df["role_label"] = "спектральный признак"
    if "limitations" not in display_df.columns:
        display_df["limitations"] = ""
    if "effect_size" not in display_df.columns:
        display_df["effect_size"] = pd.NA

    display_df["Размер эффекта"] = display_df["effect_size"].map(effect_size_to_russian)
    rename_map = {
        "expected_position_cm": "Пик / диапазон, см⁻¹",
        "expected_peak_cm": "Ожидаемый пик, см⁻¹",
        "measured_shift_cm": "Найденный максимум, см⁻¹",
        "status": "Статус",
        "patient_reference_intensity": "Интенсивность пациента",
        "mean_healthy": "Среднее у здоровых",
        "std_healthy": "Стандартное отклонение у здоровых",
        "z_score": "Z-score",
        "deviation_label": "Отклонение от контроля",
        "median_healthy": "Медиана у здоровых",
        "iqr_healthy": "IQR у здоровых",
        "mean_disease": "Среднее у пациентов с патологией",
        "group": "Группа признаков",
        "assignment": "Молекулярное отнесение",
        "role_label": "Роль признака",
        "reliability": "Надёжность",
        "interpretation": "Интерпретация",
        "limitations": "Ограничения",
    }
    display_df = display_df.rename(columns=rename_map)
    preferred = [
        "Пик / диапазон, см⁻¹",
        "Ожидаемый пик, см⁻¹",
        "Найденный максимум, см⁻¹",
        "Статус",
        "Интенсивность пациента",
        "Среднее у здоровых",
        "Стандартное отклонение у здоровых",
        "Z-score",
        "Отклонение от контроля",
        "Медиана у здоровых",
        "IQR у здоровых",
        "Среднее у пациентов с патологией",
        "Размер эффекта",
        "Группа признаков",
        "Молекулярное отнесение",
        "Роль признака",
        "Надёжность",
        "Интерпретация",
        "Ограничения",
    ]
    return display_df[[column for column in preferred if column in display_df.columns]]


def build_patient_report_markdown(report_data: dict[str, Any]) -> str:
    prediction = report_data["prediction"]
    summary = report_data["summary"]
    candidate_df = report_data["candidate_peaks_df"]
    background_df = report_data["background_peaks_df"]
    mode = str(report_data["spectroscopy_mode"])

    limitation_lines = [
        "- Результат не является медицинским диагнозом.",
        "- Отдельный пик не доказывает наличие заболевания.",
        "- Отклонения рассчитаны относительно контрольной группы текущего датасета.",
        "- Z-score не является клинической нормой концентрации биомаркера.",
        "- Модель должна быть обучена на данных того же типа, что и анализируемый спектр.",
    ]
    if mode == "sers":
        limitation_lines.append("- Для SERS результат зависит от типа подложки, пробоподготовки и условий регистрации.")
    else:
        limitation_lines.append("- Интерпретация Raman-пиков зависит от предобработки, качества спектра и состава обучающего датасета.")

    lines = [
        "# Отчёт по спектральному профилю пациента",
        "",
        "## Результат прогноза модели",
        f"- Режим интерпретации пиков: {report_data['spectroscopy_mode_label']}",
        f"- Предсказанный спектральный профиль: {prediction['predicted_label_ru']}",
        f"- Вероятность патологического спектрального профиля: {float(prediction['probability_disease']):.3f}",
        f"- Вероятность спектрального профиля нормы: {float(prediction['probability_healthy']):.3f}",
        "- Результат не является медицинским диагнозом.",
        "- Прогноз основан на сходстве спектра с группами обучающего датасета.",
        "",
        "## Сравнение с контрольной группой",
        "- Условная спектральная норма рассчитана по здоровым донорам текущего датасета.",
        (
            f"- Выше контрольной группы: {summary['higher_count']}; "
            f"ниже контрольной группы: {summary['lower_count']}; "
            f"в пределах условной спектральной нормы: {summary['normal_count']}; "
            f"недостаточно данных: {summary['unknown_count']}."
        ),
        "",
        "## Согласованность прогноза и спектральных признаков",
        report_data["consistency_text"],
        "",
        "## Кандидатные признаки патологии",
        _dataframe_to_markdown(build_detailed_peak_table(candidate_df)),
        "",
        "## Фоновые признаки сыворотки",
        "Пики 638, 725 и 1659 см⁻¹ рассматриваются как фоновые признаки сыворотки и не интерпретируются как самостоятельные маркеры сердечно-сосудистой патологии.",
        _dataframe_to_markdown(build_detailed_peak_table(background_df)),
        "",
        "## Ограничения интерпретации",
        *limitation_lines,
    ]
    return "\n".join(lines)


def build_patient_report_pdf(report_data: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    font_name = _register_pdf_font(pdfmetrics, TTFont)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyRu",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=3,
    )
    heading1_style = ParagraphStyle(
        "Heading1Ru",
        parent=styles["Heading1"],
        fontName=font_name,
        fontSize=16,
        leading=20,
        spaceAfter=8,
    )
    heading2_style = ParagraphStyle(
        "Heading2Ru",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=16,
        spaceBefore=3,
        spaceAfter=6,
    )
    header_cell_style = ParagraphStyle(
        "HeaderCellRu",
        parent=body_style,
        fontName=font_name,
        fontSize=8.2,
        leading=10.0,
        textColor=colors.white,
    )
    body_cell_style = ParagraphStyle(
        "BodyCellRu",
        parent=body_style,
        fontName=font_name,
        fontSize=7.2,
        leading=8.8,
    )

    candidate_df = build_detailed_peak_table(report_data["candidate_peaks_df"])
    background_df = build_detailed_peak_table(report_data["background_peaks_df"])
    summary = report_data["summary"]
    prediction = report_data["prediction"]
    mode = str(report_data["spectroscopy_mode"])

    portrait_width, portrait_height = A4
    landscape_width, landscape_height = landscape(A4)
    margin = 10 * mm
    portrait_frame = Frame(margin, margin, portrait_width - 2 * margin, portrait_height - 2 * margin, id="portrait_frame")
    landscape_frame = Frame(margin, margin, landscape_width - 2 * margin, landscape_height - 2 * margin, id="landscape_frame")

    buffer = BytesIO()
    document = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Отчёт по спектральному профилю пациента",
    )
    document.addPageTemplates(
        [
            PageTemplate(id="portrait", frames=[portrait_frame], pagesize=A4),
            PageTemplate(id="landscape", frames=[landscape_frame], pagesize=landscape(A4)),
        ]
    )

    limitation_lines = [
        "Результат не является медицинским диагнозом.",
        "Отдельный пик не доказывает наличие заболевания.",
        "Отклонения рассчитаны относительно контрольной группы текущего датасета.",
        "Z-score не является клинической нормой концентрации биомаркера.",
        "Модель должна быть обучена на данных того же типа, что и анализируемый спектр.",
    ]
    if mode == "sers":
        limitation_lines.append("Для SERS результат зависит от типа подложки, пробоподготовки и условий регистрации.")
    else:
        limitation_lines.append("Интерпретация Raman-пиков зависит от предобработки, качества спектра и состава обучающего датасета.")

    story: list[Any] = [
        Paragraph("Отчёт по спектральному профилю пациента", heading1_style),
        Paragraph("Результат прогноза модели", heading2_style),
        Paragraph(f"• Режим интерпретации пиков: {escape(str(report_data['spectroscopy_mode_label']))}", body_style),
        Paragraph(f"• Предсказанный спектральный профиль: {escape(str(prediction['predicted_label_ru']))}", body_style),
        Paragraph(
            f"• Вероятность патологического спектрального профиля: {float(prediction['probability_disease']):.3f}",
            body_style,
        ),
        Paragraph(
            f"• Вероятность спектрального профиля нормы: {float(prediction['probability_healthy']):.3f}",
            body_style,
        ),
        Paragraph("• Результат не является медицинским диагнозом.", body_style),
        Paragraph("• Прогноз основан на сходстве спектра с группами обучающего датасета.", body_style),
        Spacer(1, 3 * mm),
        Paragraph("Сравнение с контрольной группой", heading2_style),
        Paragraph("• Условная спектральная норма рассчитана по здоровым донорам текущего датасета.", body_style),
        Paragraph(
            "• "
            + (
                f"Выше контрольной группы: {summary['higher_count']}; "
                f"ниже контрольной группы: {summary['lower_count']}; "
                f"в пределах условной спектральной нормы: {summary['normal_count']}; "
                f"недостаточно данных: {summary['unknown_count']}."
            ),
            body_style,
        ),
        Spacer(1, 3 * mm),
        Paragraph("Согласованность прогноза и спектральных признаков", heading2_style),
        Paragraph(escape(str(report_data["consistency_text"])), body_style),
    ]

    story.extend(
        _build_pdf_table_section(
            df=candidate_df,
            section_title="Кандидатные признаки патологии",
            empty_message="Нет данных.",
            heading_style=heading2_style,
            header_style=header_cell_style,
            cell_style=body_cell_style,
            colors_module=colors,
        )
    )
    story.extend(
        _build_pdf_table_section(
            df=background_df,
            section_title="Фоновые признаки сыворотки",
            intro_text="Пики 638, 725 и 1659 см⁻¹ рассматриваются как фоновые признаки сыворотки и не интерпретируются как самостоятельные маркеры сердечно-сосудистой патологии.",
            empty_message="Нет данных.",
            heading_style=heading2_style,
            header_style=header_cell_style,
            cell_style=body_cell_style,
            colors_module=colors,
        )
    )

    story.extend(
        [
            NextPageTemplate("portrait"),
            PageBreak(),
            Paragraph("Ограничения интерпретации", heading2_style),
        ]
    )
    for line in limitation_lines:
        story.append(Paragraph(f"• {escape(line)}", body_style))

    document.build(story)
    return buffer.getvalue()


def _build_pdf_table_section(
    df: pd.DataFrame,
    section_title: str,
    heading_style: Any,
    header_style: Any,
    cell_style: Any,
    colors_module: Any,
    empty_message: str,
    intro_text: str | None = None,
) -> list[Any]:
    from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph, Spacer

    if df.empty:
        story: list[Any] = [Paragraph(section_title, heading_style)]
        if intro_text:
            story.append(Paragraph(escape(intro_text), cell_style))
            story.append(Spacer(1, 2))
        story.append(Paragraph(empty_message, cell_style))
        return story

    story = [NextPageTemplate("landscape"), PageBreak(), Paragraph(section_title, heading_style)]
    if intro_text:
        story.append(Paragraph(escape(intro_text), cell_style))
        story.append(Spacer(1, 3))

    column_chunks = _split_report_table_columns(df.columns.tolist())
    total_chunks = len(column_chunks)
    for index, columns in enumerate(column_chunks, start=1):
        if index > 1:
            story.append(Spacer(1, 6))
        if total_chunks > 1:
            story.append(Paragraph(f"{section_title}: часть {index} из {total_chunks}", heading_style))
        story.append(
            _make_pdf_table(
                df[columns],
                columns,
                _column_widths_for_chunk(columns),
                header_style,
                cell_style,
                colors_module,
            )
        )
    return story


def _make_pdf_table(
    df: pd.DataFrame,
    columns: list[str],
    col_widths_mm: list[float],
    header_style: Any,
    cell_style: Any,
    colors_module: Any,
) -> Any:
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    available_width_mm = 277.0
    total_width_mm = sum(col_widths_mm)
    if total_width_mm > 0:
        scale = min(1.18, available_width_mm / total_width_mm)
        col_widths_mm = [width * scale for width in col_widths_mm]

    header_row = [Paragraph(escape(column), header_style) for column in columns]
    body_rows: list[list[Any]] = []
    for _, row in df.iterrows():
        body_rows.append(
            [
                Paragraph(escape(_format_cell_value(row[column])), cell_style)
                for column in columns
            ]
        )
    table = Table([header_row, *body_rows], colWidths=[width * mm for width in col_widths_mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors_module.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors_module.white),
                ("GRID", (0, 0), (-1, -1), 0.45, colors_module.HexColor("#7A7A7A")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors_module.whitesmoke, colors_module.HexColor("#EAF2F8")]),
            ]
        )
    )
    return table


def _split_report_table_columns(columns: list[str]) -> list[list[str]]:
    if len(columns) <= 10:
        return [columns]
    numeric_chunk = [
        "Пик / диапазон, см⁻¹",
        "Ожидаемый пик, см⁻¹",
        "Найденный максимум, см⁻¹",
        "Статус",
        "Интенсивность пациента",
        "Среднее у здоровых",
        "Стандартное отклонение у здоровых",
        "Z-score",
        "Отклонение от контроля",
        "Медиана у здоровых",
        "IQR у здоровых",
    ]
    annotation_chunk = [
        "Пик / диапазон, см⁻¹",
        "Среднее у пациентов с патологией",
        "Размер эффекта",
        "Группа признаков",
        "Молекулярное отнесение",
        "Роль признака",
        "Надёжность",
        "Интерпретация",
        "Ограничения",
    ]
    chunks = []
    for chunk in (numeric_chunk, annotation_chunk):
        filtered = [column for column in chunk if column in columns]
        if filtered:
            chunks.append(filtered)
    return chunks or [columns]


def _column_widths_for_chunk(columns: list[str]) -> list[float]:
    width_map = {
        "Пик / диапазон, см⁻¹": 24,
        "Ожидаемый пик, см⁻¹": 21,
        "Найденный максимум, см⁻¹": 23,
        "Статус": 20,
        "Интенсивность пациента": 23,
        "Среднее у здоровых": 22,
        "Стандартное отклонение у здоровых": 27,
        "Z-score": 13,
        "Отклонение от контроля": 33,
        "Медиана у здоровых": 20,
        "IQR у здоровых": 18,
        "Среднее у пациентов с патологией": 25,
        "Размер эффекта": 21,
        "Группа признаков": 31,
        "Молекулярное отнесение": 40,
        "Роль признака": 18,
        "Надёжность": 16,
        "Интерпретация": 40,
        "Ограничения": 44,
    }
    return [width_map.get(column, 20) for column in columns]


def _format_cell_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_summary_counts(candidate_df: pd.DataFrame) -> dict[str, int]:
    if candidate_df.empty:
        return {"higher_count": 0, "lower_count": 0, "normal_count": 0, "unknown_count": 0}
    labels = candidate_df["deviation_label"].astype(str)
    return {
        "higher_count": int(labels.str.contains("выше контрольной группы", na=False).sum()),
        "lower_count": int(labels.str.contains("ниже контрольной группы", na=False).sum()),
        "normal_count": int((labels == "в пределах условной спектральной нормы контрольной группы").sum()),
        "unknown_count": int(labels.str.contains("Недостаточно данных", na=False).sum()),
    }


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "Нет данных."
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(column) for column in df.columns]
        separator = ["---"] * len(headers)
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for _, row in df.iterrows():
            values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)


def _expected_peak_short_status(deviation_label: str, status: str) -> str:
    if status == "не обнаружен":
        return "не обнаружен"
    if "Недостаточно данных" in str(deviation_label):
        return "недостаточно данных"
    if str(deviation_label) == "в пределах условной спектральной нормы контрольной группы":
        return "в пределах контрольной группы"
    if "выше контрольной группы" in str(deviation_label):
        return "выше контрольной группы"
    if "ниже контрольной группы" in str(deviation_label):
        return "ниже контрольной группы"
    return status or "обнаружен"


def _brief_meaning_from_group(group: Any) -> str:
    lowered = str(group).lower()
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


def _register_pdf_font(pdfmetrics: Any, ttfont_cls: Any) -> str:
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/verdana.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            pdfmetrics.registerFont(ttfont_cls("AppReportFont", str(font_path)))
            return "AppReportFont"
    return "Helvetica"
