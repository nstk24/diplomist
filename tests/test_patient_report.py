from __future__ import annotations

import unittest

import pandas as pd

from raman_webapp.patient_report import (
    build_compact_peak_table,
    build_detailed_peak_table,
    build_patient_report_data,
    build_patient_report_markdown,
    build_patient_report_pdf,
)


class PatientReportTest(unittest.TestCase):
    def test_compact_table_contains_only_compact_columns(self) -> None:
        peaks_df = pd.DataFrame(
            [
                {
                    "expected_position_cm": "1303.00-1359.00",
                    "status": "обнаружен",
                    "deviation_label": "умеренно выше контрольной группы",
                    "group": "смешанные белково-липидные признаки",
                    "brief_meaning": "липидный признак",
                    "mean_healthy": 0.3,
                }
            ]
        )
        compact_df = build_compact_peak_table(peaks_df)
        self.assertEqual(
            compact_df.columns.tolist(),
            ["Пик / диапазон, см⁻¹", "Статус", "Отклонение", "Группа", "Краткий смысл"],
        )

    def test_detailed_table_contains_report_columns(self) -> None:
        peaks_df = pd.DataFrame(
            [
                {
                    "expected_position_cm": "534.00",
                    "expected_peak_cm": 534.0,
                    "measured_shift_cm": 535.0,
                    "status": "обнаружен",
                    "patient_reference_intensity": 1.1,
                    "mean_healthy": 0.7,
                    "std_healthy": 0.2,
                    "z_score": 2.0,
                    "deviation_label": "выраженно выше контрольной группы",
                    "median_healthy": 0.68,
                    "iqr_healthy": 0.12,
                    "mean_disease": 1.0,
                    "effect_size": 0.9,
                    "group": "липидные признаки",
                    "assignment": "эфиры холестерина",
                    "role_label": "кандидатный признак",
                    "reliability": "средняя",
                    "interpretation": "возможный липидный вклад",
                    "limitations": "использовать как часть совокупного паттерна",
                }
            ]
        )
        detailed_df = build_detailed_peak_table(peaks_df)
        self.assertIn("Размер эффекта", detailed_df.columns.tolist())
        self.assertIn("Ограничения", detailed_df.columns.tolist())
        self.assertIn("Молекулярное отнесение", detailed_df.columns.tolist())

    def test_report_markdown_is_russian_and_contains_limitations(self) -> None:
        candidate_df = pd.DataFrame(
            [
                {
                    "expected_position_cm": "534.00",
                    "expected_peak_cm": 534.0,
                    "measured_shift_cm": 535.0,
                    "status": "обнаружен",
                    "patient_reference_intensity": 1.1,
                    "mean_healthy": 0.7,
                    "std_healthy": 0.2,
                    "z_score": 2.0,
                    "deviation_label": "выраженно выше контрольной группы",
                    "median_healthy": 0.68,
                    "iqr_healthy": 0.12,
                    "mean_disease": 1.0,
                    "effect_size": 0.9,
                    "group": "липидные признаки",
                    "assignment": "эфиры холестерина",
                    "role_label": "кандидатный признак",
                    "reliability": "средняя",
                    "interpretation": "возможный липидный вклад",
                    "limitations": "использовать как часть совокупного паттерна",
                }
            ]
        )
        report_data = build_patient_report_data(
            spectroscopy_mode="sers",
            prediction={
                "predicted_label_ru": "патологический спектральный профиль",
                "probability_disease": 0.82,
                "probability_healthy": 0.18,
            },
            candidate_df=candidate_df,
            background_df=pd.DataFrame(),
            consistency_text="Интерпретация спектральных признаков согласуется с прогнозом модели.",
        )
        markdown = build_patient_report_markdown(report_data)
        self.assertIn("## Ограничения интерпретации", markdown)
        self.assertIn("Результат не является медицинским диагнозом", markdown)
        self.assertIn("контрольной группы текущего датасета", markdown)

    def test_report_pdf_is_generated(self) -> None:
        report_data = build_patient_report_data(
            spectroscopy_mode="raman",
            prediction={
                "predicted_label_ru": "спектральный профиль нормы",
                "probability_disease": 0.2,
                "probability_healthy": 0.8,
            },
            candidate_df=pd.DataFrame(),
            background_df=pd.DataFrame(),
            consistency_text="Интерпретация спектральных признаков согласуется с прогнозом модели.",
        )
        pdf_bytes = build_patient_report_pdf(report_data)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)


if __name__ == "__main__":
    unittest.main()
