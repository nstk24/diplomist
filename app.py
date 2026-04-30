from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from raman_webapp.analysis import (
    annotate_peak_importance,
    compute_spectral_importance,
    detect_spectral_peaks,
    match_peaks_to_reference_bands,
    detect_informative_region,
    detect_signal_quality_region,
    compute_pca_projection,
    compute_tsne_projection,
    compute_umap_projection,
    mean_std_by_class,
    snr_statistics,
    snr_statistics_from_values,
)
from raman_webapp.data import (
    ProcessedDataset,
    ExternalSpectrum,
    load_external_spectrum_csv,
    load_reference_band_library,
    load_raw_excel,
    preprocess_raw_dataset,
    split_train_holdout,
)
from raman_webapp.modeling import (
    evaluate_holdout,
    generate_model_interpretation,
    predict_patient,
    run_modeling,
)
from raman_webapp.patient_report import (
    build_compact_peak_table,
    build_detailed_peak_table,
    build_patient_report_data,
    build_patient_report_markdown,
    build_patient_report_pdf,
)
from raman_webapp.gemini_analysis import generate_gemini_hypotheses, is_gemini_configured
from raman_webapp.preprocessing import batch_snr_raman
from raman_webapp.spectroscopy import (
    SERS_DEFAULT_TOLERANCE_CM,
    analyze_spectroscopy_peaks,
    build_reference_library_display_df,
    get_spectroscopy_mode_label,
    normalize_spectroscopy_mode,
)
from raman_webapp.spectral_references import (
    add_reference_comparison,
    build_peak_consistency_text,
    compute_reference_peak_statistics,
    effect_size_to_russian,
    evaluate_expected_sers_bands,
    normalize_aggregation_mode,
    summarize_reference_comparison,
)
from raman_webapp.visuals import (
    boxplot_snr_figure,
    coefficient_figure,
    histogram_snr_figure,
    model_comparison_figure,
    peak_detection_figure,
    roc_curve_figure,
    scatter_projection_figure,
    spectrum_line_figure,
    spectrum_with_band_figure,
    informative_region_figure,
)


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="Анализ спектров крови Raman/SERS", page_icon="🧪", layout="wide")


def get_available_excel_datasets() -> list[Path]:
    return sorted(ROOT.glob("*.xlsx"))


@st.cache_data(show_spinner=False)
def get_processed_dataset_from_path(dataset_path: str) -> ProcessedDataset:
    raw = load_raw_excel(Path(dataset_path))
    return preprocess_raw_dataset(raw)


@st.cache_data(show_spinner=False)
def get_processed_dataset_from_upload(file_bytes: bytes, file_name: str) -> ProcessedDataset:
    raw = load_raw_excel(BytesIO(file_bytes))
    return preprocess_raw_dataset(raw)


@st.cache_data(show_spinner=False)
def get_train_holdout_datasets_from_path(dataset_path: str) -> tuple[ProcessedDataset, ProcessedDataset | None]:
    dataset = get_processed_dataset_from_path(dataset_path)
    try:
        return split_train_holdout(dataset)
    except ValueError:
        return dataset, None


@st.cache_data(show_spinner=False)
def get_train_holdout_datasets_from_upload(
    file_bytes: bytes, file_name: str
) -> tuple[ProcessedDataset, ProcessedDataset | None]:
    dataset = get_processed_dataset_from_upload(file_bytes, file_name)
    try:
        return split_train_holdout(dataset)
    except ValueError:
        return dataset, None


@st.cache_data(show_spinner=False)
def get_analysis_summary(dataset: ProcessedDataset) -> dict[str, np.ndarray]:
    return mean_std_by_class(dataset)


@st.cache_data(show_spinner=False)
def get_snr_summary(dataset: ProcessedDataset) -> dict[str, float | pd.DataFrame]:
    return snr_statistics(dataset)


@st.cache_data(show_spinner=False)
def get_pca(dataset: ProcessedDataset):
    return compute_pca_projection(dataset)


@st.cache_data(show_spinner=False)
def get_tsne(dataset: ProcessedDataset):
    return compute_tsne_projection(dataset)


@st.cache_data(show_spinner=False)
def get_umap(dataset: ProcessedDataset):
    return compute_umap_projection(dataset)


@st.cache_data(show_spinner=False)
def get_modeling_report(dataset: ProcessedDataset):
    return run_modeling(dataset)


@st.cache_data(show_spinner=False)
def get_holdout_report(train_dataset: ProcessedDataset, holdout_dataset: ProcessedDataset):
    return evaluate_holdout(get_modeling_report(train_dataset), holdout_dataset)


@st.cache_data(show_spinner=False)
def get_informative_region(dataset: ProcessedDataset):
    return detect_informative_region(dataset)


@st.cache_data(show_spinner=False)
def get_signal_quality_region(dataset: ProcessedDataset):
    return detect_signal_quality_region(dataset)


@st.cache_data(show_spinner=False)
def get_snr_basis_dataset_from_path(dataset_path: str) -> ProcessedDataset:
    return get_processed_dataset_from_path(dataset_path)


@st.cache_data(show_spinner=False)
def get_snr_basis_dataset_from_upload(file_bytes: bytes, file_name: str) -> ProcessedDataset:
    return get_processed_dataset_from_upload(file_bytes, file_name)


@st.cache_data(show_spinner=False)
def get_custom_snr_values(
    dataset: ProcessedDataset,
    signal_low: float,
    signal_high: float,
    noise_low: float,
    noise_high: float,
    k: float,
) -> np.ndarray:
    X_source = dataset.X_snr if dataset.X_snr is not None else dataset.X
    return batch_snr_raman(
        X_source,
        dataset.wavenumber,
        signal_region=(signal_low, signal_high),
        noise_region=(noise_low, noise_high),
        k=k,
    )


@st.cache_data(show_spinner=False)
def get_reference_band_library(spectroscopy_mode: str) -> pd.DataFrame:
    return load_reference_band_library(spectroscopy_mode=spectroscopy_mode)


@st.cache_data(show_spinner=False)
def get_spectral_importance(dataset: ProcessedDataset):
    return compute_spectral_importance(dataset)


def _priority_to_russian(priority: str) -> str:
    return {
        "high": "высокий",
        "medium": "средний",
        "support": "вспомогательный",
    }.get(priority, priority)


def _match_type_to_russian(match_type: str) -> str:
    return {
        "exact": "точное",
        "close": "близкое",
    }.get(match_type, match_type)


def _format_sers_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    return summary_df.rename(
        columns={
            "group": "Группа признаков",
            "количество": "Количество пиков",
            "пики": "Полосы / диапазоны",
            "роль": "Роль",
        }
    )


def _format_sers_detected_peaks_df(peaks_df: pd.DataFrame) -> pd.DataFrame:
    if peaks_df.empty:
        return peaks_df
    display_df = peaks_df.copy()
    display_df["peak_present"] = display_df["peak_present"].map(lambda value: "да" if bool(value) else "нет")
    display_df = display_df.rename(
        columns={
            "group": "Группа признаков",
            "assignment": "Молекулярное отнесение",
            "associated_condition": "Связь с состоянием",
            "interpretation": "Биохимическая интерпретация",
            "direction": "Направление изменения",
            "reliability": "Надёжность",
            "peak_present": "Пик обнаружен",
            "presence_label": "Статус",
            "measured_shift_cm": "Измеренное положение, см⁻¹",
            "expected_position_cm": "Ожидаемое положение / диапазон, см⁻¹",
            "intensity": "Интенсивность",
            "normalized_intensity": "Нормированная интенсивность",
            "match_score": "Оценка совпадения",
            "notes": "Примечания",
        }
    )
    preferred_columns = [
        "Группа признаков",
        "Молекулярное отнесение",
        "Связь с состоянием",
        "Биохимическая интерпретация",
        "Направление изменения",
        "Надёжность",
        "Пик обнаружен",
        "Статус",
        "Измеренное положение, см⁻¹",
        "Ожидаемое положение / диапазон, см⁻¹",
        "Интенсивность",
        "Нормированная интенсивность",
        "Оценка совпадения",
        "Примечания",
    ]
    return display_df[[column for column in preferred_columns if column in display_df.columns]]


def _prediction_label_to_russian(label: str) -> str:
    return {
        "Healthy": "спектральный профиль нормы",
        "Disease": "патологический спектральный профиль",
    }.get(str(label), str(label))


def _format_reference_comparison_df(peaks_df: pd.DataFrame, spectroscopy_mode: str) -> pd.DataFrame:
    if peaks_df.empty:
        return peaks_df
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    display_df = peaks_df.copy()
    rename_map = {
        "expected_position_cm": "Ожидаемый пик, см⁻¹",
        "measured_shift_cm": "Найденный пик, см⁻¹",
        "group": "Группа признаков",
        "label": "Группа признаков",
        "assignment": "Молекулярное отнесение",
        "patient_reference_intensity": "Интенсивность пациента",
        "mean_healthy": "Среднее у здоровых",
        "std_healthy": "Std у здоровых",
        "median_healthy": "Медиана у здоровых",
        "iqr_healthy": "IQR у здоровых",
        "mean_disease": "Среднее у больных",
        "std_disease": "Std у больных",
        "z_score": "Z-score",
        "deviation_label": "Отклонение от контроля",
        "interpretation": "Интерпретация",
        "clinical_hint": "Интерпретация",
        "reliability": "Надёжность",
        "reference_warning": "Предупреждение по сравнению",
        "notes": "Примечания",
    }
    display_df = display_df.rename(columns=rename_map)
    if mode == "raman":
        preferred = [
            "Ожидаемый пик, см⁻¹",
            "Найденный пик, см⁻¹",
            "Группа признаков",
            "Молекулярное отнесение",
            "Интенсивность пациента",
            "Среднее у здоровых",
            "Std у здоровых",
            "Z-score",
            "Отклонение от контроля",
            "Интерпретация",
            "Предупреждение по сравнению",
            "Примечания",
        ]
    else:
        preferred = [
            "Ожидаемый пик, см⁻¹",
            "Найденный пик, см⁻¹",
            "Группа признаков",
            "Молекулярное отнесение",
            "Интенсивность пациента",
            "Среднее у здоровых",
            "Std у здоровых",
            "Медиана у здоровых",
            "IQR у здоровых",
            "Среднее у больных",
            "Std у больных",
            "Z-score",
            "Отклонение от контроля",
            "Интерпретация",
            "Надёжность",
            "Предупреждение по сравнению",
            "Примечания",
        ]
    return display_df[[column for column in preferred if column in display_df.columns]]


def _aggregation_mode_to_russian(aggregation_mode: str) -> str:
    return {
        "max": "максимум в зоне",
        "mean": "среднее по зоне",
        "area": "площадь под кривой",
    }.get(normalize_aggregation_mode(aggregation_mode), aggregation_mode)


def _reference_deviation_counts(peaks_df: pd.DataFrame) -> tuple[int, int, int, int]:
    if peaks_df.empty or "deviation_label" not in peaks_df.columns:
        return 0, 0, 0, 0
    labels = peaks_df["deviation_label"].astype(str)
    higher_count = int(labels.str.contains("выше контрольной группы", na=False).sum())
    lower_count = int(labels.str.contains("ниже контрольной группы", na=False).sum())
    normal_count = int((labels == "в пределах условной спектральной нормы контрольной группы").sum())
    unknown_count = int(labels.str.contains("Недостаточно данных", na=False).sum())
    return higher_count, lower_count, normal_count, unknown_count


def _band_window_from_library_row(reference_row: pd.Series, spectroscopy_mode: str) -> tuple[float, float]:
    mode = normalize_spectroscopy_mode(spectroscopy_mode)
    if mode == "sers":
        if pd.notna(reference_row.get("shift_min_cm")) and pd.notna(reference_row.get("shift_max_cm")):
            return float(reference_row["shift_min_cm"]), float(reference_row["shift_max_cm"])
        center = float(reference_row["shift_cm"])
        tolerance = float(reference_row.get("tolerance_cm", SERS_DEFAULT_TOLERANCE_CM))
        return center - tolerance, center + tolerance
    return float(reference_row["low_cm1"]), float(reference_row["high_cm1"])


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
    return status


def _format_compact_expected_peaks_df(peaks_df: pd.DataFrame) -> pd.DataFrame:
    return build_compact_peak_table(peaks_df)


def _format_detailed_expected_peaks_df(peaks_df: pd.DataFrame) -> pd.DataFrame:
    return build_detailed_peak_table(peaks_df)


def _band_display_value(peaks_df: pd.DataFrame, band_id: str) -> str:
    band_mask = peaks_df["band_id"].astype(str) == str(band_id)
    if not band_mask.any():
        return str(band_id)
    row = peaks_df.loc[band_mask].iloc[0]
    for column_name in ("expected_position_cm", "label", "group", "band_id"):
        value = row.get(column_name)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return str(band_id)


def interpretation_block(dataset: ProcessedDataset, snr_summary: dict[str, float | pd.DataFrame], modeling_report) -> str:
    class_means = snr_summary["summary_df"]
    healthy_mean = float(class_means.loc[class_means["group"] == "Healthy", "mean"].iloc[0])
    disease_mean = float(class_means.loc[class_means["group"] == "Disease", "mean"].iloc[0])
    snr_text = (
        f"The disease group has mean SNR {disease_mean:.2f} versus {healthy_mean:.2f} in healthy samples. "
        f"Mann-Whitney p-value is {snr_summary['mann_whitney_p']:.4f}, so SNR alone should be treated as a descriptive signal, "
        "not as a standalone classifier."
    )
    size_text = (
        f"The working dataset contains {dataset.X.shape[0]} spectra and {dataset.X.shape[1]} spectral features "
        f"({int((dataset.y == 0).sum())} healthy, {int((dataset.y == 1).sum())} disease)."
    )
    model_text = generate_model_interpretation(modeling_report)
    return "\n\n".join([size_text, snr_text, model_text])


def external_spectrum_quality_check(
    external: ExternalSpectrum,
    full_dataset: ProcessedDataset,
    model_dataset: ProcessedDataset,
    modeling_low: float,
    modeling_high: float,
) -> dict[str, float | str | bool]:
    mask = (full_dataset.wavenumber >= modeling_low) & (full_dataset.wavenumber <= modeling_high)
    x_patient = external.snv_processed[mask]
    healthy_mean = model_dataset.X[model_dataset.y == 0].mean(axis=0)
    disease_mean = model_dataset.X[model_dataset.y == 1].mean(axis=0)

    corr_h = float(np.corrcoef(x_patient, healthy_mean)[0, 1])
    corr_d = float(np.corrcoef(x_patient, disease_mean)[0, 1])

    patient_roughness = float(np.std(np.diff(x_patient), ddof=1))
    train_roughness = np.std(np.diff(model_dataset.X[:, :], axis=1), axis=1, ddof=1)
    roughness_ratio = float(patient_roughness / (np.median(train_roughness) + 1e-12))

    is_suspicious = roughness_ratio > 3.0 or max(corr_h, corr_d) < 0.15
    reason = (
        "Загруженный спектр выглядит значительно более шумным, чем обучающее распределение, или слабо коррелирует со средними спектрами обоих классов."
        if is_suspicious
        else "Загруженный спектр находится в правдоподобных пределах относительно обучающего распределения."
    )
    return {
        "corr_healthy": corr_h,
        "corr_disease": corr_d,
        "roughness_ratio": roughness_ratio,
        "is_suspicious": is_suspicious,
        "reason": reason,
    }


st.title("Приложение для анализа Raman/SERS-спектров крови")
st.caption("Локальное приложение для предобработки, анализа спектров, сравнения моделей и интерпретации признаков.")

with st.sidebar:
    st.header("Настройки")
    spectroscopy_mode = normalize_spectroscopy_mode(
        st.radio(
            "Режим интерпретации пиков",
            options=["raman", "sers"],
            index=0,
            format_func=get_spectroscopy_mode_label,
            help="Этот режим определяет справочник пиков и правила интерпретации спектральных признаков. Прогноз модели строится по обучающему датасету и не изменяет свой физический тип автоматически.",
        )
    )
    st.caption(
        "Этот режим определяет справочник пиков и правила интерпретации спектральных признаков. "
        "Прогноз модели строится по обучающему датасету и не изменяет свой физический тип автоматически."
    )
    if spectroscopy_mode == "sers":
        st.warning(
            "Для корректного SERS-прогноза модель должна быть обучена на SERS-спектрах, полученных в сопоставимых условиях. "
            "В текущей версии выбранный режим SERS используется прежде всего для интерпретации пиков."
        )
    else:
        st.info(
            "Интерпретация Raman-пиков выполняется по справочнику обычной рамановской спектроскопии и зависит от предобработки, качества спектра и состава обучающего датасета."
        )
    dataset_source_mode = st.radio("Обучающий датасет", ["Файл проекта Excel", "Загрузить Excel-файл"], index=0)

    if dataset_source_mode == "Файл проекта Excel":
        excel_files = get_available_excel_datasets()
        if not excel_files:
            st.error("No `.xlsx` files were found in the project folder.")
            st.stop()
        selected_path = st.selectbox(
            "Excel dataset",
            options=excel_files,
            index=0,
            format_func=lambda path: path.name,
        )
        selected_dataset_label = selected_path.name
        st.write(f"Training will be recomputed from `{selected_path.name}`.")
        dataset, holdout_dataset = get_train_holdout_datasets_from_path(str(selected_path))
        snr_basis_dataset = get_snr_basis_dataset_from_path(str(selected_path))
    else:
        uploaded_dataset = st.file_uploader("Загрузите Excel-датасет", type=["xlsx"], key="training_dataset_xlsx")
        if uploaded_dataset is None:
            st.info("Upload an `.xlsx` file to start training on a custom dataset.")
            st.stop()
        uploaded_bytes = uploaded_dataset.getvalue()
        selected_dataset_label = uploaded_dataset.name
        st.write(f"Training will be recomputed from uploaded file `{uploaded_dataset.name}`.")
        dataset, holdout_dataset = get_train_holdout_datasets_from_upload(uploaded_bytes, uploaded_dataset.name)
        snr_basis_dataset = get_snr_basis_dataset_from_upload(uploaded_bytes, uploaded_dataset.name)

informative_region = get_informative_region(dataset)
signal_quality_region = get_signal_quality_region(dataset)
analysis_summary = get_analysis_summary(dataset)
reference_band_library = get_reference_band_library(spectroscopy_mode)
spectral_importance = get_spectral_importance(dataset)
try:
    snr_reference_dataset, _ = split_train_holdout(snr_basis_dataset)
except ValueError:
    snr_reference_dataset = snr_basis_dataset

wn_min = float(snr_reference_dataset.wavenumber.min())
wn_max = float(snr_reference_dataset.wavenumber.max())
wn_step = float(snr_reference_dataset.wavenumber[1] - snr_reference_dataset.wavenumber[0])

with st.sidebar:
    st.subheader("Область моделирования")
    modeling_region_mode = st.radio(
        "Рабочий диапазон",
        ["Авто", "Авто (по качеству сигнала)", "Ручной"],
        index=0,
        help="Этот диапазон используется для PCA/t-SNE/UMAP, обучения моделей и интерпретации коэффициентов.",
    )
    manual_model_window = st.slider(
        "Окно моделирования (см^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(signal_quality_region.low, signal_quality_region.high),
        step=wn_step,
        disabled=modeling_region_mode != "Ручной",
    )

    st.subheader("Окна SNR")
    signal_window = st.slider(
        "Окно сигнала (см^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(max(wn_min, 400.0), min(wn_max, 1700.0)),
        step=wn_step,
    )
    noise_window = st.slider(
        "Окно шума (см^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(max(wn_min, 1800.0), min(wn_max, 1950.0)),
        step=wn_step,
    )
    snr_k = st.number_input("Делитель SNR k", min_value=0.1, max_value=20.0, value=6.0, step=0.1)

if modeling_region_mode == "Ручной":
    modeling_low, modeling_high = manual_model_window
    modeling_region_label = f"{modeling_low:.1f}-{modeling_high:.1f} см^-1 (ручной режим)"
elif modeling_region_mode == "Авто (по качеству сигнала)":
    modeling_low, modeling_high = signal_quality_region.low, signal_quality_region.high
    modeling_region_label = f"{modeling_low:.1f}-{modeling_high:.1f} см^-1 (авто: качество сигнала)"
else:
    modeling_low, modeling_high = informative_region.low, informative_region.high
    modeling_region_label = f"{modeling_low:.1f}-{modeling_high:.1f} см^-1 (авто: справочник зон)"

model_dataset = dataset.select_wavenumber_range(modeling_low, modeling_high)

custom_snr_values = get_custom_snr_values(
    snr_reference_dataset,
    signal_window[0],
    signal_window[1],
    noise_window[0],
    noise_window[1],
    snr_k,
)
snr_summary = snr_statistics_from_values(dataset.y, custom_snr_values)

tab_overview, tab_snr, tab_proj, tab_models, tab_interpret, tab_patient_prediction = st.tabs(
    ["Обзор", "SNR", "Проекции", "Модели", "Интерпретация", "Прогноз пациента"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Spectra", dataset.X.shape[0])
    col2.metric("Features", dataset.X.shape[1])
    col3.metric("Healthy / Disease", f"{int((dataset.y == 0).sum())} / {int((dataset.y == 1).sum())}")
    st.caption(f"Current dataset: `{selected_dataset_label}`")
    if holdout_dataset is not None:
        st.info(
            "Training views in this app use the training subset only. "
            "The 10 exported holdout patients (5 healthy, 5 disease) are excluded from training and reserved for honest checking."
        )
    else:
        st.info(
            "This dataset does not contain the predefined holdout patient names, so the full dataset is used for training."
        )
    with st.expander("Reference band library by priority", expanded=False):
        st.write(
            "This local library is centered on the current curated CVD-associated zones and groups them into `high` and `medium` clinical-interest levels. "
            "It is intended for physician-facing interpretation support, not direct diagnosis."
        )
        st.dataframe(
            build_reference_library_display_df(reference_band_library, spectroscopy_mode),
            use_container_width=True,
        )

    healthy = dataset.X[dataset.y == 0]
    disease = dataset.X[dataset.y == 1]
    sem_healthy = healthy.std(axis=0, ddof=1) / np.sqrt(healthy.shape[0])
    sem_disease = disease.std(axis=0, ddof=1) / np.sqrt(disease.shape[0])

    fig_mean = spectrum_with_band_figure(
        dataset.wavenumber,
        analysis_summary["healthy_mean"],
        sem_healthy,
        analysis_summary["disease_mean"],
        sem_disease,
        "Healthy mean +/- SEM",
        "Disease mean +/- SEM",
        "Mean spectra by class",
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    fig_var = spectrum_line_figure(
        dataset.wavenumber,
        [analysis_summary["healthy_std"], analysis_summary["disease_std"]],
        ["Healthy std", "Disease std"],
        "Spectral variability by class",
        "Std",
    )
    st.plotly_chart(fig_var, use_container_width=True)

    st.write(
        "Слой предобработки использует ALS baseline correction, после чего применяется SNV-нормализация. "
        "Вся дальнейшая аналитика в приложении работает с нормализованным рабочим датасетом."
    )
    if modeling_region_mode == "Авто":
        st.info(
            f"Автоматическое окно моделирования: {modeling_region_label}. "
            "В текущей версии оно опирается на локальный справочник CVD-ассоциированных Raman-зон."
        )
        st.plotly_chart(
            informative_region_figure(
                dataset.wavenumber,
                informative_region.quality,
                informative_region.threshold,
                informative_region.low,
                informative_region.high,
            ),
            use_container_width=True,
        )
    elif modeling_region_mode == "Авто (по качеству сигнала)":
        st.info(
            f"Автоматическое окно по качеству сигнала: {modeling_region_label}. "
            "Этот режим использует baseline-corrected интенсивность, локальный шумовой фон и присутствие сигнала в нескольких спектрах, чтобы сохранить максимально широкую информативную область."
        )
        st.plotly_chart(
            informative_region_figure(
                dataset.wavenumber,
                signal_quality_region.quality,
                signal_quality_region.threshold,
                signal_quality_region.low,
                signal_quality_region.high,
            ),
            use_container_width=True,
        )
    else:
        st.info(f"Ручное окно моделирования: {modeling_region_label}.")

with tab_snr:
    st.write(
        "SNR is recomputed from the windows selected in the sidebar. "
        "The calculation uses baseline-corrected spectra before SNV whenever available."
    )
    st.caption(
        f"Current settings: signal window = {signal_window[0]:.1f}-{signal_window[1]:.1f} cm^-1, "
        f"noise window = {noise_window[0]:.1f}-{noise_window[1]:.1f} cm^-1, k = {snr_k:.1f}"
    )
    if not (signal_window[1] < noise_window[0] or noise_window[1] < signal_window[0]):
        st.warning("Signal and noise windows overlap. This usually makes the SNR estimate less meaningful.")
    left, right = st.columns(2)
    left.plotly_chart(boxplot_snr_figure(custom_snr_values, dataset.y), use_container_width=True)
    right.plotly_chart(histogram_snr_figure(custom_snr_values, dataset.y), use_container_width=True)

    st.dataframe(snr_summary["summary_df"], use_container_width=True)
    stats_cols = st.columns(5)
    stats_cols[0].metric("Shapiro p (Healthy)", f"{snr_summary['shapiro_healthy_p']:.3g}")
    stats_cols[1].metric("Shapiro p (Disease)", f"{snr_summary['shapiro_disease_p']:.3g}")
    stats_cols[2].metric("Mann-Whitney U", f"{snr_summary['mann_whitney_u']:.1f}")
    stats_cols[3].metric("Mann-Whitney p", f"{snr_summary['mann_whitney_p']:.3g}")
    stats_cols[4].metric("corr(SNR, label)", f"{snr_summary['snr_label_corr']:.3f}")

with tab_proj:
    st.write(
        f"Projection methods below use the current modeling region: `{modeling_region_label}`."
    )

    pca_proj = get_pca(model_dataset)
    st.plotly_chart(
        scatter_projection_figure(
            pca_proj.embedding,
            model_dataset.labels,
            f"PCA projection ({modeling_low:.0f}-{modeling_high:.0f} cm^-1)",
        ),
        use_container_width=True,
    )
    st.write(
        f"Explained variance ratio: PC1={pca_proj.explained_variance_ratio[0]:.3f}, "
        f"PC2={pca_proj.explained_variance_ratio[1]:.3f}, "
        f"sum={pca_proj.explained_variance_ratio.sum():.3f}"
    )

    tsne_proj = get_tsne(model_dataset)
    st.plotly_chart(
        scatter_projection_figure(
            tsne_proj.embedding,
            model_dataset.labels,
            f"t-SNE projection ({modeling_low:.0f}-{modeling_high:.0f} cm^-1)",
        ),
        use_container_width=True,
    )

    umap_proj = get_umap(model_dataset)
    if umap_proj is None:
        st.info("UMAP is optional. Install `umap-learn` to enable this view.")
    else:
        st.plotly_chart(
            scatter_projection_figure(
                umap_proj.embedding,
                model_dataset.labels,
                f"UMAP projection ({modeling_low:.0f}-{modeling_high:.0f} cm^-1)",
            ),
            use_container_width=True,
        )

with tab_models:
    st.write(
        f"Model training and nested CV use the current modeling region `{modeling_region_label}`. "
        "This is independent from the SNR signal/noise windows in the sidebar."
    )
    with st.spinner("Running model comparison and nested CV..."):
        modeling_report = get_modeling_report(model_dataset)

    st.subheader("Exploratory screening")
    st.dataframe(modeling_report.screening_df, use_container_width=True)

    st.subheader("Метрики лидера screening")
    left, right = st.columns([1, 2])
    screening_metrics_display = modeling_report.best_model_metrics_df.copy()
    screening_metrics_display = screening_metrics_display.rename(
        columns={
            "metric": "Метрика",
            "value": "Значение",
            "std": "Std (если доступно)",
        }
    )
    left.dataframe(screening_metrics_display, use_container_width=True)
    right.plotly_chart(
        model_comparison_figure(modeling_report.screening_df, modeling_report.nested_summary_df),
        use_container_width=True,
    )
    st.caption(
        f"Слева показаны cross-validation метрики лучшего кандидата этапа screening: `{modeling_report.screening_df.iloc[0]['model']}`. "
        "Они полезны для сравнения пайплайнов, но обычно выглядят оптимистичнее, чем nested CV для всей процедуры выбора модели."
    )

    st.subheader("Nested CV для всей процедуры выбора модели")
    nested_metrics_display = modeling_report.nested_summary_df.rename(
        columns={
            "metric": "Метрика",
            "mean": "Среднее",
            "std": "Std",
        }
    )
    st.dataframe(nested_metrics_display, use_container_width=True)
    st.caption(
        "Этот блок является основной внутренней оценкой качества в приложении, потому что выбор модели повторяется внутри внешних фолдов."
    )

    st.subheader("Selected pipeline frequency across outer folds")
    st.dataframe(
        modeling_report.selected_model_counts.rename_axis("pipeline").reset_index(name="count"),
        use_container_width=True,
    )
    st.success(f"Most frequently selected pipeline: {modeling_report.final_selected_model_name}")

    if holdout_dataset is not None:
        st.subheader("Holdout evaluation on 10 excluded patients")
        st.caption("These metrics are computed on the exported patients from `patient_csv_exports`, excluded from all training.")
        holdout_model_dataset = holdout_dataset.select_wavenumber_range(modeling_low, modeling_high)
        holdout_report = get_holdout_report(model_dataset, holdout_model_dataset)
        holdout_left, holdout_mid, holdout_right = st.columns([1, 1, 2])
        holdout_left.dataframe(holdout_report.summary_df, use_container_width=True)
        if holdout_report.roc_df is not None:
            holdout_auc_row = holdout_report.summary_df.loc[holdout_report.summary_df["metric"] == "roc_auc", "value"]
            holdout_auc = float(holdout_auc_row.iloc[0]) if not holdout_auc_row.empty else None
            holdout_mid.plotly_chart(
                roc_curve_figure(
                    holdout_report.roc_df,
                    "ROC-кривая лучшей модели на holdout",
                    auc_value=holdout_auc,
                ),
                use_container_width=True,
            )
        else:
            holdout_mid.info("ROC-кривая недоступна: в holdout должен присутствовать оба класса.")
        holdout_right.dataframe(holdout_report.predictions_df, use_container_width=True)
        st.warning(
            "Если метрики на holdout выглядят идеальными, это не обязательно означает реальную надежность модели. "
            "Сейчас holdout очень маленький: всего 10 пациентов, поэтому оценки могут быть нестабильными и слишком оптимистичными."
        )
    else:
        st.info("Holdout evaluation is unavailable for the current dataset because the predefined 10 patient IDs were not found.")

with tab_interpret:
    modeling_report = get_modeling_report(model_dataset)
    st.markdown(interpretation_block(model_dataset, snr_summary, modeling_report))
    st.info(
        f"This feature map is computed only on the current modeling region `{modeling_region_label}`."
    )
    st.plotly_chart(
        coefficient_figure(
            model_dataset.wavenumber,
            modeling_report.coef_smoothed_space,
            f"Smoothed logistic-regression coefficients ({modeling_low:.0f}-{modeling_high:.0f} cm^-1)",
        ),
        use_container_width=True,
    )
    st.subheader("Top spectral bands")
    st.dataframe(modeling_report.top_bands_df, use_container_width=True)
    with st.expander("Top individual markers", expanded=False):
        st.dataframe(modeling_report.top_features_df, use_container_width=True)

with tab_patient_prediction:
    modeling_report = get_modeling_report(model_dataset)
    st.subheader("Прогноз состояния пациента")
    st.write(
        "Загрузите CSV-файл спектра пациента. Поддерживаются форматы: "
        "две числовые колонки (`wavenumber`, `intensity`), одна колонка интенсивности той же длины, "
        "что и эталонная ось, или одна строка интенсивностей той же длины."
    )
    if holdout_dataset is not None:
        st.caption(
            "Модель в этой вкладке обучена только на тренировочной части датасета. "
            "Файлы из `patient_csv_exports` остаются независимыми holdout-примерами."
        )
    else:
        st.caption("Модель в этой вкладке обучена на всём выбранном датасете.")
    uploaded_file = st.file_uploader("Загрузите CSV-файл спектра пациента", type=["csv"], key="patient_csv")

    if uploaded_file is not None:
        try:
            external = load_external_spectrum_csv(uploaded_file, dataset.wavenumber)
            external_model_vector = external.snv_processed[
                (dataset.wavenumber >= modeling_low) & (dataset.wavenumber <= modeling_high)
            ]
            quality = external_spectrum_quality_check(
                external,
                dataset,
                model_dataset,
                modeling_low,
                modeling_high,
            )

            st.caption(f"Режим чтения CSV: {external.parser_mode}")
            qc_cols = st.columns(3)
            qc_cols[0].metric("Корреляция со средним Healthy", f"{quality['corr_healthy']:.3f}")
            qc_cols[1].metric("Корреляция со средним Disease", f"{quality['corr_disease']:.3f}")
            qc_cols[2].metric("Коэффициент шероховатости", f"{quality['roughness_ratio']:.2f}")

            if quality["is_suspicious"]:
                st.error(
                    "Загруженный спектр плохо согласуется с обучающими данными. "
                    "Предсказание может быть ненадежным. Проверь формат CSV, порядок оси и preprocessing."
                )
            else:
                st.success("Проверка качества спектра пройдена.")
            st.write(str(quality["reason"]))
            prediction = predict_patient(modeling_report, external_model_vector)
            st.subheader("Результат прогноза")
            pred_cols = st.columns(4)
            prediction_label_ru = _prediction_label_to_russian(str(prediction["predicted_label"]))
            pred_cols[0].metric("Режим интерпретации пиков", get_spectroscopy_mode_label(spectroscopy_mode))
            pred_cols[1].metric("Предсказанный спектральный профиль", prediction_label_ru)
            pred_cols[2].metric(
                "Вероятность патологического спектрального профиля",
                f"{prediction['probability_disease']:.3f}",
            )
            pred_cols[3].metric("Вероятность спектрального профиля нормы", f"{prediction['probability_healthy']:.3f}")
            st.warning(
                "Результат не является медицинским диагнозом. Прогноз основан на сходстве спектра с группами обучающего датасета."
            )
            if spectroscopy_mode == "sers":
                st.warning(
                    "При выборе SERS-режима интерпретация пиков выполняется по отдельному SERS-справочнику и не должна считаться эквивалентом обычной рамановской спектроскопии."
                )

            mean_healthy = model_dataset.X[model_dataset.y == 0].mean(axis=0)
            mean_disease = model_dataset.X[model_dataset.y == 1].mean(axis=0)
            patient_fig = spectrum_line_figure(
                model_dataset.wavenumber,
                [mean_healthy, mean_disease, external_model_vector],
                ["Средний профиль нормы", "Средний патологический профиль", "Спектр пациента"],
                f"Сравнение спектра пациента со средними профилями ({modeling_low:.0f}-{modeling_high:.0f} см⁻¹)",
                "SNV-интенсивность",
            )
            st.plotly_chart(patient_fig, use_container_width=True)
            st.caption(
                f"Прогноз построен моделью `{modeling_report.final_selected_model_name}` в диапазоне {modeling_region_label}."
            )

            with st.expander("Показать интерпретацию пиков", expanded=False):
                st.write(
                    "Этот блок ищет локальные максимумы на спектре пациента и формирует дополнительную спектральную интерпретацию. "
                    "Он не изменяет результат модели, а помогает его интерпретировать."
                )
                peak_col1, peak_col2, peak_col3 = st.columns(3)
                peak_prominence = peak_col1.number_input(
                    "Минимальная prominence пика",
                    min_value=0.001,
                    max_value=5.0,
                    value=0.08,
                    step=0.01,
                    format="%.3f",
                    key="peak_prominence",
                )
                peak_distance = peak_col2.number_input(
                    "Минимальное расстояние между пиками (см⁻¹)",
                    min_value=1.0,
                    max_value=200.0,
                    value=12.0,
                    step=1.0,
                    key="peak_distance",
                )
                peak_top_k = int(
                    peak_col3.number_input(
                        "Сколько пиков показать",
                        min_value=3,
                        max_value=50,
                        value=15,
                        step=1,
                        key="peak_top_k",
                    )
                )
                match_col1, match_col2 = st.columns(2)
                peak_matching_mode_label = match_col1.radio(
                    "Режим сопоставления с библиотекой",
                    ["Близкое совпадение", "Точное совпадение"],
                    index=0,
                    horizontal=True,
                    help="По умолчанию используется близкое совпадение: пик может немного выходить за границы полосы.",
                )
                peak_tolerance_cm = match_col2.number_input(
                    "Допуск для близкого совпадения (см⁻¹)",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(SERS_DEFAULT_TOLERANCE_CM if spectroscopy_mode == "sers" else 5.0),
                    step=0.5,
                    format="%.1f",
                    key="peak_tolerance_cm",
                    disabled=peak_matching_mode_label != "Близкое совпадение",
                )

                peak_basis_mode = st.radio(
                    "Основа для поиска пиков",
                    ["SNV-нормированный спектр пациента", "Выровненный спектр после baseline correction"],
                    index=1,
                    horizontal=True,
                    help="Для сравнения с контрольной группой используется тот же тип интенсивности, что и для спектра пациента.",
                )
                reference_aggregation_label = st.radio(
                    "Как сравнивать интенсивность зоны с контрольной группой",
                    ["Максимум в зоне", "Среднее по зоне", "Площадь под кривой"],
                    index=0,
                    horizontal=True,
                    help="Для широких зон среднее или площадь часто устойчивее, чем локальный максимум.",
                )
                reference_aggregation_mode = {
                    "Максимум в зоне": "max",
                    "Среднее по зоне": "mean",
                    "Площадь под кривой": "area",
                }[reference_aggregation_label]
                if peak_basis_mode == "Выровненный спектр после baseline correction":
                    peak_plot_wavenumber = dataset.wavenumber
                    peak_plot_intensity = external.baseline_corrected
                    reference_intensity_basis = "baseline_corrected"
                    peak_plot_result = detect_spectral_peaks(
                        peak_plot_wavenumber,
                        peak_plot_intensity,
                        prominence=float(peak_prominence),
                        min_distance_cm=float(peak_distance),
                        top_k=peak_top_k,
                    )
                    peak_y_axis_title = "Интенсивность после baseline correction"
                else:
                    peak_plot_wavenumber = dataset.wavenumber
                    peak_plot_intensity = external.snv_processed
                    reference_intensity_basis = "snv"
                    peak_plot_result = detect_spectral_peaks(
                        peak_plot_wavenumber,
                        peak_plot_intensity,
                        prominence=float(peak_prominence),
                        min_distance_cm=float(peak_distance),
                        top_k=peak_top_k,
                    )
                    peak_y_axis_title = "SNV-интенсивность"

                peak_analysis_result = analyze_spectroscopy_peaks(
                    peaks_df=peak_plot_result.peaks_df,
                    wavenumber=peak_plot_wavenumber,
                    intensity=peak_plot_intensity,
                    reference_library_df=reference_band_library,
                    spectroscopy_mode=spectroscopy_mode,
                    matching_mode="close" if peak_matching_mode_label == "Близкое совпадение" else "exact",
                    tolerance_cm=float(peak_tolerance_cm),
                )
                reference_stats_df = compute_reference_peak_statistics(
                    dataset=dataset,
                    reference_library_df=reference_band_library,
                    spectroscopy_mode=spectroscopy_mode,
                    intensity_basis=reference_intensity_basis,
                    aggregation_mode=reference_aggregation_mode,
                )
                annotated_peak_df = annotate_peak_importance(peak_plot_result.peaks_df, spectral_importance)
                peak_labels_by_wavenumber: dict[float, str] = {}
                if spectroscopy_mode == "raman":
                    for _, row in peak_analysis_result.matched_peaks_df.iterrows():
                        peak_wn = float(row["wavenumber_cm-1"])
                        if peak_wn not in peak_labels_by_wavenumber:
                            peak_labels_by_wavenumber[peak_wn] = (
                                f"{row['label']} ({_match_type_to_russian(str(row['match_type']))})"
                            )
                else:
                    for _, row in peak_analysis_result.matched_peaks_df.iterrows():
                        if bool(row["peak_present"]):
                            peak_wn = float(row["measured_shift_cm"])
                            if peak_wn not in peak_labels_by_wavenumber:
                                peak_labels_by_wavenumber[peak_wn] = str(row["group"])
                peak_text = [
                    peak_labels_by_wavenumber.get(float(peak_plot_wavenumber[idx]), "")
                    for idx in peak_plot_result.peak_indices
                ]

                st.subheader("Спектральные признаки")
                st.plotly_chart(
                    peak_detection_figure(
                        peak_plot_wavenumber,
                        peak_plot_intensity,
                        peak_plot_result.peak_indices,
                        "Найденные локальные пики на спектре пациента",
                        peak_y_axis_title,
                        peak_text=peak_text,
                    ),
                    use_container_width=True,
                )

                if peak_plot_result.peaks_df.empty:
                    st.info("С текущими порогами пики не найдены. Попробуйте уменьшить минимальную prominence.")
                else:
                    peak_display_df = annotated_peak_df.copy()
                    peak_display_df["class_importance_priority"] = peak_display_df["class_importance_priority"].map(
                        _priority_to_russian
                    )
                    peak_display_df = peak_display_df.rename(
                        columns={
                            "rank_by_prominence": "Ранг",
                            "wavenumber_cm-1": "Волновое число, см⁻¹",
                            "intensity": "Интенсивность",
                            "prominence": "Prominence",
                            "width_cm-1": "Ширина, см⁻¹",
                            "class_importance_score": "Значимость по различию классов",
                            "class_importance_priority": "Приоритет по различию классов",
                        }
                    )
                    st.dataframe(peak_display_df, use_container_width=True)
                    st.caption(
                        "Значимость по различию классов показывает, насколько локальная область помогает разделять группы обучающего датасета."
                    )

                    st.subheader("Интерпретация найденных пиков")
                    st.markdown(peak_analysis_result.interpretation)
                    st.warning(
                        "Отклонение интенсивности пика рассчитано относительно контрольной группы текущего датасета, а не относительно клинической нормы концентрации биомаркера."
                    )
                    st.warning("Наличие отдельного пика не доказывает наличие сердечно-сосудистого заболевания.")
                    if spectroscopy_mode == "sers":
                        st.warning("SERS-спектры зависят от типа подложки, пробоподготовки и условий регистрации.")
                    else:
                        st.warning("Интерпретация Raman-пиков зависит от качества спектра, предобработки и состава обучающего датасета.")
                    for warning_text in peak_analysis_result.warnings:
                        st.warning(warning_text)

                    if spectroscopy_mode == "raman":
                        summary_display_df = peak_analysis_result.summary_df.copy()
                        if not summary_display_df.empty:
                            summary_display_df["priority"] = summary_display_df["priority"].map(_priority_to_russian)
                            summary_display_df["match_type"] = summary_display_df["match_type"].map(_match_type_to_russian)
                            summary_display_df = summary_display_df.rename(
                                columns={
                                    "priority": "Приоритет",
                                    "match_type": "Тип совпадения",
                                    "n_matches": "Количество совпадений",
                                    "labels": "Совпавшие полосы",
                                }
                            )
                            st.dataframe(summary_display_df, use_container_width=True)

                        candidate_reference_df = add_reference_comparison(
                            peak_analysis_result.matched_peaks_df,
                            reference_stats_df,
                            patient_intensity_col="intensity",
                            allow_reference_comparison=True,
                            patient_wavenumber=peak_plot_wavenumber,
                            patient_intensity=peak_plot_intensity,
                            reference_library_df=reference_band_library,
                            spectroscopy_mode=spectroscopy_mode,
                            aggregation_mode=reference_aggregation_mode,
                        )
                        background_reference_df = pd.DataFrame()
                        if candidate_reference_df.empty:
                            st.info(
                                "Ни один из найденных пиков не попал в текущую библиотеку характерных полос. Это не исключает наличие изменений, а только отсутствие совпадений со справочником."
                            )
                        else:
                            st.subheader("Краткая интерпретация спектральных признаков")
                            st.dataframe(_format_compact_expected_peaks_df(candidate_reference_df), use_container_width=True)
                            st.subheader("Сравнение с контрольной группой")
                            st.caption(
                                f"Основа сравнения: {peak_y_axis_title.lower()}. Метрика зоны: {_aggregation_mode_to_russian(reference_aggregation_mode)}."
                            )
                            higher_count, lower_count, normal_count, unknown_count = _reference_deviation_counts(candidate_reference_df)
                            st.info(
                                "По сравнению с контрольной группой здоровых доноров "
                                f"{higher_count} найденных признаков имеют интенсивность выше условной спектральной нормы, "
                                f"{lower_count} признаков ниже контрольной группы, "
                                f"{normal_count} признаков находятся в пределах контрольного диапазона. "
                                f"Для {unknown_count} признаков недостаточно данных для расчёта отклонения."
                            )
                            selected_band_id = st.selectbox(
                                "Зона для локального референсного графика",
                                candidate_reference_df["band_id"].astype(str).tolist(),
                                format_func=lambda band_id: _band_display_value(candidate_reference_df, str(band_id)),
                                key="raman_reference_band_plot",
                            )
                            reference_row = reference_band_library.loc[
                                reference_band_library["band_id"].astype(str) == str(selected_band_id)
                            ].iloc[0]
                            band_low, band_high = _band_window_from_library_row(reference_row, spectroscopy_mode)
                            plot_padding = 15.0
                            plot_mask = (peak_plot_wavenumber >= band_low - plot_padding) & (peak_plot_wavenumber <= band_high + plot_padding)
                            reference_X = dataset.X_snr if reference_intensity_basis == "baseline_corrected" and dataset.X_snr is not None else dataset.X
                            healthy_profile = reference_X[dataset.y == 0].mean(axis=0)
                            disease_profile = reference_X[dataset.y == 1].mean(axis=0) if np.any(dataset.y == 1) else np.full_like(healthy_profile, np.nan)
                            st.plotly_chart(
                                spectrum_line_figure(
                                    peak_plot_wavenumber[plot_mask],
                                    [
                                        healthy_profile[plot_mask],
                                        disease_profile[plot_mask],
                                        peak_plot_intensity[plot_mask],
                                    ],
                                    ["Средний профиль нормы", "Средний патологический профиль", "Спектр пациента"],
                                    f"Локальное сравнение в зоне {band_low:.2f}-{band_high:.2f} см⁻¹",
                                    peak_y_axis_title,
                                ),
                                use_container_width=True,
                            )
                    else:
                        st.caption(f"Выбран режим интерпретации пиков: {get_spectroscopy_mode_label(spectroscopy_mode)}.")
                        expected_sers_df = evaluate_expected_sers_bands(
                            patient_wavenumber=peak_plot_wavenumber,
                            patient_intensity=peak_plot_intensity,
                            reference_library_df=reference_band_library,
                            reference_stats_df=reference_stats_df,
                            aggregation_mode=reference_aggregation_mode,
                        )
                        candidate_reference_df = expected_sers_df.loc[expected_sers_df["peak_role"] == "candidate"].copy()
                        background_reference_df = expected_sers_df.loc[expected_sers_df["peak_role"] == "background"].copy()

                        sers_metrics = st.columns(4)
                        sers_metrics[0].metric("Кандидатные признаки патологии", int(len(candidate_reference_df)))
                        sers_metrics[1].metric("Фоновые признаки сыворотки", int(len(background_reference_df)))
                        sers_metrics[2].metric(
                            "Оценка SERS-паттерна",
                            f"{peak_analysis_result.sers_cvd_score:.2f}" if peak_analysis_result.sers_cvd_score is not None else "0.00",
                        )
                        sers_metrics[3].metric(
                            "Уровень SERS-паттерна",
                            peak_analysis_result.sers_cvd_pattern_level or "не определён",
                        )

                        st.subheader("Ожидаемые спектральные признаки")
                        st.dataframe(_format_compact_expected_peaks_df(expected_sers_df), use_container_width=True)

                        st.subheader("Краткая интерпретация спектральных признаков")
                        st.dataframe(_format_compact_expected_peaks_df(candidate_reference_df), use_container_width=True)
                        if not background_reference_df.empty:
                            st.subheader("Фоновые признаки сыворотки")
                            st.dataframe(_format_compact_expected_peaks_df(background_reference_df), use_container_width=True)

                        st.subheader("Сравнение с контрольной группой")
                        st.caption(
                            f"Основа сравнения: {peak_y_axis_title.lower()}. Метрика зоны: {_aggregation_mode_to_russian(reference_aggregation_mode)}."
                        )
                        higher_count, lower_count, normal_count, unknown_count = _reference_deviation_counts(candidate_reference_df)
                        st.info(
                            "По сравнению с контрольной группой здоровых доноров "
                            f"{higher_count} найденных признаков имеют интенсивность выше условной спектральной нормы, "
                            f"{lower_count} признаков ниже контрольной группы, "
                            f"{normal_count} признаков находятся в пределах контрольного диапазона. "
                            f"Для {unknown_count} признаков недостаточно данных для расчёта отклонения."
                        )
                        st.caption(summarize_reference_comparison(candidate_reference_df, background_reference_df, spectroscopy_mode))

                        selected_band_id = st.selectbox(
                            "Зона для локального референсного графика",
                            expected_sers_df["band_id"].astype(str).tolist(),
                            format_func=lambda band_id: _band_display_value(expected_sers_df, str(band_id)),
                            key="sers_reference_band_plot",
                        )
                        reference_row = reference_band_library.loc[
                            reference_band_library["band_id"].astype(str) == str(selected_band_id)
                        ].iloc[0]
                        band_low, band_high = _band_window_from_library_row(reference_row, spectroscopy_mode)
                        plot_padding = 15.0
                        plot_mask = (peak_plot_wavenumber >= band_low - plot_padding) & (peak_plot_wavenumber <= band_high + plot_padding)
                        reference_X = dataset.X_snr if reference_intensity_basis == "baseline_corrected" and dataset.X_snr is not None else dataset.X
                        healthy_profile = reference_X[dataset.y == 0].mean(axis=0)
                        disease_profile = reference_X[dataset.y == 1].mean(axis=0) if np.any(dataset.y == 1) else np.full_like(healthy_profile, np.nan)
                        st.plotly_chart(
                            spectrum_line_figure(
                                peak_plot_wavenumber[plot_mask],
                                [
                                    healthy_profile[plot_mask],
                                    disease_profile[plot_mask],
                                    peak_plot_intensity[plot_mask],
                                ],
                                ["Средний профиль нормы", "Средний патологический профиль", "Спектр пациента"],
                                f"Локальное сравнение в зоне {band_low:.2f}-{band_high:.2f} см⁻¹",
                                peak_y_axis_title,
                            ),
                            use_container_width=True,
                        )

                    consistency_text = build_peak_consistency_text(str(prediction["predicted_label"]), candidate_reference_df)
                    st.subheader("Согласованность прогноза и спектральных признаков")
                    st.info(consistency_text)

                    with st.expander("Показать подробную таблицу для отчёта", expanded=False):
                        detailed_df = pd.concat([candidate_reference_df, background_reference_df], ignore_index=True) if not candidate_reference_df.empty or not background_reference_df.empty else pd.DataFrame()
                        st.subheader("Подробная интерпретация спектральных признаков")
                        if detailed_df.empty:
                            st.info("Подробные данные для отчёта пока отсутствуют.")
                        else:
                            st.dataframe(_format_detailed_expected_peaks_df(detailed_df), use_container_width=True)

                    report_prediction = dict(prediction)
                    report_prediction["predicted_label_ru"] = prediction_label_ru
                    report_data = build_patient_report_data(
                        spectroscopy_mode=spectroscopy_mode,
                        prediction=report_prediction,
                        candidate_df=candidate_reference_df,
                        background_df=background_reference_df,
                        consistency_text=consistency_text,
                    )
                    report_markdown = build_patient_report_markdown(report_data)
                    report_pdf = build_patient_report_pdf(report_data)
                    st.download_button(
                        "Скачать отчёт в PDF",
                        data=report_pdf,
                        file_name="patient_report.pdf",
                        mime="application/pdf",
                        key="download_patient_report_pdf",
                    )
                    with st.expander("Предпросмотр текста отчёта", expanded=False):
                        st.markdown(report_markdown)

                st.subheader("Gemini: гипотезы по спектральным областям")
                gemini_ready, gemini_status = is_gemini_configured()
                if gemini_ready:
                    st.caption(
                        "Gemini получает только агрегированные спектральные области пиков и краткий контекст. "
                        "Сырые массивы спектра и наши грубые назначения веществ не отправляются."
                    )
                else:
                    st.info(
                        "Gemini сейчас недоступен. "
                        f"{gemini_status} Запустите приложение из того же терминала, где задан GEMINI_API_KEY."
                    )

                gemini_button = st.button(
                    "Сгенерировать гипотезы через Gemini",
                    key="gemini_generate_hypotheses",
                    disabled=not gemini_ready or peak_plot_result.peaks_df.empty,
                )
                gemini_state_key = "gemini_hypotheses_response"
                if gemini_button:
                    with st.spinner("Gemini анализирует спектральные области..."):
                        gemini_result = generate_gemini_hypotheses(
                            peaks_df=annotated_peak_df,
                            prediction_probability_disease=float(prediction["probability_disease"]),
                            peak_basis_label=peak_basis_mode,
                            spectrum_quality_text=str(quality["reason"]),
                            model_name=str(modeling_report.final_selected_model_name),
                        )
                    st.session_state[gemini_state_key] = {
                        "ok": gemini_result.ok,
                        "text": gemini_result.text,
                        "prompt_payload": gemini_result.prompt_payload,
                        "model_used": gemini_result.model_used,
                    }

                gemini_state = st.session_state.get(gemini_state_key)
                if gemini_state:
                    if gemini_state["ok"]:
                        st.warning(
                            "Ниже приведена модельная интерпретация для поддержки врача. "
                            "Это не диагноз и не подтверждение конкретного вещества."
                        )
                        if gemini_state.get("model_used"):
                            st.caption(f"Ответ получен от модели: `{gemini_state['model_used']}`")
                        st.markdown(gemini_state["text"])
                        with st.expander("Какие данные были отправлены в Gemini", expanded=False):
                            st.json(gemini_state["prompt_payload"])
                    else:
                        st.error(str(gemini_state["text"]))
        except Exception as exc:
            st.error(f"Не удалось обработать загруженный спектр: {exc}")
