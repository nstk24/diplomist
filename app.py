from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from raman_webapp.analysis import (
    detect_informative_region,
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
    load_processed_csvs,
    load_raw_excel,
    preprocess_raw_dataset,
)
from raman_webapp.modeling import generate_model_interpretation, predict_patient, run_modeling
from raman_webapp.preprocessing import batch_snr_raman
from raman_webapp.visuals import (
    boxplot_snr_figure,
    coefficient_figure,
    histogram_snr_figure,
    model_comparison_figure,
    scatter_projection_figure,
    spectrum_line_figure,
    spectrum_with_band_figure,
    informative_region_figure,
)


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="Raman Blood Spectra App", page_icon="🧪", layout="wide")


@st.cache_data(show_spinner=False)
def get_processed_dataset(data_source: str) -> ProcessedDataset:
    if data_source == "Use precomputed CSV":
        return load_processed_csvs(
            ROOT / "X_clean.csv",
            ROOT / "y_clean.csv",
            ROOT / "wavenumber.csv",
            ROOT / "snr_clean.csv",
        )

    raw = load_raw_excel(ROOT / "Raman_krov_SSZ-zdorovye.xlsx")
    return preprocess_raw_dataset(raw)


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
def get_informative_region(dataset: ProcessedDataset):
    return detect_informative_region(dataset)


@st.cache_data(show_spinner=False)
def get_snr_basis_dataset() -> ProcessedDataset:
    raw = load_raw_excel(ROOT / "Raman_krov_SSZ-zdorovye.xlsx")
    return preprocess_raw_dataset(raw)


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
        "The uploaded spectrum looks much rougher than the training distribution or poorly correlated with both class means."
        if is_suspicious
        else "The uploaded spectrum is within a plausible range for the training distribution."
    )
    return {
        "corr_healthy": corr_h,
        "corr_disease": corr_d,
        "roughness_ratio": roughness_ratio,
        "is_suspicious": is_suspicious,
        "reason": reason,
    }


st.title("Raman Spectra Classification Workbench")
st.caption("Local app for preprocessing, exploratory analysis, model comparison and interpretation.")

with st.sidebar:
    st.header("Configuration")
    data_source = st.radio("Dataset source", ["Use precomputed CSV", "Recompute from Excel"], index=0)
    st.write("`Use precomputed CSV` starts fast. `Recompute from Excel` reruns baseline correction and SNV.")

dataset = get_processed_dataset(data_source)
informative_region = get_informative_region(dataset)
analysis_summary = get_analysis_summary(dataset)
snr_basis_dataset = get_snr_basis_dataset()
snr_reference_dataset = snr_basis_dataset if snr_basis_dataset.y.shape[0] == dataset.y.shape[0] else dataset

wn_min = float(snr_reference_dataset.wavenumber.min())
wn_max = float(snr_reference_dataset.wavenumber.max())
wn_step = float(snr_reference_dataset.wavenumber[1] - snr_reference_dataset.wavenumber[0])

with st.sidebar:
    st.subheader("Modeling Region")
    modeling_region_mode = st.radio(
        "Modeling range",
        ["Auto", "Manual"],
        index=0,
        help="This range is used for PCA/t-SNE/UMAP, model training and coefficient interpretation.",
    )
    manual_model_window = st.slider(
        "Modeling window (cm^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(informative_region.low, informative_region.high),
        step=wn_step,
        disabled=modeling_region_mode != "Manual",
    )

    st.subheader("SNR Windows")
    signal_window = st.slider(
        "Signal window (cm^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(max(wn_min, 400.0), min(wn_max, 1700.0)),
        step=wn_step,
    )
    noise_window = st.slider(
        "Noise window (cm^-1)",
        min_value=wn_min,
        max_value=wn_max,
        value=(max(wn_min, 1800.0), min(wn_max, 1950.0)),
        step=wn_step,
    )
    snr_k = st.number_input("SNR divisor k", min_value=0.1, max_value=20.0, value=6.0, step=0.1)

if modeling_region_mode == "Manual":
    modeling_low, modeling_high = manual_model_window
    modeling_region_label = f"{modeling_low:.1f}-{modeling_high:.1f} cm^-1 (manual)"
else:
    modeling_low, modeling_high = informative_region.low, informative_region.high
    modeling_region_label = f"{modeling_low:.1f}-{modeling_high:.1f} cm^-1 (auto)"

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
    ["Overview", "SNR", "Projections", "Models", "Interpretation", "Patient Prediction"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Spectra", dataset.X.shape[0])
    col2.metric("Features", dataset.X.shape[1])
    col3.metric("Healthy / Disease", f"{int((dataset.y == 0).sum())} / {int((dataset.y == 1).sum())}")

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
        "The preprocessing layer uses ALS baseline correction followed by SNV normalization. "
        "All downstream analysis in the app uses the normalized working dataset."
    )
    if modeling_region_mode == "Auto":
        st.info(f"Automatic informative window for modeling: {modeling_region_label}.")
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
    else:
        st.info(f"Manual modeling window: {modeling_region_label}.")

with tab_snr:
    st.write(
        "SNR is recomputed from the windows selected in the sidebar. "
        "The calculation uses baseline-corrected spectra before SNV whenever available."
    )
    if snr_reference_dataset is not snr_basis_dataset:
        st.warning(
            "The current dataset size does not match the Excel source exactly, so custom SNR is being recomputed "
            "from the currently loaded dataset rather than from the full baseline-corrected Excel batch."
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

    st.subheader("Nested CV estimate of the full selection procedure")
    left, right = st.columns([1, 2])
    left.dataframe(modeling_report.nested_summary_df, use_container_width=True)
    right.plotly_chart(
        model_comparison_figure(modeling_report.screening_df, modeling_report.nested_summary_df),
        use_container_width=True,
    )

    st.subheader("Selected pipeline frequency across outer folds")
    st.dataframe(
        modeling_report.selected_model_counts.rename_axis("pipeline").reset_index(name="count"),
        use_container_width=True,
    )
    st.success(f"Most frequently selected pipeline: {modeling_report.final_selected_model_name}")

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
    st.write(
        "Upload a patient spectrum as CSV. Supported formats: "
        "two numeric columns (`wavenumber`, `intensity`), one intensity column with the same length as the reference axis, "
        "or one row of intensities with the same length as the reference axis."
    )
    uploaded_file = st.file_uploader("Patient spectrum CSV", type=["csv"], key="patient_csv")

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

            st.caption(f"CSV parsing mode: {external.parser_mode}")
            qc_cols = st.columns(3)
            qc_cols[0].metric("corr vs Healthy mean", f"{quality['corr_healthy']:.3f}")
            qc_cols[1].metric("corr vs Disease mean", f"{quality['corr_disease']:.3f}")
            qc_cols[2].metric("roughness ratio", f"{quality['roughness_ratio']:.2f}")

            if quality["is_suspicious"]:
                st.error(
                    "The uploaded spectrum does not look compatible with the training data. "
                    "Prediction is likely unreliable. Check CSV format, axis order and preprocessing."
                )
            else:
                st.success("Spectrum quality check passed.")
            st.write(str(quality["reason"]))

            prediction = predict_patient(modeling_report, external_model_vector)

            pred_cols = st.columns(3)
            pred_cols[0].metric("Predicted class", prediction["predicted_label"])
            pred_cols[1].metric("P(Disease)", f"{prediction['probability_disease']:.3f}")
            pred_cols[2].metric("P(Healthy)", f"{prediction['probability_healthy']:.3f}")

            mean_healthy = model_dataset.X[model_dataset.y == 0].mean(axis=0)
            mean_disease = model_dataset.X[model_dataset.y == 1].mean(axis=0)
            patient_fig = spectrum_line_figure(
                model_dataset.wavenumber,
                [mean_healthy, mean_disease, external_model_vector],
                ["Healthy mean", "Disease mean", "Patient spectrum"],
                f"Patient spectrum vs class means ({modeling_low:.0f}-{modeling_high:.0f} cm^-1)",
                "SNV intensity",
            )
            st.plotly_chart(patient_fig, use_container_width=True)

            st.caption(
                f"Prediction uses deployment pipeline: {modeling_report.final_selected_model_name} "
                f"on modeling region {modeling_region_label}."
            )
        except Exception as exc:
            st.error(f"Could not process uploaded spectrum: {exc}")
