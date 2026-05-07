from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(31,119,180,{alpha})"
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def spectrum_line_figure(
    wavenumber: np.ndarray,
    y_values: list[np.ndarray],
    labels: list[str],
    title: str,
    y_axis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for values, label in zip(y_values, labels):
        fig.add_trace(go.Scatter(x=wavenumber, y=values, mode="lines", name=label))
    fig.update_layout(title=title, xaxis_title="Wavenumber, cm^-1", yaxis_title=y_axis_title)
    return fig


def peak_detection_figure(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    peak_indices: np.ndarray,
    title: str,
    y_axis_title: str,
    peak_text: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=intensity, mode="lines", name="Spectrum"))

    if len(peak_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=wavenumber[peak_indices],
                y=intensity[peak_indices],
                mode="markers+text" if peak_text else "markers",
                name="Detected peaks",
                marker=dict(size=9, color="#d62728", symbol="diamond"),
                text=peak_text,
                textposition="top center",
            )
        )

    fig.update_layout(title=title, xaxis_title="Wavenumber, cm^-1", yaxis_title=y_axis_title)
    return fig


def spectrum_with_band_figure(
    wavenumber: np.ndarray,
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    y_axis_title: str = "Intensity",
) -> go.Figure:
    fig = go.Figure()
    for mean, sem, label, color in [
        (mean_a, sem_a, label_a, "#1f77b4"),
        (mean_b, sem_b, label_b, "#d62728"),
    ]:
        fig.add_trace(go.Scatter(x=wavenumber, y=mean, mode="lines", name=label, line=dict(color=color)))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([wavenumber, wavenumber[::-1]]),
                y=np.concatenate([mean - sem, (mean + sem)[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.18),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.update_layout(title=title, xaxis_title="Wavenumber, cm^-1", yaxis_title=y_axis_title)
    return fig


def scatter_projection_figure(embedding: np.ndarray, labels: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    for class_name, color in [("Healthy", "#1f77b4"), ("Disease", "#d62728")]:
        mask = labels == class_name
        fig.add_trace(
            go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode="markers",
                name=class_name,
                marker=dict(size=9, color=color, opacity=0.75),
            )
        )
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    return fig


def boxplot_snr_figure(snr: np.ndarray, y: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=snr[y == 0], name="Healthy", boxpoints="outliers"))
    fig.add_trace(go.Box(y=snr[y == 1], name="Disease", boxpoints="outliers"))
    fig.update_layout(title="SNR distribution by class", yaxis_title="SNR")
    return fig


def histogram_snr_figure(snr: np.ndarray, y: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=snr[y == 0], name="Healthy", opacity=0.65, nbinsx=15))
    fig.add_trace(go.Histogram(x=snr[y == 1], name="Disease", opacity=0.65, nbinsx=15))
    fig.update_layout(title="SNR distributions by class", xaxis_title="SNR", yaxis_title="Count", barmode="overlay")
    return fig


def model_comparison_figure(screening_df: pd.DataFrame, nested_summary_df: pd.DataFrame) -> go.Figure:
    plot_df = screening_df.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    nested_roc = nested_summary_df.loc[nested_summary_df["metric"] == "roc_auc", "mean"].iloc[0]
    nested_std = nested_summary_df.loc[nested_summary_df["metric"] == "roc_auc", "std"].iloc[0]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["model"],
            y=plot_df["roc_auc_mean"],
            error_y=dict(type="data", array=plot_df["roc_auc_std"]),
            name="Screening ROC-AUC",
        )
    )
    fig.add_hline(
        y=nested_roc,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Nested CV selected procedure: {nested_roc:.3f} +/- {nested_std:.3f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Candidate screening vs nested-CV estimate",
        xaxis_title="Pipeline",
        yaxis_title="ROC-AUC",
    )
    return fig


def roc_curve_figure(roc_df: pd.DataFrame, title: str, auc_value: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_df["fpr"],
            y=roc_df["tpr"],
            mode="lines+markers",
            name="ROC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Random",
            line=dict(dash="dash", color="gray"),
        )
    )
    final_title = title if auc_value is None else f"{title} (AUC={auc_value:.3f})"
    fig.update_layout(
        title=final_title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig


def coefficient_figure(wavenumber: np.ndarray, coef: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=coef, mode="lines", name="Coefficient"))
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(title=title, xaxis_title="Wavenumber, cm^-1", yaxis_title="Coefficient")
    return fig


def informative_region_figure(
    wavenumber: np.ndarray,
    quality: np.ndarray,
    threshold: float,
    low: float,
    high: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=quality, mode="lines", name="Quality profile"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="crimson", annotation_text="selection threshold")
    fig.add_vrect(x0=low, x1=high, fillcolor="rgba(31,119,180,0.12)", line_width=0)
    fig.update_layout(
        title="Automatically detected informative spectral region",
        xaxis_title="Wavenumber, cm^-1",
        yaxis_title="Signal quality score",
    )
    return fig
