from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import ProcessedDataset


SCORING = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}


@dataclass
class ModelingReport:
    screening_df: pd.DataFrame
    nested_fold_df: pd.DataFrame
    nested_summary_df: pd.DataFrame
    selected_model_counts: pd.Series
    final_selected_model_name: str
    deployment_model: Pipeline
    interpretation_model: Pipeline
    coef_original_space: np.ndarray
    coef_smoothed_space: np.ndarray
    top_features_df: pd.DataFrame
    top_bands_df: pd.DataFrame


@dataclass
class HoldoutReport:
    summary_df: pd.DataFrame
    predictions_df: pd.DataFrame


def build_pipelines() -> dict[str, Pipeline]:
    models = {
        "LogReg": LogisticRegression(max_iter=5000, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "kNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    pipelines: dict[str, Pipeline] = {}
    for model_name, clf in models.items():
        pipelines[f"{model_name} on X (no scaler)"] = Pipeline([("clf", clf)])

        if model_name not in {"RandomForest", "GradientBoosting"}:
            pipelines[f"{model_name} on X (scaled)"] = Pipeline(
                [("scaler", StandardScaler()), ("clf", clf)]
            )

        if model_name in {"LogReg", "SVM (RBF)", "kNN (k=7)"}:
            pipelines[f"{model_name} + Scaler + PCA(0.95)"] = Pipeline(
                [("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42)), ("clf", clf)]
            )
            pipelines[f"{model_name} + Scaler + PCA(10)"] = Pipeline(
                [("scaler", StandardScaler()), ("pca", PCA(n_components=10, random_state=42)), ("clf", clf)]
            )

    return pipelines


def _get_score_values(fitted_pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(fitted_pipe, "predict_proba"):
        return fitted_pipe.predict_proba(X)[:, 1]
    return fitted_pipe.decision_function(X)


def run_modeling(dataset: ProcessedDataset) -> ModelingReport:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    screening_rows: list[dict[str, float | str]] = []
    for name, pipe in build_pipelines().items():
        scores = cross_validate(pipe, dataset.X, dataset.y, cv=outer_cv, scoring=SCORING)
        screening_rows.append(
            {
                "model": name,
                "roc_auc_mean": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
                "accuracy_mean": scores["test_accuracy"].mean(),
                "f1_mean": scores["test_f1"].mean(),
                "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
            }
        )
    screening_df = pd.DataFrame(screening_rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)

    nested_rows: list[dict[str, float | str | int]] = []
    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(dataset.X, dataset.y), start=1):
        X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
        y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]

        inner_summary: dict[str, float] = {}
        for name, pipe in build_pipelines().items():
            inner_scores = cross_validate(pipe, X_train, y_train, cv=inner_cv, scoring=SCORING)
            inner_summary[name] = inner_scores["test_roc_auc"].mean()

        best_name = max(inner_summary, key=inner_summary.get)
        best_pipe = clone(build_pipelines()[best_name])
        best_pipe.fit(X_train, y_train)

        y_score = _get_score_values(best_pipe, X_test)
        y_pred = best_pipe.predict(X_test)

        nested_rows.append(
            {
                "fold": fold_id,
                "selected_model": best_name,
                "roc_auc": roc_auc_score(y_test, y_score),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            }
        )

    nested_fold_df = pd.DataFrame(nested_rows)
    selected_model_counts = nested_fold_df["selected_model"].value_counts()
    final_selected_model_name = str(selected_model_counts.idxmax())

    nested_summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": nested_fold_df[metric].mean(),
                "std": nested_fold_df[metric].std(ddof=1),
            }
            for metric in ["roc_auc", "accuracy", "f1", "balanced_accuracy"]
        ]
    )

    deployment_model = clone(build_pipelines()[final_selected_model_name])
    deployment_model.fit(dataset.X, dataset.y)

    interpretation_model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]
    )
    interpretation_model.fit(dataset.X, dataset.y)
    scaler = interpretation_model.named_steps["scaler"]
    clf = interpretation_model.named_steps["clf"]
    coef_original = clf.coef_.ravel() / scaler.scale_
    coef_smoothed = _smooth_coefficients(coef_original)

    top_idx = _select_spaced_top_indices(dataset.wavenumber, coef_original, top_k=15, min_spacing_cm=8.0)
    top_features_df = pd.DataFrame(
        {
            "wavenumber_cm-1": dataset.wavenumber[top_idx],
            "coefficient": coef_original[top_idx],
            "abs_coefficient": np.abs(coef_original[top_idx]),
        }
    )
    top_bands_df = _build_coefficient_bands(dataset.wavenumber, coef_smoothed)

    return ModelingReport(
        screening_df=screening_df,
        nested_fold_df=nested_fold_df,
        nested_summary_df=nested_summary_df,
        selected_model_counts=selected_model_counts,
        final_selected_model_name=final_selected_model_name,
        deployment_model=deployment_model,
        interpretation_model=interpretation_model,
        coef_original_space=coef_original,
        coef_smoothed_space=coef_smoothed,
        top_features_df=top_features_df,
        top_bands_df=top_bands_df,
    )


def generate_model_interpretation(report: ModelingReport) -> str:
    best_screening = report.screening_df.iloc[0]
    nested_roc = report.nested_summary_df.loc[report.nested_summary_df["metric"] == "roc_auc", "mean"].iloc[0]
    return (
        f"Exploratory screening ranks '{best_screening['model']}' first by ROC-AUC "
        f"({best_screening['roc_auc_mean']:.3f}), while nested CV estimates the full selection procedure "
        f"at ROC-AUC {nested_roc:.3f}. The most frequently selected pipeline across outer folds is "
        f"'{report.final_selected_model_name}'. Coefficient-based interpretation is shown with logistic regression "
        "in the original feature space and should be treated as explanatory, not as an unbiased test estimate. "
        "Because neighboring wavelengths are strongly collinear, the app summarizes interpretation by spectral bands, "
        "not only by individual coefficient spikes."
    )


def predict_patient(report: ModelingReport, X: np.ndarray) -> dict[str, float | int | str]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    pred = int(report.deployment_model.predict(X)[0])
    if hasattr(report.deployment_model, "predict_proba"):
        prob_disease = float(report.deployment_model.predict_proba(X)[0, 1])
    else:
        score = float(report.deployment_model.decision_function(X)[0])
        prob_disease = float(1.0 / (1.0 + np.exp(-score)))
    return {
        "predicted_label": "Disease" if pred == 1 else "Healthy",
        "predicted_class": pred,
        "probability_disease": prob_disease,
        "probability_healthy": 1.0 - prob_disease,
    }


def evaluate_holdout(report: ModelingReport, dataset: ProcessedDataset) -> HoldoutReport:
    y_score = _get_score_values(report.deployment_model, dataset.X)
    y_pred = report.deployment_model.predict(dataset.X).astype(int)

    metrics_rows: list[dict[str, float | str]] = [
        {"metric": "accuracy", "value": accuracy_score(dataset.y, y_pred)},
        {"metric": "balanced_accuracy", "value": balanced_accuracy_score(dataset.y, y_pred)},
        {"metric": "f1", "value": f1_score(dataset.y, y_pred)},
    ]
    if len(np.unique(dataset.y)) == 2:
        metrics_rows.append({"metric": "roc_auc", "value": roc_auc_score(dataset.y, y_score)})

    sample_names = (
        dataset.sample_names
        if dataset.sample_names is not None
        else np.array([f"sample_{idx + 1}" for idx in range(dataset.X.shape[0])], dtype=str)
    )
    predictions_df = pd.DataFrame(
        {
            "sample_name": sample_names,
            "true_label": np.where(dataset.y == 0, "Healthy", "Disease"),
            "predicted_label": np.where(y_pred == 0, "Healthy", "Disease"),
            "probability_disease": y_score,
            "correct": y_pred == dataset.y,
        }
    )

    return HoldoutReport(
        summary_df=pd.DataFrame(metrics_rows),
        predictions_df=predictions_df,
    )


def _smooth_coefficients(coef: np.ndarray) -> np.ndarray:
    n = len(coef)
    if n < 7:
        return coef.copy()
    window = min(31, n if n % 2 == 1 else n - 1)
    if window < 5:
        return coef.copy()
    return savgol_filter(coef, window_length=window, polyorder=3)


def _build_coefficient_bands(wavenumber: np.ndarray, coef_smoothed: np.ndarray) -> pd.DataFrame:
    strength = np.abs(coef_smoothed)
    threshold = float(np.quantile(strength, 0.9))
    mask = strength >= threshold

    bands: list[dict[str, float]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            end = idx - 1
            bands.append(_summarize_band(wavenumber, coef_smoothed, start, end))
            start = None
    if start is not None:
        bands.append(_summarize_band(wavenumber, coef_smoothed, start, len(mask) - 1))

    if not bands:
        peak_idx = int(np.argmax(strength))
        bands.append(_summarize_band(wavenumber, coef_smoothed, peak_idx, peak_idx))

    band_df = pd.DataFrame(bands).sort_values("band_score", ascending=False).reset_index(drop=True)
    return band_df.head(10)


def _select_spaced_top_indices(
    wavenumber: np.ndarray,
    coef: np.ndarray,
    top_k: int,
    min_spacing_cm: float,
) -> np.ndarray:
    ranked_idx = np.argsort(np.abs(coef))[::-1]
    selected: list[int] = []
    for idx in ranked_idx:
        wn = float(wavenumber[idx])
        if all(abs(wn - float(wavenumber[chosen])) >= min_spacing_cm for chosen in selected):
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    return np.array(selected, dtype=int)


def _summarize_band(wavenumber: np.ndarray, coef_smoothed: np.ndarray, start: int, end: int) -> dict[str, float]:
    band_slice = slice(start, end + 1)
    band_coef = coef_smoothed[band_slice]
    peak_local_idx = int(np.argmax(np.abs(band_coef)))
    peak_idx = start + peak_local_idx
    return {
        "start_cm-1": float(wavenumber[start]),
        "end_cm-1": float(wavenumber[end]),
        "peak_cm-1": float(wavenumber[peak_idx]),
        "peak_coefficient": float(coef_smoothed[peak_idx]),
        "band_score": float(np.sum(np.abs(band_coef))),
        "n_points": int(end - start + 1),
    }
