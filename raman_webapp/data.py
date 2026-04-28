from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import re
from typing import BinaryIO

import numpy as np
import pandas as pd

from .preprocessing import als_baseline, snr_raman, snv


@dataclass
class RawDataset:
    wavenumber: np.ndarray
    x_health: np.ndarray
    x_disease: np.ndarray
    health_names: np.ndarray
    disease_names: np.ndarray


@dataclass
class ProcessedDataset:
    X: np.ndarray
    y: np.ndarray
    wavenumber: np.ndarray
    snr: np.ndarray
    X_snr: np.ndarray | None = None
    sample_names: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray:
        return np.where(self.y == 0, "Healthy", "Disease")

    def select_wavenumber_range(self, low: float, high: float) -> "ProcessedDataset":
        mask = (self.wavenumber >= low) & (self.wavenumber <= high)
        return ProcessedDataset(
            X=self.X[:, mask],
            y=self.y.copy(),
            wavenumber=self.wavenumber[mask].copy(),
            snr=self.snr.copy(),
            X_snr=self.X_snr[:, mask].copy() if self.X_snr is not None else None,
            sample_names=self.sample_names.copy() if self.sample_names is not None else None,
        )

    def subset_by_mask(self, mask: np.ndarray) -> "ProcessedDataset":
        mask = np.asarray(mask, dtype=bool)
        return ProcessedDataset(
            X=self.X[mask].copy(),
            y=self.y[mask].copy(),
            wavenumber=self.wavenumber.copy(),
            snr=self.snr[mask].copy(),
            X_snr=self.X_snr[mask].copy() if self.X_snr is not None else None,
            sample_names=self.sample_names[mask].copy() if self.sample_names is not None else None,
        )


@dataclass
class ExternalSpectrum:
    source_wavenumber: np.ndarray
    source_intensity: np.ndarray
    aligned_intensity: np.ndarray
    baseline_corrected: np.ndarray
    snv_processed: np.ndarray
    parser_mode: str


HOLDOUT_SAMPLE_NAMES = {
    "healthy1",
    "healthy2",
    "healthy3",
    "healthy4",
    "healthy5",
    "heart_patient1",
    "heart_patient2",
    "heart_patient3",
    "heart_patient4",
    "heart_patient5",
}


def load_raw_excel(path: str | Path | BinaryIO | io.BytesIO) -> RawDataset:
    excel_source = Path(path) if isinstance(path, (str, Path)) else path
    df_health = pd.read_excel(excel_source, sheet_name="health")
    df_disease = pd.read_excel(excel_source, sheet_name="heart disease")

    wavenumber = df_health["wavenumber"].to_numpy(dtype=float)
    x_health = df_health.drop(columns=["wavenumber"]).T.to_numpy(dtype=float)
    x_disease = df_disease.drop(columns=["wavenumber"]).T.to_numpy(dtype=float)
    health_names = df_health.drop(columns=["wavenumber"]).columns.to_numpy(dtype=str)
    disease_names = df_disease.drop(columns=["wavenumber"]).columns.to_numpy(dtype=str)

    return RawDataset(
        wavenumber=wavenumber,
        x_health=x_health,
        x_disease=x_disease,
        health_names=health_names,
        disease_names=disease_names,
    )


def preprocess_raw_dataset(
    raw: RawDataset,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10,
) -> ProcessedDataset:
    X = np.vstack([raw.x_health, raw.x_disease])
    y = np.concatenate(
        [
            np.zeros(raw.x_health.shape[0], dtype=int),
            np.ones(raw.x_disease.shape[0], dtype=int),
        ]
    )
    sample_names = np.concatenate([raw.health_names, raw.disease_names])

    X_bc = np.empty_like(X, dtype=float)
    for idx in range(X.shape[0]):
        X_bc[idx] = X[idx] - als_baseline(X[idx], lam=lam, p=p, niter=niter)

    X_snv = snv(X_bc)
    snr = np.array([snr_raman(row, raw.wavenumber) for row in X_bc], dtype=float)

    return ProcessedDataset(
        X=X_snv,
        y=y,
        wavenumber=raw.wavenumber.copy(),
        snr=snr,
        X_snr=X_bc,
        sample_names=sample_names.copy(),
    )


def load_processed_csvs(
    x_path: str | Path,
    y_path: str | Path,
    wavenumber_path: str | Path,
    snr_path: str | Path,
) -> ProcessedDataset:
    X = _read_numeric_csv_matrix(x_path)
    y = _read_numeric_csv_vector(y_path, dtype=int)
    wavenumber = _read_numeric_csv_vector(wavenumber_path, dtype=float)
    snr = _read_numeric_csv_vector(snr_path, dtype=float)
    sample_names = _default_sample_names_from_labels(y)
    return ProcessedDataset(X=X, y=y, wavenumber=wavenumber, snr=snr, X_snr=None, sample_names=sample_names)


def _read_numeric_csv_matrix(path: str | Path) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.to_numpy(dtype=float)


def _read_numeric_csv_vector(path: str | Path, dtype: type[int] | type[float]) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        df = df.iloc[:, :1]
    series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return series.to_numpy(dtype=dtype)


def _default_sample_names_from_labels(y: np.ndarray) -> np.ndarray:
    names: list[str] = []
    healthy_idx = 0
    disease_idx = 0
    for label in y:
        if int(label) == 0:
            healthy_idx += 1
            names.append(f"healthy{healthy_idx}")
        else:
            disease_idx += 1
            names.append(f"heart_patient{disease_idx}")
    return np.array(names, dtype=str)


def split_train_holdout(dataset: ProcessedDataset) -> tuple[ProcessedDataset, ProcessedDataset]:
    if dataset.sample_names is None:
        raise ValueError("Dataset sample names are required for holdout splitting.")

    normalized_holdout_names = {_normalize_sample_name(name) for name in HOLDOUT_SAMPLE_NAMES}
    normalized_sample_names = np.array([_normalize_sample_name(name) for name in dataset.sample_names], dtype=str)

    holdout_mask = np.isin(normalized_sample_names, list(normalized_holdout_names))
    matched_holdout_names = set(normalized_sample_names[holdout_mask].tolist())
    if len(matched_holdout_names) != len(normalized_holdout_names):
        missing = sorted(normalized_holdout_names.difference(matched_holdout_names))
        raise ValueError(f"Could not locate all holdout samples in dataset: {missing}")

    train_dataset = dataset.subset_by_mask(~holdout_mask)
    holdout_dataset = dataset.subset_by_mask(holdout_mask)
    return train_dataset, holdout_dataset


def load_reference_band_library(
    path: str | Path | None = None,
    spectroscopy_mode: str = "raman",
) -> pd.DataFrame:
    if path is None:
        file_name = "sers_reference_peaks.json" if spectroscopy_mode == "sers" else "reference_bands.json"
        path = Path(__file__).resolve().parent / file_name
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    band_df = pd.DataFrame(records)
    if band_df.empty:
        return band_df

    if spectroscopy_mode == "sers":
        role_order = {"candidate": 0, "background": 1}
        band_df["role_rank"] = band_df["peak_role"].map(role_order).fillna(99).astype(int)
        if "shift_cm" not in band_df.columns:
            band_df["shift_cm"] = np.nan
        if "shift_min_cm" not in band_df.columns:
            band_df["shift_min_cm"] = np.nan
        if "shift_max_cm" not in band_df.columns:
            band_df["shift_max_cm"] = np.nan
        band_df["sort_position"] = band_df["shift_cm"].fillna(band_df["shift_min_cm"]).fillna(1e9)
        band_df = band_df.sort_values(["role_rank", "sort_position", "shift_max_cm"]).reset_index(drop=True)
        return band_df.drop(columns=["role_rank", "sort_position"], errors="ignore")

    priority_order = {"high": 0, "medium": 1, "support": 2}
    band_df["priority_rank"] = band_df["priority"].map(priority_order).fillna(99).astype(int)
    band_df = band_df.sort_values(["priority_rank", "low_cm1", "high_cm1"]).reset_index(drop=True)
    return band_df


def load_external_spectrum_csv(file_obj, reference_wavenumber: np.ndarray) -> ExternalSpectrum:
    file_bytes = file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    text = file_bytes.decode("utf-8-sig", errors="replace")

    delimiter, decimal = _guess_csv_format(text)
    df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    if numeric_df.shape[1] >= 2 and numeric_df.iloc[:, :2].notna().all().all():
        source_wavenumber = numeric_df.iloc[:, 0].to_numpy(dtype=float)
        source_intensity = numeric_df.iloc[:, 1].to_numpy(dtype=float)
        parser_mode = "two-column"
    elif numeric_df.shape[1] == 1:
        source_intensity = numeric_df.iloc[:, 0].dropna().to_numpy(dtype=float)
        if source_intensity.size != reference_wavenumber.size:
            raise ValueError(
                "Single-column spectrum CSV must have the same number of points as the reference spectrum."
            )
        source_wavenumber = np.asarray(reference_wavenumber, dtype=float)
        parser_mode = "single-column"
    elif numeric_df.shape[0] == 1:
        source_intensity = numeric_df.iloc[0].dropna().to_numpy(dtype=float)
        if source_intensity.size != reference_wavenumber.size:
            raise ValueError(
                "Single-row spectrum CSV must have the same number of points as the reference spectrum."
            )
        source_wavenumber = np.asarray(reference_wavenumber, dtype=float)
        parser_mode = "single-row"
    else:
        raise ValueError(
            "Could not interpret the CSV. Expected either two numeric columns (wavenumber, intensity) or one intensity vector."
        )

    order = np.argsort(source_wavenumber)
    source_wavenumber = source_wavenumber[order]
    source_intensity = source_intensity[order]

    unique_wavenumber, unique_idx = np.unique(source_wavenumber, return_index=True)
    unique_intensity = source_intensity[unique_idx]

    aligned_intensity = np.interp(reference_wavenumber, unique_wavenumber, unique_intensity)
    baseline_corrected = aligned_intensity - als_baseline(aligned_intensity)
    snv_processed = snv(baseline_corrected[None, :])[0]

    return ExternalSpectrum(
        source_wavenumber=unique_wavenumber,
        source_intensity=unique_intensity,
        aligned_intensity=aligned_intensity,
        baseline_corrected=baseline_corrected,
        snv_processed=snv_processed,
        parser_mode=parser_mode,
    )


def _guess_csv_format(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sample = lines[:5]
    semicolon_count = sum(line.count(";") for line in sample)
    comma_count = sum(line.count(",") for line in sample)
    tab_count = sum(line.count("\t") for line in sample)

    if semicolon_count >= max(comma_count, tab_count):
        return ";", ","
    if tab_count > max(semicolon_count, comma_count):
        return "\t", "."
    return ",", "."


def _normalize_sample_name(name: str) -> str:
    normalized = str(name).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized
