from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io

import numpy as np
import pandas as pd

from .preprocessing import als_baseline, snr_raman, snv


@dataclass
class RawDataset:
    wavenumber: np.ndarray
    x_health: np.ndarray
    x_disease: np.ndarray


@dataclass
class ProcessedDataset:
    X: np.ndarray
    y: np.ndarray
    wavenumber: np.ndarray
    snr: np.ndarray
    X_snr: np.ndarray | None = None

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
        )


@dataclass
class ExternalSpectrum:
    source_wavenumber: np.ndarray
    source_intensity: np.ndarray
    aligned_intensity: np.ndarray
    baseline_corrected: np.ndarray
    snv_processed: np.ndarray
    parser_mode: str


def load_raw_excel(path: str | Path) -> RawDataset:
    path = Path(path)
    df_health = pd.read_excel(path, sheet_name="health")
    df_disease = pd.read_excel(path, sheet_name="heart disease")

    wavenumber = df_health["wavenumber"].to_numpy(dtype=float)
    x_health = df_health.drop(columns=["wavenumber"]).T.to_numpy(dtype=float)
    x_disease = df_disease.drop(columns=["wavenumber"]).T.to_numpy(dtype=float)

    return RawDataset(
        wavenumber=wavenumber,
        x_health=x_health,
        x_disease=x_disease,
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
    return ProcessedDataset(X=X, y=y, wavenumber=wavenumber, snr=snr, X_snr=None)


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
