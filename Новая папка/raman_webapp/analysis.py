from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .data import ProcessedDataset


@dataclass
class Projection2D:
    embedding: np.ndarray
    explained_variance_ratio: np.ndarray | None = None


@dataclass
class InformativeRegion:
    low: float
    high: float
    mask: np.ndarray
    quality: np.ndarray
    threshold: float


@dataclass
class PeakDetectionResult:
    peak_indices: np.ndarray
    peaks_df: pd.DataFrame


@dataclass
class PeakBandMatchResult:
    matched_peaks_df: pd.DataFrame
    summary_df: pd.DataFrame


def mean_std_by_class(dataset: ProcessedDataset) -> dict[str, np.ndarray]:
    healthy = dataset.X[dataset.y == 0]
    disease = dataset.X[dataset.y == 1]
    return {
        "healthy_mean": healthy.mean(axis=0),
        "healthy_std": healthy.std(axis=0, ddof=1),
        "disease_mean": disease.mean(axis=0),
        "disease_std": disease.std(axis=0, ddof=1),
    }


def snr_statistics(dataset: ProcessedDataset) -> dict[str, float | pd.DataFrame]:
    return snr_statistics_from_values(dataset.y, dataset.snr)


def snr_statistics_from_values(y: np.ndarray, snr_values: np.ndarray) -> dict[str, float | pd.DataFrame]:
    snr_healthy = snr_values[y == 0]
    snr_disease = snr_values[y == 1]
    sh_healthy = stats.shapiro(snr_healthy)
    sh_disease = stats.shapiro(snr_disease)
    mann_whitney = stats.mannwhitneyu(snr_healthy, snr_disease, alternative="two-sided")

    summary = pd.DataFrame(
        [
            {
                "group": "Healthy",
                "mean": snr_healthy.mean(),
                "median": np.median(snr_healthy),
                "std": snr_healthy.std(ddof=1),
            },
            {
                "group": "Disease",
                "mean": snr_disease.mean(),
                "median": np.median(snr_disease),
                "std": snr_disease.std(ddof=1),
            },
        ]
    )

    return {
        "summary_df": summary,
        "shapiro_healthy_p": float(sh_healthy.pvalue),
        "shapiro_disease_p": float(sh_disease.pvalue),
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p": float(mann_whitney.pvalue),
        "snr_label_corr": float(np.corrcoef(snr_values, y)[0, 1]),
    }


def compute_pca_projection(dataset: ProcessedDataset) -> Projection2D:
    X_scaled = StandardScaler().fit_transform(dataset.X)
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_scaled)
    return Projection2D(embedding=embedding, explained_variance_ratio=pca.explained_variance_ratio_)


def compute_tsne_projection(dataset: ProcessedDataset, perplexity: float = 20.0) -> Projection2D:
    X_scaled = StandardScaler().fit_transform(dataset.X)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    return Projection2D(embedding=tsne.fit_transform(X_scaled))


def compute_umap_projection(dataset: ProcessedDataset) -> Projection2D | None:
    try:
        import umap  # type: ignore
    except ImportError:
        return None

    X_scaled = StandardScaler().fit_transform(dataset.X)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    return Projection2D(embedding=reducer.fit_transform(X_scaled))


def detect_informative_region(
    dataset: ProcessedDataset,
    smoothing_window: int = 61,
    min_fraction: float = 0.15,
    quality_quantile: float = 0.2,
) -> InformativeRegion:
    n_features = dataset.X.shape[1]

    if n_features < 5:
        mask = np.ones(n_features, dtype=bool)
        return InformativeRegion(
            low=float(dataset.wavenumber[0]),
            high=float(dataset.wavenumber[-1]),
            mask=mask,
            quality=np.ones(n_features, dtype=float),
            threshold=1.0,
        )

    summary = mean_std_by_class(dataset)
    healthy_mean = summary["healthy_mean"]
    disease_mean = summary["disease_mean"]
    avg_mean = 0.5 * (healthy_mean + disease_mean)
    class_diff = healthy_mean - disease_mean

    avg_trend = uniform_filter1d(avg_mean, size=smoothing_window, mode="reflect")
    diff_trend = uniform_filter1d(class_diff, size=smoothing_window, mode="reflect")

    avg_structure = np.sqrt(
        uniform_filter1d((avg_mean - avg_trend) ** 2, size=smoothing_window, mode="reflect")
    )
    diff_structure = np.sqrt(
        uniform_filter1d((class_diff - diff_trend) ** 2, size=smoothing_window, mode="reflect")
    )

    quality = avg_structure + 0.75 * diff_structure
    threshold = float(np.quantile(quality, quality_quantile))
    mask = quality >= threshold

    structure_close = np.ones(max(5, smoothing_window // 4), dtype=bool)
    structure_open = np.ones(max(5, smoothing_window // 6), dtype=bool)
    mask = binary_closing(mask, structure=structure_close)
    mask = binary_opening(mask, structure=structure_open)

    seed_idx = int(np.argmax(quality))
    expand_threshold = threshold
    best_start, best_end = _expand_from_seed(mask, quality, seed_idx, expand_threshold)

    min_points = max(10, int(min_fraction * n_features))
    if best_start is not None and best_end is not None and (best_end - best_start + 1) < min_points:
        center = (best_start + best_end) // 2
        half = min_points // 2
        best_start = max(0, center - half)
        best_end = min(n_features - 1, best_start + min_points - 1)
        best_start = max(0, best_end - min_points + 1)

    if best_start is None or best_end is None:
        mask = np.ones(n_features, dtype=bool)
        best_start, best_end = 0, n_features - 1
    else:
        final_mask = np.zeros(n_features, dtype=bool)
        final_mask[best_start : best_end + 1] = True
        mask = final_mask

    return InformativeRegion(
        low=float(dataset.wavenumber[best_start]),
        high=float(dataset.wavenumber[best_end]),
        mask=mask,
        quality=quality,
        threshold=threshold,
    )


def detect_spectral_peaks(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    prominence: float = 0.08,
    min_distance_cm: float = 12.0,
    min_width_cm: float = 6.0,
    top_k: int = 20,
) -> PeakDetectionResult:
    wavenumber = np.asarray(wavenumber, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if wavenumber.ndim != 1 or intensity.ndim != 1 or wavenumber.size != intensity.size:
        raise ValueError("Peak detection expects one-dimensional wavenumber and intensity arrays of equal length.")

    if wavenumber.size < 5:
        return PeakDetectionResult(
            peak_indices=np.array([], dtype=int),
            peaks_df=pd.DataFrame(
                columns=["wavenumber_cm-1", "intensity", "prominence", "width_cm-1", "rank_by_prominence"]
            ),
        )

    wn_step = float(np.median(np.abs(np.diff(wavenumber))))
    min_distance_pts = max(1, int(round(min_distance_cm / max(wn_step, 1e-12))))
    min_width_pts = max(1, int(round(min_width_cm / max(wn_step, 1e-12))))

    peak_indices, properties = find_peaks(
        intensity,
        prominence=prominence,
        distance=min_distance_pts,
        width=min_width_pts,
    )

    if peak_indices.size == 0:
        return PeakDetectionResult(
            peak_indices=peak_indices,
            peaks_df=pd.DataFrame(
                columns=["wavenumber_cm-1", "intensity", "prominence", "width_cm-1", "rank_by_prominence"]
            ),
        )

    widths_pts = peak_widths(intensity, peak_indices, rel_height=0.5)[0]
    widths_cm = widths_pts * wn_step

    peaks_df = pd.DataFrame(
        {
            "wavenumber_cm-1": wavenumber[peak_indices],
            "intensity": intensity[peak_indices],
            "prominence": properties["prominences"],
            "width_cm-1": widths_cm,
        }
    ).sort_values("prominence", ascending=False).reset_index(drop=True)

    if top_k > 0:
        peaks_df = peaks_df.head(top_k).copy()

    peaks_df["rank_by_prominence"] = np.arange(1, len(peaks_df) + 1)
    selected_peak_indices = np.array(
        [
            int(np.argmin(np.abs(wavenumber - peak_wn)))
            for peak_wn in peaks_df["wavenumber_cm-1"].to_numpy(dtype=float)
        ],
        dtype=int,
    )
    return PeakDetectionResult(
        peak_indices=selected_peak_indices,
        peaks_df=peaks_df[
            ["rank_by_prominence", "wavenumber_cm-1", "intensity", "prominence", "width_cm-1"]
        ],
    )


def match_peaks_to_reference_bands(
    peaks_df: pd.DataFrame,
    reference_bands_df: pd.DataFrame,
    matching_mode: str = "close",
    tolerance_cm: float = 5.0,
) -> PeakBandMatchResult:
    if peaks_df.empty or reference_bands_df.empty:
        empty_matches = pd.DataFrame(
            columns=[
                "rank_by_prominence",
                "wavenumber_cm-1",
                "match_type",
                "priority",
                "label",
                "assignment",
                "clinical_hint",
                "notes",
                "distance_to_band_center_cm-1",
            ]
        )
        empty_summary = pd.DataFrame(columns=["priority", "n_matches", "labels"])
        return PeakBandMatchResult(matched_peaks_df=empty_matches, summary_df=empty_summary)

    match_rows: list[dict[str, float | str | int]] = []
    for peak_row in peaks_df.to_dict(orient="records"):
        peak_wn = float(peak_row["wavenumber_cm-1"])
        if matching_mode == "exact":
            overlaps = reference_bands_df[
                (reference_bands_df["low_cm1"] <= peak_wn) & (reference_bands_df["high_cm1"] >= peak_wn)
            ].copy()
            overlaps["match_type"] = "exact"
        else:
            exact_overlaps = reference_bands_df[
                (reference_bands_df["low_cm1"] <= peak_wn) & (reference_bands_df["high_cm1"] >= peak_wn)
            ].copy()
            exact_overlaps["match_type"] = "exact"

            close_overlaps = reference_bands_df[
                (reference_bands_df["low_cm1"] - tolerance_cm <= peak_wn)
                & (reference_bands_df["high_cm1"] + tolerance_cm >= peak_wn)
            ].copy()
            close_overlaps["match_type"] = "close"

            overlaps = pd.concat([exact_overlaps, close_overlaps], ignore_index=True)
            if not overlaps.empty:
                overlaps = overlaps.sort_values("match_type").drop_duplicates(subset=["band_id"], keep="first")
        for _, band in overlaps.iterrows():
            band_center = 0.5 * (float(band["low_cm1"]) + float(band["high_cm1"]))
            match_rows.append(
                {
                    "rank_by_prominence": int(peak_row["rank_by_prominence"]),
                    "wavenumber_cm-1": peak_wn,
                    "intensity": float(peak_row["intensity"]),
                    "prominence": float(peak_row["prominence"]),
                    "width_cm-1": float(peak_row["width_cm-1"]),
                    "match_type": str(band["match_type"]),
                    "priority": str(band["priority"]),
                    "label": str(band["label"]),
                    "assignment": str(band["assignment"]),
                    "clinical_hint": str(band["clinical_hint"]),
                    "notes": str(band["notes"]),
                    "distance_to_band_center_cm-1": abs(peak_wn - band_center),
                }
            )

    if not match_rows:
        empty_summary = pd.DataFrame(columns=["priority", "n_matches", "labels"])
        return PeakBandMatchResult(
            matched_peaks_df=pd.DataFrame(
                columns=[
                    "rank_by_prominence",
                    "wavenumber_cm-1",
                    "match_type",
                    "priority",
                    "label",
                    "assignment",
                    "clinical_hint",
                    "notes",
                    "distance_to_band_center_cm-1",
                ]
            ),
            summary_df=empty_summary,
        )

    priority_order = {"high": 0, "medium": 1, "support": 2}
    matched_df = pd.DataFrame(match_rows)
    matched_df["priority_rank"] = matched_df["priority"].map(priority_order).fillna(99).astype(int)
    matched_df["match_rank"] = matched_df["match_type"].map({"exact": 0, "close": 1}).fillna(99).astype(int)
    matched_df = matched_df.sort_values(
        ["priority_rank", "match_rank", "rank_by_prominence", "distance_to_band_center_cm-1"]
    ).reset_index(drop=True)

    summary_df = (
        matched_df.groupby(["priority", "match_type"], sort=False)
        .agg(
            n_matches=("label", "count"),
            labels=("label", lambda x: ", ".join(dict.fromkeys(x))),
        )
        .reset_index()
    )
    summary_df["priority_rank"] = summary_df["priority"].map(priority_order).fillna(99).astype(int)
    summary_df["match_rank"] = summary_df["match_type"].map({"exact": 0, "close": 1}).fillna(99).astype(int)
    summary_df = summary_df.sort_values(["priority_rank", "match_rank"]).drop(
        columns=["priority_rank", "match_rank"]
    ).reset_index(drop=True)

    return PeakBandMatchResult(
        matched_peaks_df=matched_df[
            [
                "rank_by_prominence",
                "wavenumber_cm-1",
                "match_type",
                "priority",
                "label",
                "assignment",
                "clinical_hint",
                "notes",
                "distance_to_band_center_cm-1",
            ]
        ],
        summary_df=summary_df,
    )

def _expand_from_seed(
    mask: np.ndarray, quality: np.ndarray, seed_idx: int, expand_threshold: float
) -> tuple[int | None, int | None]:
    if not mask[seed_idx]:
        return None, None

    left = seed_idx
    right = seed_idx

    while left > 0 and mask[left - 1] and quality[left - 1] >= expand_threshold:
        left -= 1
    while right < len(mask) - 1 and mask[right + 1] and quality[right + 1] >= expand_threshold:
        right += 1

    return left, right
