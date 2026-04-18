from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
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
