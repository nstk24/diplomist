from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_DATASET_PATH = ROOT / "Raman_krov_SSZ-zdorovye.xlsx"
NEW_DATASET_PATH = ROOT / "serum_spectra_project_format.xlsx"
OUTPUT_DATASET_PATH = ROOT / "combined_raman_serum_50_50.xlsx"
EXPORT_DIR = ROOT / "patient_csv_exports"
EXTERNAL_EXPORT_DIR = ROOT / "patient_csv_external"

SHEET_TO_CLASS = {
    "health": "healthy",
    "heart disease": "disease",
}

OLD_SAMPLE_COUNT_PER_CLASS = 50
NEW_TRAIN_SAMPLE_COUNT_PER_CLASS = 70
NEW_HOLDOUT_SAMPLE_COUNT_PER_CLASS = 5
RANDOM_SEED = 42
NEW_SAMPLE_PREFIX = "serum"


@dataclass
class SheetData:
    wavenumber: np.ndarray
    intensity_df: pd.DataFrame


def load_sheet(path: Path, sheet_name: str) -> SheetData:
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "wavenumber" not in df.columns:
        raise ValueError(f"Sheet '{sheet_name}' in {path.name} does not contain 'wavenumber'.")
    wavenumber = df["wavenumber"].to_numpy(dtype=float)
    intensity_df = df.drop(columns=["wavenumber"]).copy()
    return SheetData(wavenumber=wavenumber, intensity_df=intensity_df)


def normalize_new_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    meta = metadata_df.copy()
    class_series = meta["class_label"].astype(str).str.lower()
    meta["normalized_class"] = np.where(class_series.str.contains("healthy", na=False), "healthy", "disease")
    meta["project_sample_name"] = meta["project_sample_name"].astype(str)
    return meta


def interpolate_to_grid(
    source_wavenumber: np.ndarray,
    source_df: pd.DataFrame,
    target_wavenumber: np.ndarray,
) -> pd.DataFrame:
    interpolated: dict[str, np.ndarray] = {}
    for column_name in source_df.columns:
        interpolated[column_name] = np.interp(
            target_wavenumber,
            source_wavenumber,
            source_df[column_name].to_numpy(dtype=float),
        )
    return pd.DataFrame(interpolated)


def choose_balanced_samples(
    metadata_df: pd.DataFrame,
    target_count: int,
    rng: np.random.Generator,
    exclude_names: set[str] | None = None,
) -> list[str]:
    exclude_names = exclude_names or set()
    meta = metadata_df.loc[~metadata_df["project_sample_name"].isin(exclude_names)].copy()
    if meta.empty:
        raise ValueError("No eligible samples remain after exclusions.")

    selected: list[str] = []
    remaining = meta.copy()
    patient_ids = remaining["patient_id"].dropna().astype(str).unique().tolist()
    rng.shuffle(patient_ids)

    while len(selected) < target_count and not remaining.empty:
        progress_made = False
        for patient_id in patient_ids:
            patient_rows = remaining.loc[remaining["patient_id"].astype(str) == patient_id]
            if patient_rows.empty:
                continue
            row_index = int(rng.choice(patient_rows.index.to_numpy()))
            sample_name = str(remaining.loc[row_index, "project_sample_name"])
            selected.append(sample_name)
            remaining = remaining.drop(index=row_index)
            progress_made = True
            if len(selected) >= target_count:
                break
        if not progress_made:
            break

    if len(selected) < target_count:
        leftover = remaining["project_sample_name"].astype(str).tolist()
        rng.shuffle(leftover)
        selected.extend(leftover[: target_count - len(selected)])

    if len(selected) < target_count:
        raise ValueError(f"Requested {target_count} samples, but only {len(selected)} were available.")

    return selected[:target_count]


def export_spectrum_csvs(
    output_dir: Path,
    sample_prefix: str,
    healthy_sheet: SheetData,
    disease_sheet: SheetData,
    names_by_sheet: dict[str, list[str]],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    export_plan = [
        ("health", "healthy", healthy_sheet),
        ("heart disease", "disease", disease_sheet),
    ]
    for sheet_name, group_name, sheet_data in export_plan:
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in group_dir.glob(f"{sample_prefix}_*.csv"):
            stale_file.unlink()
        for sample_name in names_by_sheet[sheet_name]:
            out_name = f"{sample_prefix}_{sample_name}"
            out_path = group_dir / f"{out_name}.csv"
            intensity = sheet_data.intensity_df[str(sample_name)].to_numpy(dtype=float)
            with out_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["wavenumber", "intensity"])
                for wn, value in zip(sheet_data.wavenumber, intensity):
                    writer.writerow([f"{float(wn):.12f}", f"{float(value):.12f}"])
            rows.append(
                {
                    "group": group_name,
                    "patient_name": out_name,
                    "relative_path": f"{group_name}/{out_name}.csv",
                    "source_dataset": NEW_DATASET_PATH.name,
                    "source_sample_name": str(sample_name),
                }
            )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8")
    return manifest_df


def build_sheet(
    sheet_name: str,
    old_sheet: SheetData,
    new_sheet: SheetData,
    new_metadata_df: pd.DataFrame,
    target_wavenumber: np.ndarray,
    training_new_names: list[str],
    holdout_new_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_name = SHEET_TO_CLASS[sheet_name]
    old_columns = old_sheet.intensity_df.columns.tolist()
    if len(old_columns) < OLD_SAMPLE_COUNT_PER_CLASS:
        raise ValueError(f"Old sheet '{sheet_name}' has only {len(old_columns)} samples.")
    old_selected_columns = old_columns[:OLD_SAMPLE_COUNT_PER_CLASS]

    old_selected_df = old_sheet.intensity_df.loc[:, old_selected_columns].copy()
    serum_all_names = training_new_names + holdout_new_names
    new_selected_df = new_sheet.intensity_df.loc[:, serum_all_names].copy()
    new_selected_df = interpolate_to_grid(new_sheet.wavenumber, new_selected_df, target_wavenumber)
    new_selected_df = new_selected_df.rename(columns=lambda name: f"{NEW_SAMPLE_PREFIX}_{name}")

    combined_df = pd.concat([old_selected_df, new_selected_df], axis=1)
    combined_df.insert(0, "wavenumber", target_wavenumber)

    meta_index = new_metadata_df.set_index("project_sample_name", drop=False)
    metadata_rows: list[dict[str, object]] = []
    for sample_name in old_selected_columns:
        metadata_rows.append(
            {
                "sheet_name": sheet_name,
                "combined_sample_name": sample_name,
                "source_dataset": OLD_DATASET_PATH.name,
                "source_sample_name": sample_name,
                "source_class": class_name,
                "patient_id": np.nan,
                "is_holdout": False,
            }
        )

    for sample_name in serum_all_names:
        meta_row = meta_index.loc[str(sample_name)]
        metadata_rows.append(
            {
                "sheet_name": sheet_name,
                "combined_sample_name": f"{NEW_SAMPLE_PREFIX}_{sample_name}",
                "source_dataset": NEW_DATASET_PATH.name,
                "source_sample_name": sample_name,
                "source_class": class_name,
                "patient_id": meta_row.get("patient_id", np.nan),
                "is_holdout": sample_name in set(holdout_new_names),
            }
        )

    return combined_df, pd.DataFrame(metadata_rows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    old_health = load_sheet(OLD_DATASET_PATH, "health")
    old_disease = load_sheet(OLD_DATASET_PATH, "heart disease")
    new_health = load_sheet(NEW_DATASET_PATH, "health")
    new_disease = load_sheet(NEW_DATASET_PATH, "heart disease")
    new_metadata_df = normalize_new_metadata(pd.read_excel(NEW_DATASET_PATH, sheet_name="metadata"))

    overlap_low = max(
        old_health.wavenumber.min(),
        old_disease.wavenumber.min(),
        new_health.wavenumber.min(),
        new_disease.wavenumber.min(),
    )
    overlap_high = min(
        old_health.wavenumber.max(),
        old_disease.wavenumber.max(),
        new_health.wavenumber.max(),
        new_disease.wavenumber.max(),
    )
    if overlap_low >= overlap_high:
        raise ValueError("Datasets do not have an overlapping wavenumber region.")

    target_mask = (old_health.wavenumber >= overlap_low) & (old_health.wavenumber <= overlap_high)
    target_wavenumber = old_health.wavenumber[target_mask].copy()
    if target_wavenumber.size == 0:
        raise ValueError("The overlap region produced an empty target grid.")

    old_health_overlap = SheetData(target_wavenumber, old_health.intensity_df.loc[target_mask].reset_index(drop=True))
    old_disease_overlap = SheetData(target_wavenumber, old_disease.intensity_df.loc[target_mask].reset_index(drop=True))

    healthy_meta = new_metadata_df.loc[new_metadata_df["normalized_class"] == "healthy"].copy()
    disease_meta = new_metadata_df.loc[new_metadata_df["normalized_class"] == "disease"].copy()

    healthy_holdout_names = choose_balanced_samples(healthy_meta, NEW_HOLDOUT_SAMPLE_COUNT_PER_CLASS, rng)
    disease_holdout_names = choose_balanced_samples(disease_meta, NEW_HOLDOUT_SAMPLE_COUNT_PER_CLASS, rng)

    healthy_training_names = choose_balanced_samples(
        healthy_meta,
        NEW_TRAIN_SAMPLE_COUNT_PER_CLASS,
        rng,
        exclude_names=set(healthy_holdout_names),
    )
    disease_training_names = choose_balanced_samples(
        disease_meta,
        NEW_TRAIN_SAMPLE_COUNT_PER_CLASS,
        rng,
        exclude_names=set(disease_holdout_names),
    )

    healthy_external_names = sorted(
        set(healthy_meta["project_sample_name"].astype(str).tolist())
        - set(healthy_training_names)
        - set(healthy_holdout_names)
    )
    disease_external_names = sorted(
        set(disease_meta["project_sample_name"].astype(str).tolist())
        - set(disease_training_names)
        - set(disease_holdout_names)
    )

    combined_health_df, health_meta_df = build_sheet(
        sheet_name="health",
        old_sheet=old_health_overlap,
        new_sheet=new_health,
        new_metadata_df=new_metadata_df,
        target_wavenumber=target_wavenumber,
        training_new_names=healthy_training_names,
        holdout_new_names=healthy_holdout_names,
    )
    combined_disease_df, disease_meta_df = build_sheet(
        sheet_name="heart disease",
        old_sheet=old_disease_overlap,
        new_sheet=new_disease,
        new_metadata_df=new_metadata_df,
        target_wavenumber=target_wavenumber,
        training_new_names=disease_training_names,
        holdout_new_names=disease_holdout_names,
    )

    manifest_df = export_spectrum_csvs(
        output_dir=EXPORT_DIR,
        sample_prefix=NEW_SAMPLE_PREFIX,
        healthy_sheet=new_health,
        disease_sheet=new_disease,
        names_by_sheet={
            "health": healthy_holdout_names,
            "heart disease": disease_holdout_names,
        },
    )
    external_manifest_df = export_spectrum_csvs(
        output_dir=EXTERNAL_EXPORT_DIR,
        sample_prefix=NEW_SAMPLE_PREFIX,
        healthy_sheet=new_health,
        disease_sheet=new_disease,
        names_by_sheet={
            "health": healthy_external_names,
            "heart disease": disease_external_names,
        },
    )

    combined_meta_df = pd.concat([health_meta_df, disease_meta_df], ignore_index=True)
    summary_df = pd.DataFrame(
        [
            {
                "old_dataset": OLD_DATASET_PATH.name,
                "new_dataset": NEW_DATASET_PATH.name,
                "output_dataset": OUTPUT_DATASET_PATH.name,
                "overlap_low_cm-1": float(overlap_low),
                "overlap_high_cm-1": float(overlap_high),
                "target_points": int(target_wavenumber.size),
                "old_samples_per_class": OLD_SAMPLE_COUNT_PER_CLASS,
                "new_train_samples_per_class": NEW_TRAIN_SAMPLE_COUNT_PER_CLASS,
                "new_holdout_samples_per_class": NEW_HOLDOUT_SAMPLE_COUNT_PER_CLASS,
                "total_samples_per_class": OLD_SAMPLE_COUNT_PER_CLASS + NEW_TRAIN_SAMPLE_COUNT_PER_CLASS + NEW_HOLDOUT_SAMPLE_COUNT_PER_CLASS,
                "random_seed": RANDOM_SEED,
            }
        ]
    )

    with pd.ExcelWriter(OUTPUT_DATASET_PATH, engine="openpyxl") as writer:
        combined_health_df.to_excel(writer, sheet_name="health", index=False)
        combined_disease_df.to_excel(writer, sheet_name="heart disease", index=False)
        combined_meta_df.to_excel(writer, sheet_name="metadata", index=False)
        summary_df.to_excel(writer, sheet_name="build_info", index=False)

    print(f"Built {OUTPUT_DATASET_PATH.name}")
    print(f"Overlap: {overlap_low:.3f} - {overlap_high:.3f} cm^-1")
    print(f"Grid points: {target_wavenumber.size}")
    print(
        "Samples in combined dataset: "
        f"{len(combined_health_df.columns) - 1} healthy, {len(combined_disease_df.columns) - 1} disease"
    )
    print(
        "Serum holdout exported: "
        f"{len(healthy_holdout_names)} healthy, {len(disease_holdout_names)} disease "
        f"to {manifest_df.shape[0]} CSV files"
    )
    print(
        "Serum external exported: "
        f"{len(healthy_external_names)} healthy, {len(disease_external_names)} disease "
        f"to {external_manifest_df.shape[0]} CSV files"
    )


if __name__ == "__main__":
    main()
