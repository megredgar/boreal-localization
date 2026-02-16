# -*- coding: utf-8 -*-
"""
Batch localization of multiple species using opensoundscape.
Each species is processed independently: threshold → detect → localize → save.

Adapted from Erica Alex's script by Megan Edgar
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import shelve

from pyproj import Transformer
from opensoundscape.localization import SynchronizedRecorderArray
from opensoundscape import Spectrogram

# =============================================================================
# CONFIGURATION
# =============================================================================

# Raw HawkEars output (low threshold, all species)
LABELS_PATH = r"D:/BARLT Localization Project/localization_05312025/hawkears_lowthresh/HawkEars_labels.csv"

# Directory containing the trimmed wav files
WAV_DIR = r"D:/BARLT Localization Project/localization_05312025/localizationtrim_new"

# Site coordinates (lat/lon)
SITES_PATH = r"D:/BARLT Localization Project/LocalizationSites_CWS_2025.csv"

# Base output directory — each species gets a subfolder
BASE_OUT_DIR = r"D:/BARLT Localization Project/localization_05312025"

# Species to process and their thresholds
SPECIES_CONFIG = {
    "RCKI": 0.7,
    "TEWA": 0.7,
    "RWBL": 0.7,
    "CHSP": 0.7,
    "YEWA": 0.7,
    "WTSP": 0.7,
    "AMRO": 0.7,
    "YBFL": 0.7,
    "AMRE": 0.7,
}

# Localization parameters
MIN_N_RECEIVERS = 5
MAX_RECEIVER_DIST = 80  # metres

# Recording start timestamp (latest trimmed recording)
LOCAL_TIMESTAMP = datetime(2025, 5, 31, 5, 23, 42)
LOCAL_TIMEZONE = pytz.timezone("America/Winnipeg")

# UTM conversion
UTM_EPSG = "EPSG:32614"

# ARU ID regex pattern from filenames
ARU_ID_PATTERN = r"L\d+N\d+E\d+"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_raw_labels(labels_path):
    """Load the full HawkEars labels CSV."""
    raw = pd.read_csv(labels_path)
    # Standardize column names if needed
    # Expected: filename, start_time, end_time, class_name, class_code, score
    return raw


def extract_aru_id(filename):
    """Extract ARU ID (e.g., L1N5E1) from filename."""
    match = pd.Series([filename]).str.extract(ARU_ID_PATTERN)
    if match.isna().values[0][0]:
        return None
    return match.values[0][0]


def build_species_detections(raw_labels, species_code, threshold, wav_dir):
    """
    Filter raw HawkEars labels for one species at a given threshold,
    then build the 3-second binned detection matrix.

    Returns: (detections_df, aru_coords_df) or (None, None) if no detections.
    """
    # Filter for this species and threshold
    det = raw_labels.copy()
    det["aru_id"] = det["filename"].str.extract(f"({ARU_ID_PATTERN})")
    det["file"] = det["filename"].apply(lambda f: os.path.join(wav_dir, f))
    det["tag_start"] = pd.to_numeric(det["start_time"], errors="coerce")
    det["tag_end"] = pd.to_numeric(det["end_time"], errors="coerce")

    det = det.dropna(subset=["aru_id"])
    det = det[det["aru_id"] != ""]
    det = det[det["class_code"] == species_code]
    det = det[det["score"] >= threshold]

    if len(det) == 0:
        return None, None

    print(f"    Detections after threshold ({threshold}): {len(det)}")

    # Build aru_coords from files that have detections
    sites_clean = pd.read_csv(SITES_PATH).rename(
        columns={"SiteID": "aru_id", "Longitude": "x", "Latitude": "y"}
    )[["aru_id", "x", "y"]]

    aru_coords = (
        det[["file", "aru_id"]]
        .drop_duplicates()
        .merge(sites_clean, on="aru_id", how="left")
        [["file", "x", "y"]]
        .drop_duplicates()
    )

    missing = aru_coords[aru_coords["x"].isna() | aru_coords["y"].isna()]
    if len(missing) > 0:
        print(f"    WARNING: {len(missing)} files have missing coordinates")

    aru_coords = aru_coords.dropna(subset=["x", "y"])
    aru_coords = aru_coords.set_index("file")

    # Build 3-second bins
    max_end = det["tag_end"].max()
    bins = pd.DataFrame({
        "bin_start": np.arange(0, max_end + 3, 3)
    })
    bins["bin_end"] = bins["bin_start"] + 3

    all_files = det[["file"]].drop_duplicates()
    file_bins = all_files.merge(bins, how="cross")

    # Find which bins overlap with detections
    hits = det.merge(bins, how="cross")
    hits = hits[~((hits["bin_end"] <= hits["tag_start"]) | (hits["bin_start"] >= hits["tag_end"]))]
    hits = hits[["file", "bin_start", "bin_end"]].drop_duplicates()
    hits[species_code] = 1

    # Join back to full grid
    detections = file_bins.merge(
        hits, on=["file", "bin_start", "bin_end"], how="left"
    )
    detections[species_code] = detections[species_code].fillna(0).astype(int)
    detections = detections.sort_values(["file", "bin_start"]).rename(
        columns={"bin_start": "start_time", "bin_end": "end_time"}
    )

    return detections, aru_coords


def prepare_detections_for_localization(detections_df, species_code):
    """
    Add start_timestamp and set the multi-index expected by OpenSoundscape.
    Drops any leftover 'index' column to prevent class_name='index' bug.
    """
    det = detections_df.reset_index(drop=True)

    # Drop leftover index column if present
    if "index" in det.columns:
        det = det.drop(columns=["index"])

    det["start_timestamp"] = [
        LOCAL_TIMEZONE.localize(LOCAL_TIMESTAMP) + timedelta(seconds=s)
        for s in det["start_time"]
    ]

    det = det.set_index(["file", "start_time", "end_time", "start_timestamp"])

    # Verify the only remaining column is the species code
    remaining = det.columns.tolist()
    if remaining != [species_code]:
        print(f"    WARNING: Expected columns [{species_code}], got {remaining}")

    return det


def convert_coords_to_utm(aru_coords):
    """Convert lat/lon aru_coords to UTM. Returns a copy."""
    coords = aru_coords.copy()
    transformer = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
    coords["x"], coords["y"] = transformer.transform(
        coords["x"].values, coords["y"].values
    )
    return coords


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("Loading raw HawkEars labels...")
    raw_labels = pd.read_csv(LABELS_PATH)
    print(f"  Total rows: {len(raw_labels)}")
    print(f"  Species found: {sorted(raw_labels['class_code'].unique())}\n")

    results_summary = []

    for species_code, threshold in SPECIES_CONFIG.items():
        print(f"{'='*60}")
        print(f"  Processing: {species_code} (threshold = {threshold})")
        print(f"{'='*60}")

        # Output directory for this species
        species_dir = os.path.join(BASE_OUT_DIR, f"hawkears_0_7_{species_code}")
        os.makedirs(species_dir, exist_ok=True)

        # --- Step 1: Build detections and aru_coords ---
        detections_df, aru_coords_df = build_species_detections(
            raw_labels, species_code, threshold, WAV_DIR
        )

        if detections_df is None:
            print(f"    No detections for {species_code} at threshold {threshold}. Skipping.\n")
            results_summary.append({
                "species": species_code, "threshold": threshold,
                "detections": 0, "position_estimates": 0, "status": "no detections"
            })
            continue

        # Save intermediate CSVs
        aru_coords_path = os.path.join(species_dir, "aru_coords.csv")
        detections_path = os.path.join(species_dir, f"detections_{species_code}.csv")
        aru_coords_df.to_csv(aru_coords_path)
        detections_df.to_csv(detections_path, index=False)

        # --- Step 2: Convert to UTM ---
        aru_coords_utm = convert_coords_to_utm(aru_coords_df)

        # Check we have enough ARUs
        n_arus = len(aru_coords_utm)
        print(f"    ARUs with detections: {n_arus}")
        if n_arus < MIN_N_RECEIVERS:
            print(f"    Not enough ARUs ({n_arus} < {MIN_N_RECEIVERS}). Skipping.\n")
            results_summary.append({
                "species": species_code, "threshold": threshold,
                "detections": len(detections_df), "position_estimates": 0,
                "status": f"only {n_arus} ARUs"
            })
            continue

        # --- Step 3: Prepare for localization ---
        array = SynchronizedRecorderArray(aru_coords_utm)

        det_for_loc = prepare_detections_for_localization(detections_df, species_code)

        # Sanity check file overlap
        det_files = set(det_for_loc.index.get_level_values("file").unique())
        coord_files = set(array.file_coords.index)
        print(f"    Files in detections: {len(det_files)}")
        print(f"    Files in coords: {len(coord_files)}")
        print(f"    Missing coords: {len(det_files - coord_files)}")

        # --- Step 4: Localize ---
        print(f"    Running localization...")
        try:
            position_estimates = array.localize_detections(
                det_for_loc,
                min_n_receivers=MIN_N_RECEIVERS,
                max_receiver_dist=MAX_RECEIVER_DIST,
            )
        except Exception as e:
            print(f"    ERROR during localization: {e}")
            results_summary.append({
                "species": species_code, "threshold": threshold,
                "detections": len(detections_df), "position_estimates": 0,
                "status": f"localization error: {e}"
            })
            continue

        n_estimates = len(position_estimates)
        print(f"    Position estimates: {n_estimates}")

        if n_estimates == 0:
            print(f"    No position estimates for {species_code}. Skipping save.\n")
            results_summary.append({
                "species": species_code, "threshold": threshold,
                "detections": len(detections_df), "position_estimates": 0,
                "status": "no estimates"
            })
            continue

        # --- Step 5: Quick RMS summary ---
        rms_values = [e.residual_rms for e in position_estimates]
        print(f"    RMS — min: {min(rms_values):.1f}, median: {np.median(rms_values):.1f}, "
              f"max: {max(rms_values):.1f}, <30m: {sum(1 for r in rms_values if r < 30)}")

        # --- Step 6: Save shelf ---
        shelf_dir = os.path.join(species_dir, "pythonoutput")
        os.makedirs(shelf_dir, exist_ok=True)
        shelf_path = os.path.join(shelf_dir, f"{species_code.lower()}.out")

        with shelve.open(shelf_path, "n") as my_shelf:
            my_shelf["position_estimates"] = position_estimates

        print(f"    Saved to: {shelf_path}\n")

        results_summary.append({
            "species": species_code, "threshold": threshold,
            "detections": len(detections_df),
            "position_estimates": n_estimates,
            "rms_median": round(np.median(rms_values), 1),
            "rms_lt30": sum(1 for r in rms_values if r < 30),
            "status": "ok"
        })

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("  BATCH SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(BASE_OUT_DIR, "batch_localization_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()