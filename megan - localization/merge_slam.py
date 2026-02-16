#!/usr/bin/env python3
"""
Merge individual species SLAM folders into a single master SLAM dataset.

Combines localized_events.csv, classes.csv, audio_file_table.csv, and
point_table.csv from all species. Copies (or symlinks) audio folders.
Writes a master README.

Usage:
    python merge_slam.py
"""

import os
import csv
import shutil
from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(r"D:/BARLT Localization Project/localization_05312025")

SPECIES_FOLDERS = {
    "VEER": "SLAM-VEER",
    "ALFL": "SLAM-ALFL",
    "AMRE": "SLAM-AMRE",
    "CONW": "SLAM-CONW",
    "COYE": "SLAM-COYE",
    "MAWA": "SLAM-MAWA",
    "TEWA": "SLAM-TEWA",
    "WTSP": "SLAM-WTSP",
    "YBFL": "SLAM-YBFL",
}

# Master output
MASTER_PROJECT = "barlt_localization_05312025"
MASTER_DIR = BASE_DIR / "SLAM-MASTER" / MASTER_PROJECT

# Species metadata for classes.csv and README
SPECIES_INFO = {
    "VEER": ("Veery", "Catharus fuscescens"),
    "ALFL": ("Alder Flycatcher", "Empidonax alnorum"),
    "AMRE": ("American Redstart", "Setophaga ruticilla"),
    "CONW": ("Connecticut Warbler", "Oporornis agilis"),
    "COYE": ("Common Yellowthroat", "Geothlypis trichas"),
    "MAWA": ("Magnolia Warbler", "Setophaga magnolia"),
    "TEWA": ("Tennessee Warbler", "Leiothlypis peregrina"),
    "WTSP": ("White-throated Sparrow", "Zonotrichia albicollis"),
    "YBFL": ("Yellow-bellied Flycatcher", "Empidonax flaviventris"),
}

UTM_ZONE = "14N"

# Set to True to COPY audio files into the master folder.
# Set to False to skip audio (saves disk space; you can manually link later).
COPY_AUDIO = True


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Create master directory structure
    for subdir in ["script", "localization_metadata", "observed_events", "audio"]:
        (MASTER_DIR / subdir).mkdir(parents=True, exist_ok=True)

    all_events = []
    all_classes = []
    all_audio_files = []
    all_point_ids = {}  # deduplicate across species
    species_stats = {}

    for species_code, folder_name in sorted(SPECIES_FOLDERS.items()):
        slam_dir = BASE_DIR / folder_name

        # Find the project subfolder (e.g., veer_localization_05312025/)
        project_dirs = [d for d in slam_dir.iterdir() if d.is_dir()]
        if len(project_dirs) == 0:
            print(f"  {species_code}: No project folder found in {slam_dir}. Skipping.")
            continue

        project_dir = project_dirs[0]
        print(f"  {species_code}: Reading from {project_dir.name}")

        # --- localized_events.csv ---
        events_path = project_dir / "localized_events.csv"
        if events_path.exists():
            df = pd.read_csv(events_path)
            n_events = len(df)
            all_events.append(df)
        else:
            n_events = 0
            print(f"    WARNING: No localized_events.csv")

        # --- classes.csv ---
        classes_path = project_dir / "classes.csv"
        if classes_path.exists():
            df = pd.read_csv(classes_path)
            all_classes.append(df)

        # --- audio_file_table.csv ---
        aft_path = project_dir / "localization_metadata" / "audio_file_table.csv"
        if aft_path.exists():
            df = pd.read_csv(aft_path)
            if len(df) > 0:
                # Prefix relative paths with species subfolder
                df["relative_path"] = df["relative_path"].apply(
                    lambda p: os.path.join("audio", species_code, *Path(p).parts[1:])
                )
                all_audio_files.append(df)

        # --- point_table.csv ---
        pt_path = project_dir / "localization_metadata" / "point_table.csv"
        if pt_path.exists():
            df = pd.read_csv(pt_path)
            for _, row in df.iterrows():
                pid = row["point_id"]
                if pid not in all_point_ids:
                    all_point_ids[pid] = row.to_dict()

        # --- Copy audio ---
        if COPY_AUDIO:
            src_audio = project_dir / "audio"
            if src_audio.exists():
                dst_audio = MASTER_DIR / "audio" / species_code
                if dst_audio.exists():
                    shutil.rmtree(dst_audio)
                shutil.copytree(src_audio, dst_audio)
                n_clips = sum(1 for _ in dst_audio.rglob("*.flac")) + sum(1 for _ in dst_audio.rglob("*.wav"))
                print(f"    Copied {n_clips} audio clips")
            else:
                n_clips = 0
        else:
            n_clips = 0

        species_stats[species_code] = {"events": n_events, "clips": n_clips}

    # =========================================================================
    # Write merged CSVs
    # =========================================================================

    # --- localized_events.csv ---
    if all_events:
        merged_events = pd.concat(all_events, ignore_index=True)
        merged_events = merged_events.sort_values("start_timestamp").reset_index(drop=True)
        merged_events.to_csv(MASTER_DIR / "localized_events.csv", index=False)
        print(f"\nMerged localized_events.csv: {len(merged_events)} events")
    else:
        merged_events = pd.DataFrame()
        print("\nWARNING: No events found across any species.")

    # --- classes.csv (deduplicated, plus fill in any missing from SPECIES_INFO) ---
    classes_rows = []
    seen_classes = set()
    if all_classes:
        for df in all_classes:
            for _, row in df.iterrows():
                if row["class"] not in seen_classes:
                    classes_rows.append(row.to_dict())
                    seen_classes.add(row["class"])

    # Add any species from SPECIES_INFO not already present
    for code, (common, sci) in SPECIES_INFO.items():
        if code not in seen_classes:
            classes_rows.append({
                "class": code, "species": common,
                "scientific_name": sci,
                "vocalization_type": "song", "description": "",
            })

    classes_df = pd.DataFrame(classes_rows).sort_values("class").reset_index(drop=True)
    classes_df.to_csv(MASTER_DIR / "classes.csv", index=False)
    print(f"classes.csv: {len(classes_df)} species")

    # --- audio_file_table.csv ---
    if all_audio_files:
        merged_aft = pd.concat(all_audio_files, ignore_index=True)
        merged_aft.to_csv(
            MASTER_DIR / "localization_metadata" / "audio_file_table.csv", index=False
        )
        print(f"audio_file_table.csv: {len(merged_aft)} rows")
    else:
        pd.DataFrame(columns=["file_id", "relative_path", "point_id", "start_timestamp"]).to_csv(
            MASTER_DIR / "localization_metadata" / "audio_file_table.csv", index=False
        )

    # --- point_table.csv (deduplicated across species) ---
    pt_df = pd.DataFrame(list(all_point_ids.values()))
    if len(pt_df) > 0:
        pt_df = pt_df.sort_values("point_id").reset_index(drop=True)
    pt_df.to_csv(MASTER_DIR / "localization_metadata" / "point_table.csv", index=False)
    print(f"point_table.csv: {len(pt_df)} unique ARU positions")

    # --- Placeholder CSVs ---
    for fname, fields in [
        ("observed_events/playbacks.csv",
         ["playback_id", "class_label", "start_timestamp", "duration",
          "position_x", "position_y", "position_z", "utm_zone"]),
        ("observed_events/observations.csv",
         ["observed_event_id", "class_label", "start_timestamp", "duration",
          "position_x", "position_y", "position_z", "utm_zone",
          "direction", "comments"]),
    ]:
        path = MASTER_DIR / fname
        pd.DataFrame(columns=fields).to_csv(path, index=False)

    # --- environment.yml ---
    with open(MASTER_DIR / "script" / "environment.yml", "w") as f:
        f.write("name: slam_barlt_master\n")
        f.write("dependencies:\n  - python>=3.9\n  - opensoundscape\n")
        f.write("  - pandas\n  - numpy\n  - pyproj\n  - pytz\n")

    # =========================================================================
    # Write master README
    # =========================================================================
    n_total_events = len(merged_events) if len(merged_events) > 0 else 0
    n_species = len([s for s, st in species_stats.items() if st["events"] > 0])

    # Build per-species summary table for README
    species_table_lines = []
    for code in sorted(species_stats.keys()):
        common, sci = SPECIES_INFO.get(code, (code, ""))
        n_ev = species_stats[code]["events"]
        species_table_lines.append(f"| {code} | {common} | *{sci}* | {n_ev} |")

    species_table = "\n".join(species_table_lines)

    readme = f"""## {MASTER_PROJECT}

- Creators: Megan Edgar
  - [fill in email]
- Affiliations: University of Alberta
- Version number: [v1]
- DOI:

### General characteristics

- audio format: [fill in]
- dimensions localized: 2
- number of localization arrays: 1
- array geometry: [fill in — describe shape and spacing]
- species localized: {n_species}
- total localized events: {n_total_events}
- number of ARUs: {len(pt_df)}
- size: [fill in] GB

## Study description

Study purpose: Multi-species acoustic localization of songbird vocalizations
using a synchronized recorder array in the boreal forest at the BARLT
localization project site, Manitoba, Canada.

Personnel: [fill in]

Data types collected: audio recordings, automated species detections
(HawkEars CNN classifier), acoustic localizations (OpenSoundscape)

## Species summary

| Code | Common name | Scientific name | Events |
|------|-------------|-----------------|--------|
{species_table}

## Files

**localized_events.csv**: {n_total_events} acoustically localized song events
across {n_species} species, averaged across estimates per detection time window.

Columns:
- event_id: unique ID (format XXXX_NNNNN, where XXXX is the species alpha code)
- label: species alpha code
- start_timestamp: event onset in ISO 8601 format
- duration: detection window length in seconds
- position_x: UTM easting (m), zone {UTM_ZONE}
- position_y: UTM northing (m), zone {UTM_ZONE}
- position_z: [empty — 2D localization]
- utm_zone: {UTM_ZONE}
- file_ids: semicolon-separated list of receiver files contributing to localization
- file_start_time_offsets: semicolon-separated offsets (seconds) into source recordings
- mean_residual_rms: mean residual RMS (m) across constituent estimates
- n_estimates: number of position estimates averaged for this event

**classes.csv**: species lookup table ({len(classes_df)} species)

**/script/**: localization and export scripts
**/script/environment.yml**: conda environment

**/localization_metadata/audio_file_table.csv**: audio file index
**/localization_metadata/point_table.csv**: {len(pt_df)} ARU positions in UTM

**/audio/**: audio clips organized by species then by ARU site
  - audio/VEER/L1N5E1/VEER_00001_L1N5E1.flac
  - audio/TEWA/L1N3E5/TEWA_00042_L1N3E5.flac
  - (etc.)

**/observed_events/**: [optional playback and field observation records]

## Sites

- Site name: BARLT, Manitoba, Canada
- Ecosystem description: [fill in — e.g. boreal mixedwood forest]

## Hardware

- Recorder source: [fill in]
- Recorder model: [fill in]
- Firmware version: [fill in]

## Recording properties

Date range of data: 2025-05-31 to [fill in]

Recording schedule description:
- Times of recording: [fill in]
- Sleep-wake schedule: [fill in]
- Sample rate: [fill in] Hz
- Other relevant settings: [fill in]

Data aggregation notes:
- Recordings trimmed to common start time (latest recorder: L1N5E1,
  start 2025-05-31 05:23:42 CDT)

## Recorder positioning

Placement:
- Spatial pattern or geometry: [fill in]
- Range of spacing between adjacent mics: [fill in]
- Dimensions of array: [fill in]

Deployment:
- [fill in mounting, housing, orientation details]

Position measurement:
- Method of measurement: [fill in — e.g. RTK GPS]
- Measurement postprocessing: [fill in]
- Coordinates converted from WGS84 lat/lon to UTM zone {UTM_ZONE} using pyproj
- General accuracy (lat/long): [fill in]
- General accuracy (elevation): not measured (2D localization)

## Synchronization

Synchronization type: [fill in — e.g. GPS]
Synchronization frequency: [fill in]

Synchronization method:
- Methods used: [fill in]
- Recording start/end time trimming: recordings trimmed to common start time
- Scripts/resources: [fill in]

## Sound detection

Audio preprocessing:
- [fill in any resampling or denoising]

Classes localized:
- All species detected using HawkEars CNN classifier
- Species-specific score thresholds applied (0.7 for most species,
  0.2 for VEER, COYE, MAWA — [verify and adjust])

Detection pipeline:
1. Initial detection: HawkEars on full recordings (low threshold)
2. Localization: OpenSoundscape SynchronizedRecorderArray
3. Minimum spectrogram filtering of localized events
4. Re-confirmation: HawkEars on min-spec clips (species-specific threshold)

- Detection strategy: convolutional neural network
- Detector name and version: HawkEars [fill in version]
- Link to detector information: [fill in URL]
- Scripts/resources: batch_localization.py, batch_minspec.py

## Localization

- Tools/packages used: OpenSoundscape (opensoundscape.localization.SynchronizedRecorderArray)
- Localization algorithm: correlation-sum
- Time delay calculation algorithm: GCC (generalized cross-correlation)
- References: [fill in OpenSoundscape citation]
- Error rejection parameters:
  - min_n_receivers: 5
  - max_receiver_dist: 80 m
  - Residual RMS threshold: varies by species (30–50 m)
  - Minimum estimates per time window: 1
- Scripts: batch_localization.py, batch_export_slam.py, batch_extract_clips.py, merge_slam.py

### Manual review

- [fill in any manual review of localizations]

## Acknowledgements

Adapted from Erica Alex's localization scripts.
[fill in additional acknowledgements, data ownership, restrictions]

## License

[fill in — e.g. CC0 https://creativecommons.org/public-domain/cc0/]

## Works Cited and Links

- OpenSoundscape: [citation and URL]
- HawkEars: [citation and URL]
"""

    with open(MASTER_DIR / "readme.md", "w") as f:
        f.write(readme)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  MASTER SLAM DATASET")
    print(f"{'='*60}")
    print(f"  Location: {MASTER_DIR}")
    print(f"  Species:  {n_species}")
    print(f"  Events:   {n_total_events}")
    print(f"  ARUs:     {len(pt_df)}")
    print()

    for dirpath, dirnames, filenames in os.walk(str(MASTER_DIR)):
        level = dirpath.replace(str(MASTER_DIR), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        if level < 2:  # only show files for top 2 levels
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                size = os.path.getsize(fpath)
                print(f"{'  ' * (level + 1)}{fname}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
