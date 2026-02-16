#!/usr/bin/env python3
"""
Export SLAM dataset from OpenSoundscape localization results (shelve file).

Reads position_estimates from a shelve file, filters by residual RMS and
minimum estimates per time window, averages positions, and writes the full
SLAM folder structure.

NOTE: This script assumes ARU coordinates were converted to UTM *before*
localization, so position_estimates are already in UTM (easting/northing).
No coordinate conversion is applied to position estimates.

Usage:
    python export_slam_CONW.py

Edit the configuration section below before running.

Requirements:
    pip install numpy pandas pyproj
"""

import os
import csv
import shelve
from collections import defaultdict

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION — edit these paths and parameters
# =============================================================================

# Input paths
SHELF_PATH = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/pythonoutput/conw_confirmed.out"
ARU_COORDS_PATH = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/aru_coords.csv"

# Output
PROJECT_NAME = "conw_localization_05312025"
OUTPUT_DIR = r"D:/BARLT Localization Project/localization_05312025/SLAM-CONW"

# Filtering parameters (should match your analysis choices)
RMS_THRESHOLD = 30          # max residual RMS (m) for "good" localizations
MIN_ESTIMATES_PER_WINDOW = 1  # min number of estimates to keep a time window
TARGET_CLASS = "index"      # class_name on CONW position objects (index bug from reset_index)

# UTM zone label for metadata (must match what you used before localization)
UTM_ZONE = "14N"

# Are ARU coords already in UTM? If True, aru_coords.csv has easting/northing.
# If False, aru_coords.csv has lon/lat and will be converted using UTM_EPSG.
ARU_COORDS_ALREADY_UTM = True
UTM_EPSG = "EPSG:32614"  # only used if ARU_COORDS_ALREADY_UTM = False


# =============================================================================
# Helper functions
# =============================================================================

def lonlat_to_utm(lons, lats, epsg):
    """Convert arrays of lon/lat to UTM easting/northing."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    eastings, northings = transformer.transform(lons, lats)
    return eastings, northings


# =============================================================================
# Main export
# =============================================================================

def main():
    root = os.path.join(OUTPUT_DIR, PROJECT_NAME)

    # ---- Load shelve --------------------------------------------------------
    print(f"Loading position estimates from: {SHELF_PATH}")
    with shelve.open(SHELF_PATH, "r") as db:
        position_estimates = db["position_estimates"]
    print(f"  Loaded {len(position_estimates)} raw estimates")

    # ---- Load ARU coordinates -----------------------------------------------
    aru_coords = pd.read_csv(ARU_COORDS_PATH, index_col=0)
    print(f"  Loaded {len(aru_coords)} ARU positions")

    if ARU_COORDS_ALREADY_UTM:
        aru_utm = aru_coords.copy()
        aru_utm.rename(columns={"x": "utm_easting", "y": "utm_northing"}, inplace=True)
        print(f"  ARU coords treated as UTM (no conversion)")
    else:
        eastings, northings = lonlat_to_utm(
            aru_coords["x"].values, aru_coords["y"].values, UTM_EPSG
        )
        aru_utm = aru_coords.copy()
        aru_utm["utm_easting"] = eastings
        aru_utm["utm_northing"] = northings
        print(f"  Converted ARU coords to UTM ({UTM_EPSG})")

    # ---- Filter estimates ---------------------------------------------------
    target_estimates = [
        e for e in position_estimates
        if e.class_name == TARGET_CLASS
        and e.residual_rms < RMS_THRESHOLD
        and np.isfinite(e.location_estimate[0])
        and np.isfinite(e.location_estimate[1])
    ]
    print(f"  {len(target_estimates)} estimates pass RMS < {RMS_THRESHOLD} m + finite coordinate filter")

    # ---- Group by time window and average -----------------------------------
    grouped = defaultdict(list)
    for e in target_estimates:
        grouped[e.start_timestamp].append(e)

    filtered_groups = {
        ts: events for ts, events in grouped.items()
        if len(events) >= MIN_ESTIMATES_PER_WINDOW
    }
    print(f"  {len(filtered_groups)} time windows with >= {MIN_ESTIMATES_PER_WINDOW} estimates")

    if len(filtered_groups) == 0:
        raise SystemExit("No events passed filters. Adjust RMS_THRESHOLD or MIN_ESTIMATES_PER_WINDOW.")

    # Build averaged event records
    events = []
    for i, (timestamp, estimates) in enumerate(
        sorted(filtered_groups.items(), key=lambda kv: kv[0])
    ):
        xs = [e.location_estimate[0] for e in estimates]
        ys = [e.location_estimate[1] for e in estimates]

        coords = [(x, y) for x, y in zip(xs, ys)
                  if np.isfinite(x) and np.isfinite(y)]

        if len(coords) < MIN_ESTIMATES_PER_WINDOW:
            continue

        x_avg = np.mean([c[0] for c in coords])
        y_avg = np.mean([c[1] for c in coords])

        rms_values = [e.residual_rms for e in estimates]
        mean_rms = np.mean(rms_values)
        n_estimates = len(coords)
        duration = estimates[0].duration

        all_file_ids = set()
        for e in estimates:
            all_file_ids.update(e.receiver_files)

        events.append({
            "event_id": f"CONW_{i:05d}",
            "label": TARGET_CLASS,
            "start_timestamp": timestamp.isoformat(),
            "duration": duration,
            "position_x": round(x_avg, 2),
            "position_y": round(y_avg, 2),
            "position_z": "",
            "utm_zone": UTM_ZONE,
            "file_ids": ";".join(sorted(all_file_ids)),
            "file_start_time_offsets": "",
            "mean_residual_rms": round(mean_rms, 2),
            "n_estimates": n_estimates,
        })

    print(f"  {len(events)} averaged events to write")

    # ---- Create directory structure -----------------------------------------
    dirs = [
        root,
        os.path.join(root, "script"),
        os.path.join(root, "localization_metadata"),
        os.path.join(root, "observed_events"),
        os.path.join(root, "audio"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # ---- Write localized_events.csv -----------------------------------------
    events_path = os.path.join(root, "localized_events.csv")
    events_fields = [
        "event_id", "label", "start_timestamp", "duration",
        "position_x", "position_y", "position_z", "utm_zone",
        "file_ids", "file_start_time_offsets",
        "mean_residual_rms", "n_estimates",
    ]
    write_csv(events_path, events_fields, events)
    print(f"  Wrote {len(events)} events to localized_events.csv")

    # ---- Write classes.csv --------------------------------------------------
    classes_path = os.path.join(root, "classes.csv")
    write_csv(classes_path,
        ["class", "species", "scientific_name", "vocalization_type", "description"],
        [{"class": "CONW", "species": "Connecticut Warbler",
          "scientific_name": "Oporornis agilis",
          "vocalization_type": "song", "description": ""}],
    )

    # ---- Write point_table.csv (ARU positions) ------------------------------
    pt_path = os.path.join(root, "localization_metadata", "point_table.csv")
    pt_fields = ["point_id", "utm_easting", "utm_northing", "elevation", "utm_zone"]
    pt_rows = []
    for idx, row in aru_utm.iterrows():
        pt_rows.append({
            "point_id": idx,
            "utm_easting": round(row["utm_easting"], 3),
            "utm_northing": round(row["utm_northing"], 3),
            "elevation": "",
            "utm_zone": UTM_ZONE,
        })
    write_csv(pt_path, pt_fields, pt_rows)
    print(f"  Wrote {len(pt_rows)} ARU positions to point_table.csv")

    # ---- Write audio_file_table.csv (placeholder) ---------------------------
    aft_path = os.path.join(root, "localization_metadata", "audio_file_table.csv")
    write_csv(aft_path,
        ["file_id", "relative_path", "point_id", "start_timestamp"],
        [],
    )

    # ---- Write observed_events placeholders ---------------------------------
    write_csv(
        os.path.join(root, "observed_events", "playbacks.csv"),
        ["playback_id", "class_label", "start_timestamp", "duration",
         "position_x", "position_y", "position_z", "utm_zone"],
        [],
    )
    write_csv(
        os.path.join(root, "observed_events", "observations.csv"),
        ["observed_event_id", "class_label", "start_timestamp", "duration",
         "position_x", "position_y", "position_z", "utm_zone",
         "direction", "comments"],
        [],
    )

    # ---- Write environment.yml placeholder ----------------------------------
    env_path = os.path.join(root, "script", "environment.yml")
    with open(env_path, "w") as f:
        f.write("# Conda environment for reproducing localization pipeline\n")
        f.write("# Generate frozen env with: conda env export > environment.yml\n")
        f.write("name: slam_conw\n")
        f.write("dependencies:\n")
        f.write("  - python>=3.9\n")
        f.write("  - opensoundscape\n")
        f.write("  - pandas\n")
        f.write("  - numpy\n")
        f.write("  - pyproj\n")
        f.write("  - pytz\n")

    # ---- Write README -------------------------------------------------------
    readme_path = os.path.join(root, "readme.md")
    with open(readme_path, "w") as f:
        f.write(build_readme(
            project_name=PROJECT_NAME,
            n_events=len(events),
            n_arus=len(aru_coords),
            utm_zone=UTM_ZONE,
            rms_threshold=RMS_THRESHOLD,
            min_estimates=MIN_ESTIMATES_PER_WINDOW,
        ))

    # ---- Summary ------------------------------------------------------------
    print(f"\nSLAM dataset written to: {os.path.abspath(root)}")
    print("\nDirectory contents:")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            size = os.path.getsize(fpath)
            print(f"{'  ' * (level + 1)}{fname}  ({size:,} bytes)")


def write_csv(filepath, fieldnames, rows):
    """Write a CSV with headers and optional rows (list of dicts)."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_readme(project_name, n_events, n_arus, utm_zone, rms_threshold, min_estimates):
    """Generate a partially filled SLAM readme."""
    return f"""## {project_name}

- Creators: Megan Edgar
  - [fill in email]
- Affiliations: University of Alberta
- Version number [v1]
- DOI:

### General characteristics

- audio format: [fill in — short clips or long recordings]
- dimensions localized: 2
- number of localization arrays: 1
- array geometry: [fill in — describe shape and spacing]
- sounds localized: Connecticut Warbler (CONW) songs
- number of localized events: {n_events}
- number of ARUs: {n_arus}
- size: [fill in] GB

## Study description

Study purpose: Acoustic localization of Connecticut Warbler (Oporornis agilis)
songs using a synchronized recorder array in the boreal forest.

Personnel: [fill in — who led, managed, participated]

Data types collected: audio recordings, automated species detections,
acoustic localizations

Notes:
- Data collected at the BARLT localization project site
- [describe site characteristics, vegetation, etc.]

## Files

localized_events.csv: {n_events} acoustically localized Connecticut Warbler
song events, averaged across estimates per detection time window.

Columns:
- event_id: unique ID (format CONW_XXXXX)
- label: species alpha code (CONW)
- start_timestamp: event onset in ISO 8601 format
- duration: detection window length in seconds
- position_x: UTM easting (m), zone {utm_zone}
- position_y: UTM northing (m), zone {utm_zone}
- position_z: [empty — 2D localization]
- utm_zone: {utm_zone}
- file_ids: semicolon-separated list of receiver files contributing to localization
- file_start_time_offsets: [empty — not applicable for averaged events]
- mean_residual_rms: mean residual RMS (m) across constituent estimates
- n_estimates: number of position estimates averaged for this event

classes.csv: species lookup table

/script/: localization and export scripts
/script/environment.yml: conda environment

/localization_metadata/audio_file_table.csv: audio file index
/localization_metadata/point_table.csv: {n_arus} ARU positions in UTM

/audio/: [populate with audio clips]
/observed_events/: [optional playback and field observation records]

## Sites

- Site name: BARLT, [Province], Canada
- Ecosystem description: [fill in — e.g. boreal mixedwood forest]

## Hardware

- Recorder source: [fill in — e.g. Open Acoustic Devices]
- Recorder model: [fill in — e.g. AudioMoth with GPS]
- Firmware version: [fill in]

## Recording properties

Date range of data: 2025-05-31 to [fill in]

Recording schedule description:
- Times of recording: [fill in]
- Sleep-wake schedule: [fill in]
- Sample rate: [fill in] Hz
- Other relevant settings: [fill in — e.g. gain]

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
- Coordinates converted from WGS84 lat/lon to UTM zone {utm_zone} using pyproj
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
- Connecticut Warbler (CONW) songs detected using HawkEars at 0.7 score threshold

- Detection strategy: convolutional neural network
- Detector name and version: HawkEars [fill in version]
- Link to detector information: [fill in URL]
- Post-processing detector outputs:
  - Binarization/thresholding: score threshold >= 0.7
  - Manual review process: [fill in if applicable]
- Scripts/resources: [fill in]

## Localization

- Tools/packages used: OpenSoundscape (opensoundscape.localization.SynchronizedRecorderArray)
- Localization algorithm: correlation-sum
- Time delay calculation algorithm: GCC (generalized cross-correlation)
- References: [fill in OpenSoundscape citation]
- Error rejection parameters:
  - min_n_receivers: 5
  - max_receiver_dist: 80 m
  - Residual RMS threshold: {rms_threshold} m
  - Minimum estimates per time window: {min_estimates}
- Scripts: export_slam_CONW.py, localization_CONW.py

### Manual review

- [fill in any manual review of localizations]

## Acknowledgements

Adapted from Erica Alex's localization script.
[fill in additional acknowledgements, data ownership, restrictions]

## License

[fill in — e.g. CC0 https://creativecommons.org/public-domain/cc0/]

## Works Cited and Links

- OpenSoundscape: [citation and URL]
- HawkEars: [citation and URL]
"""


if __name__ == "__main__":
    main()