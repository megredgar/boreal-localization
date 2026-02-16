#!/usr/bin/env python3
"""
Batch SLAM export for multiple species.

For each species, loads confirmed position estimates, filters/averages,
and writes the full SLAM folder structure.
"""

import os
import csv
import shelve
from collections import defaultdict

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = r"D:/BARLT Localization Project/localization_05312025"

SPECIES_LIST = ["RCKI", "TEWA", "RWBL", "CHSP", "YEWA", "WTSP", "AMRO", "YBFL", "AMRE"]

# Scientific names for classes.csv and README
SPECIES_INFO = {
    "RCKI": ("Ruby-crowned Kinglet", "Corthylio calendula"),
    "TEWA": ("Tennessee Warbler", "Leiothlypis peregrina"),
    "RWBL": ("Red-winged Blackbird", "Agelaius phoeniceus"),
    "CHSP": ("Chipping Sparrow", "Spizella passerina"),
    "YEWA": ("Yellow Warbler", "Setophaga petechia"),
    "WTSP": ("White-throated Sparrow", "Zonotrichia albicollis"),
    "AMRO": ("American Robin", "Turdus migratorius"),
    "YBFL": ("Yellow-bellied Flycatcher", "Empidonax flaviventris"),
    "AMRE": ("American Redstart", "Setophaga ruticilla"),
}

RMS_THRESHOLD = 50
MIN_ESTIMATES_PER_WINDOW = 1
UTM_ZONE = "14N"
UTM_EPSG = "EPSG:32614"


# =============================================================================
# HELPERS
# =============================================================================

def write_csv(filepath, fieldnames, rows):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_target_class(species_code, shelf_path):
    """Check what class_name is on the position objects."""
    with shelve.open(shelf_path, "r") as db:
        pe = db["position_estimates"]
    if len(pe) == 0:
        return species_code
    # Return the actual class_name from the first estimate
    return pe[0].class_name


# =============================================================================
# MAIN
# =============================================================================

def main():
    summary = []

    for species_code in SPECIES_LIST:
        print(f"\n{'='*60}")
        print(f"  SLAM export: {species_code}")
        print(f"{'='*60}")

        species_dir = os.path.join(BASE_DIR, f"hawkears_0_7_{species_code}")
        shelf_path = os.path.join(species_dir, "pythonoutput", f"{species_code.lower()}_confirmed.out")
        aru_coords_path = os.path.join(species_dir, "aru_coords.csv")
        slam_dir = os.path.join(BASE_DIR, f"SLAM-{species_code}")
        project_name = f"{species_code.lower()}_localization_05312025"
        root = os.path.join(slam_dir, project_name)

        # Check shelf exists
        if not os.path.exists(shelf_path + ".dat") and not os.path.exists(shelf_path + ".db"):
            # Try without extension
            try:
                with shelve.open(shelf_path, "r") as db:
                    pass
            except Exception:
                print(f"    Shelf not found: {shelf_path}. Skipping.")
                summary.append({"species": species_code, "events": 0, "status": "no shelf"})
                continue

        # Detect actual class_name
        target_class = get_target_class(species_code, shelf_path)
        print(f"    class_name on objects: '{target_class}'")

        # Load estimates
        with shelve.open(shelf_path, "r") as db:
            position_estimates = db["position_estimates"]
        print(f"    Raw estimates: {len(position_estimates)}")

        # Load ARU coords
        aru_coords = pd.read_csv(aru_coords_path, index_col=0)
        aru_utm = aru_coords.copy()
        aru_utm.rename(columns={"x": "utm_easting", "y": "utm_northing"}, inplace=True)

        # Filter
        target_estimates = [
            e for e in position_estimates
            if e.class_name == target_class
            and e.residual_rms < RMS_THRESHOLD
            and np.isfinite(e.location_estimate[0])
            and np.isfinite(e.location_estimate[1])
        ]
        print(f"    Pass filters: {len(target_estimates)}")

        # Group and average
        grouped = defaultdict(list)
        for e in target_estimates:
            grouped[e.start_timestamp].append(e)

        filtered_groups = {
            ts: evts for ts, evts in grouped.items()
            if len(evts) >= MIN_ESTIMATES_PER_WINDOW
        }
        print(f"    Time windows: {len(filtered_groups)}")

        if len(filtered_groups) == 0:
            print(f"    No events passed filters. Skipping.")
            summary.append({"species": species_code, "events": 0, "status": "no events"})
            continue

        events = []
        for i, (timestamp, estimates) in enumerate(
            sorted(filtered_groups.items(), key=lambda kv: kv[0])
        ):
            coords = [
                (e.location_estimate[0], e.location_estimate[1])
                for e in estimates
                if np.isfinite(e.location_estimate[0]) and np.isfinite(e.location_estimate[1])
            ]
            if len(coords) < MIN_ESTIMATES_PER_WINDOW:
                continue

            x_avg = np.mean([c[0] for c in coords])
            y_avg = np.mean([c[1] for c in coords])
            mean_rms = np.mean([e.residual_rms for e in estimates])
            duration = estimates[0].duration

            all_file_ids = set()
            for e in estimates:
                all_file_ids.update(e.receiver_files)

            events.append({
                "event_id": f"{species_code}_{i:05d}",
                "label": species_code,
                "start_timestamp": timestamp.isoformat(),
                "duration": duration,
                "position_x": round(x_avg, 2),
                "position_y": round(y_avg, 2),
                "position_z": "",
                "utm_zone": UTM_ZONE,
                "file_ids": ";".join(sorted(all_file_ids)),
                "file_start_time_offsets": "",
                "mean_residual_rms": round(mean_rms, 2),
                "n_estimates": len(coords),
            })

        print(f"    Events to write: {len(events)}")

        # Create directories
        for d in [root, os.path.join(root, "script"),
                  os.path.join(root, "localization_metadata"),
                  os.path.join(root, "observed_events"),
                  os.path.join(root, "audio")]:
            os.makedirs(d, exist_ok=True)

        # Write localized_events.csv
        write_csv(
            os.path.join(root, "localized_events.csv"),
            ["event_id", "label", "start_timestamp", "duration",
             "position_x", "position_y", "position_z", "utm_zone",
             "file_ids", "file_start_time_offsets",
             "mean_residual_rms", "n_estimates"],
            events,
        )

        # Write classes.csv
        common_name, sci_name = SPECIES_INFO.get(species_code, (species_code, ""))
        write_csv(
            os.path.join(root, "classes.csv"),
            ["class", "species", "scientific_name", "vocalization_type", "description"],
            [{"class": species_code, "species": common_name,
              "scientific_name": sci_name,
              "vocalization_type": "song", "description": ""}],
        )

        # Write point_table.csv
        pt_rows = []
        for idx, row in aru_utm.iterrows():
            pt_rows.append({
                "point_id": idx,
                "utm_easting": round(row["utm_easting"], 3),
                "utm_northing": round(row["utm_northing"], 3),
                "elevation": "",
                "utm_zone": UTM_ZONE,
            })
        write_csv(
            os.path.join(root, "localization_metadata", "point_table.csv"),
            ["point_id", "utm_easting", "utm_northing", "elevation", "utm_zone"],
            pt_rows,
        )

        # Write placeholders
        write_csv(
            os.path.join(root, "localization_metadata", "audio_file_table.csv"),
            ["file_id", "relative_path", "point_id", "start_timestamp"], [],
        )
        write_csv(
            os.path.join(root, "observed_events", "playbacks.csv"),
            ["playback_id", "class_label", "start_timestamp", "duration",
             "position_x", "position_y", "position_z", "utm_zone"], [],
        )
        write_csv(
            os.path.join(root, "observed_events", "observations.csv"),
            ["observed_event_id", "class_label", "start_timestamp", "duration",
             "position_x", "position_y", "position_z", "utm_zone",
             "direction", "comments"], [],
        )

        # environment.yml
        with open(os.path.join(root, "script", "environment.yml"), "w") as f:
            f.write(f"name: slam_{species_code.lower()}\n")
            f.write("dependencies:\n  - python>=3.9\n  - opensoundscape\n")
            f.write("  - pandas\n  - numpy\n  - pyproj\n  - pytz\n")

        print(f"    SLAM written to: {root}")
        summary.append({"species": species_code, "events": len(events), "status": "ok"})

    # Summary
    print(f"\n{'='*60}")
    print("  BATCH SLAM SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
