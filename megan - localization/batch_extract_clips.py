#!/usr/bin/env python3
"""
Batch extract 3-second audio clips for all localized species events.

For each species, loads the confirmed shelf, reproduces the same filtering
as batch_export_slam.py, and extracts clips from the original recordings.
"""

import os
import re
import shelve
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from opensoundscape.audio import Audio

# =============================================================================
# CONFIGURATION — must match batch_export_slam.py
# =============================================================================

BASE_DIR = r"D:/BARLT Localization Project/localization_05312025"

SPECIES_LIST = ["RCKI", "TEWA", "RWBL", "CHSP", "YEWA", "WTSP", "AMRO", "YBFL", "AMRE"]

RMS_THRESHOLD = 50
MIN_ESTIMATES_PER_WINDOW = 1
CLIP_DURATION = 3


# =============================================================================
# HELPERS
# =============================================================================

def parse_file_start_time(filepath):
    """Extract recording start datetime from filename."""
    fname = os.path.basename(filepath)
    match = re.search(r"_S(\d{8}T\d{6}\.\d+)([+-]\d{4})_", fname)
    if not match:
        raise ValueError(f"Cannot parse start time from filename: {fname}")

    dt_str = match.group(1)
    tz_str = match.group(2)

    tz_hours = int(tz_str[:3])
    tz_mins = int(tz_str[0] + tz_str[3:5])
    tz_off = timezone(timedelta(hours=tz_hours, minutes=tz_mins))

    dt = datetime.strptime(dt_str, "%Y%m%dT%H%M%S.%f").replace(tzinfo=tz_off)
    return dt


def get_site_name(filepath):
    """Extract ARU site name (first 6 chars of filename)."""
    return os.path.basename(filepath)[:6]


def get_target_class(shelf_path):
    """Check what class_name is on the position objects."""
    with shelve.open(shelf_path, "r") as db:
        pe = db["position_estimates"]
    if len(pe) == 0:
        return None
    return pe[0].class_name


# =============================================================================
# MAIN
# =============================================================================

def main():
    summary = []

    for species_code in SPECIES_LIST:
        print(f"\n{'='*60}")
        print(f"  Extract clips: {species_code}")
        print(f"{'='*60}")

        species_dir = os.path.join(BASE_DIR, f"hawkears_0_7_{species_code}")
        shelf_path = os.path.join(species_dir, "pythonoutput", f"{species_code.lower()}_confirmed.out")
        slam_dir = os.path.join(BASE_DIR, f"SLAM-{species_code}")
        project_name = f"{species_code.lower()}_localization_05312025"
        slam_root = os.path.join(slam_dir, project_name)

        # Check shelf exists
        try:
            with shelve.open(shelf_path, "r") as db:
                position_estimates = db["position_estimates"]
        except Exception:
            print(f"    Shelf not found: {shelf_path}. Skipping.")
            summary.append({"species": species_code, "clips": 0, "status": "no shelf"})
            continue

        # Detect class_name
        target_class = get_target_class(shelf_path)
        if target_class is None:
            print(f"    Empty shelf. Skipping.")
            summary.append({"species": species_code, "clips": 0, "status": "empty shelf"})
            continue

        print(f"    class_name: '{target_class}', estimates: {len(position_estimates)}")

        # Filter
        filtered = [
            e for e in position_estimates
            if e.class_name == target_class
            and e.residual_rms < RMS_THRESHOLD
            and np.isfinite(e.location_estimate[0])
            and np.isfinite(e.location_estimate[1])
        ]
        print(f"    Pass filters: {len(filtered)}")

        # Group
        grouped = defaultdict(list)
        for e in filtered:
            grouped[e.start_timestamp].append(e)

        filtered_groups = {
            ts: evts for ts, evts in grouped.items()
            if len(evts) >= MIN_ESTIMATES_PER_WINDOW
        }
        print(f"    Time windows: {len(filtered_groups)}")

        if len(filtered_groups) == 0:
            print(f"    No events. Skipping.")
            summary.append({"species": species_code, "clips": 0, "status": "no events"})
            continue

        # Check SLAM directory exists
        events_path = os.path.join(slam_root, "localized_events.csv")
        if not os.path.exists(events_path):
            print(f"    SLAM not found at {slam_root}. Run batch_export_slam.py first.")
            summary.append({"species": species_code, "clips": 0, "status": "no SLAM"})
            continue

        audio_dir = os.path.join(slam_root, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Extract clips
        audio_file_rows = []
        event_file_map = {}
        n_clips = 0
        n_errors = 0

        for i, (timestamp, estimates) in enumerate(
            sorted(filtered_groups.items(), key=lambda kv: kv[0])
        ):
            event_id = f"{species_code}_{i:05d}"

            all_files = set()
            for e in estimates:
                all_files.update(e.receiver_files)

            event_file_ids = []
            event_offsets = []

            for filepath in sorted(all_files):
                site = get_site_name(filepath)

                try:
                    file_start = parse_file_start_time(filepath)
                except ValueError as ex:
                    n_errors += 1
                    continue

                offset_sec = (timestamp - file_start).total_seconds()
                if offset_sec < 0:
                    continue

                clip_fname = f"{event_id}_{site}.flac"
                rel_path = os.path.join("audio", site, clip_fname)
                out_path = os.path.join(slam_root, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                try:
                    audio = Audio.from_file(
                        filepath, offset=offset_sec, duration=CLIP_DURATION,
                    )
                    audio.save(out_path)
                    n_clips += 1
                except Exception as ex:
                    print(f"      ERROR {clip_fname}: {ex}")
                    n_errors += 1
                    continue

                file_id = clip_fname
                event_file_ids.append(file_id)
                event_offsets.append(round(offset_sec, 6))

                audio_file_rows.append({
                    "file_id": file_id,
                    "relative_path": rel_path,
                    "point_id": site,
                    "start_timestamp": timestamp.isoformat(),
                })

            event_file_map[event_id] = {
                "file_ids": event_file_ids,
                "offsets": event_offsets,
            }

        print(f"    Clips: {n_clips}, Errors: {n_errors}")

        # Write audio_file_table.csv
        aft_path = os.path.join(slam_root, "localization_metadata", "audio_file_table.csv")
        pd.DataFrame(audio_file_rows).to_csv(aft_path, index=False)

        # Update localized_events.csv
        events_df = pd.read_csv(events_path)
        for idx, row in events_df.iterrows():
            eid = row["event_id"]
            if eid in event_file_map:
                info = event_file_map[eid]
                events_df.at[idx, "file_ids"] = ";".join(info["file_ids"])
                events_df.at[idx, "file_start_time_offsets"] = ";".join(
                    str(o) for o in info["offsets"]
                )
        events_df.to_csv(events_path, index=False)

        print(f"    Updated audio_file_table.csv and localized_events.csv")
        summary.append({"species": species_code, "clips": n_clips, "status": "ok"})

    # Summary
    print(f"\n{'='*60}")
    print("  BATCH CLIP EXTRACTION SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
