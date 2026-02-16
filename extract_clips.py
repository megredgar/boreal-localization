#!/usr/bin/env python3
"""
Extract 3-second audio clips for each localized CONW event.

For each averaged event in the SLAM export, this script:
  1. Identifies which receiver files contributed to that time window
  2. Calculates the offset into each recording
  3. Extracts a 3-second clip
  4. Saves clips organized by ARU site (first 6 chars of filename)
  5. Writes audio_file_table.csv linking clips to ARU positions
  6. Updates localized_events.csv with correct file_ids and file_start_time_offsets

Assumes position_estimates are already filtered and grouped the same way
as export_slam_CONW.py (same RMS_THRESHOLD, MIN_ESTIMATES_PER_WINDOW, etc.).

Usage:
    python extract_clips_CONW.py

Requirements:
    pip install opensoundscape numpy pandas
"""

import os
import shelve
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from opensoundscape.audio import Audio

# =============================================================================
# CONFIGURATION — must match export_slam_CONW.py settings
# =============================================================================

SHELF_PATH = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/pythonoutput/conw_confirmed.out"

# SLAM output root (same as export_slam_CONW.py OUTPUT_DIR/PROJECT_NAME)
SLAM_ROOT = r"D:/BARLT Localization Project/localization_05312025/SLAM-CONW/conw_localization_05312025"

# Filtering — keep identical to export_slam_CONW.py
RMS_THRESHOLD = 50
MIN_ESTIMATES_PER_WINDOW = 1
TARGET_CLASS = "index"      # class_name on CONW position objects (index bug from reset_index)

# Clip duration in seconds (matches detection window)
CLIP_DURATION = 3


# =============================================================================
# Helpers
# =============================================================================

def parse_file_start_time(filepath):
    """
    Extract recording start datetime from filename.

    Expected pattern: ..._SYYYYMMDDTHHMMSS.ffffff-HHMM_...
    e.g. L1N6E7_S20250531T052148.068213-0500_E...
    """
    fname = os.path.basename(filepath)
    match = re.search(r"_S(\d{8}T\d{6}\.\d+)([+-]\d{4})_", fname)
    if not match:
        raise ValueError(f"Cannot parse start time from filename: {fname}")

    dt_str = match.group(1)      # 20250531T052148.068213
    tz_str = match.group(2)      # -0500

    tz_hours = int(tz_str[:3])   # -05
    tz_mins = int(tz_str[0] + tz_str[3:5])  # -00
    tz_off = timezone(timedelta(hours=tz_hours, minutes=tz_mins))

    dt = datetime.strptime(dt_str, "%Y%m%dT%H%M%S.%f").replace(tzinfo=tz_off)
    return dt


def get_site_name(filepath):
    """Extract ARU site name (first 6 chars of filename)."""
    return os.path.basename(filepath)[:6]


# =============================================================================
# Main
# =============================================================================

def main():
    audio_dir = os.path.join(SLAM_ROOT, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # ---- Load and filter estimates (same logic as export_slam_CONW.py) ------
    print(f"Loading position estimates from: {SHELF_PATH}")
    with shelve.open(SHELF_PATH, "r") as db:
        position_estimates = db["position_estimates"]
    print(f"  {len(position_estimates)} raw estimates")

    filtered = [
        e for e in position_estimates
        if e.class_name == TARGET_CLASS
        and e.residual_rms < RMS_THRESHOLD
        and np.isfinite(e.location_estimate[0])
        and np.isfinite(e.location_estimate[1])
    ]
    print(f"  {len(filtered)} pass RMS + finite filter")

    grouped = defaultdict(list)
    for e in filtered:
        grouped[e.start_timestamp].append(e)

    filtered_groups = {
        ts: events for ts, events in grouped.items()
        if len(events) >= MIN_ESTIMATES_PER_WINDOW
    }
    print(f"  {len(filtered_groups)} time windows\n")

    # ---- Extract clips and build file table --------------------------------
    audio_file_rows = []   # for audio_file_table.csv
    event_file_map = {}    # event_id -> {file_ids: [...], offsets: [...]}

    n_clips = 0
    n_errors = 0

    for i, (timestamp, estimates) in enumerate(
        sorted(filtered_groups.items(), key=lambda kv: kv[0])
    ):
        event_id = f"CONW_{i:05d}"

        # Unique receiver files across all estimates in this window
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
                print(f"  WARN: {ex}")
                n_errors += 1
                continue

            # Offset from recording start to event start
            offset_sec = (timestamp - file_start).total_seconds()

            if offset_sec < 0:
                # Event is before this file started — skip
                continue

            # Clip filename: eventID_site.flac
            clip_fname = f"{event_id}_{site}.flac"
            # Relative path from SLAM root for audio_file_table
            rel_path = os.path.join("audio", site, clip_fname)
            # Absolute output path
            out_path = os.path.join(SLAM_ROOT, rel_path)

            # Create site folder
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Extract clip
            try:
                audio = Audio.from_file(
                    filepath,
                    offset=offset_sec,
                    duration=CLIP_DURATION,
                )
                audio.save(out_path)
                n_clips += 1
            except Exception as ex:
                print(f"  ERROR extracting {clip_fname}: {ex}")
                n_errors += 1
                continue

            # Track for file table and event linkage
            file_id = clip_fname

            event_file_ids.append(file_id)
            event_offsets.append(round(offset_sec, 6))

            # Build audio_file_table row
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

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(filtered_groups)} events ({n_clips} clips)...")

    print(f"\nExtraction complete: {n_clips} clips, {n_errors} errors")

    # ---- Write audio_file_table.csv ----------------------------------------
    aft_path = os.path.join(SLAM_ROOT, "localization_metadata", "audio_file_table.csv")
    aft_df = pd.DataFrame(audio_file_rows)
    aft_df.to_csv(aft_path, index=False)
    print(f"Wrote {len(aft_df)} rows to audio_file_table.csv")

    # ---- Update localized_events.csv with file_ids and offsets -------------
    events_path = os.path.join(SLAM_ROOT, "localized_events.csv")
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
    print(f"Updated localized_events.csv with file_ids and offsets")

    # ---- Summary -----------------------------------------------------------
    sites = set(get_site_name(f) for rows in event_file_map.values()
                for f in rows["file_ids"])
    print(f"\nAudio organized into {len(sites)} site folders:")
    for site in sorted(sites):
        site_dir = os.path.join(audio_dir, site)
        if os.path.isdir(site_dir):
            count = len(os.listdir(site_dir))
            print(f"  {site}/  ({count} clips)")


if __name__ == "__main__":
    main()