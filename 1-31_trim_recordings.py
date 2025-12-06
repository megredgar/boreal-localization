# -*- coding: utf-8 -*-
"""
Trim BAR-LT recordings to a common overlap window for localization.

- Uses the *resampled/synced* recordings (output of sync script)
- Finds an overlapping 600 s window across recorders
- Restricts that window to lie between 04:30 and 08:00 local time
- Trims each recorder to that window using frontierlabsutils.get_audio_from_time
- Saves trimmed files in the same naming convention as the original trim script:
    <recorder>_<original_filename>.wav

Requires:
- frontierlabsutils with:
    - extract_start_end
    - get_audio_from_time
- opensoundscape
"""

from pathlib import Path
from datetime import timedelta, time

import pandas as pd
from opensoundscape.audio import Audio

from frontierlabsutils import (
    extract_start_end,
    get_audio_from_time,
)

# ---------------------------------------------------------------------
# USER PARAMS
# ---------------------------------------------------------------------

# Folder with your synchronized/resampled recordings
RESAMPLED_DIR = Path(r"E:/BAR-LT_LocalizationProject/localizationresample")

# Where to save the trimmed clips
OUT_DIR = Path(r"E:/BAR-LT_LocalizationProject/localizationtrim_new")

# Date you want to trim (YYYYMMDD)
TARGET_DATE = "20250531"

# Length of desired clip (seconds)
CLIP_LENGTH_S = 600  # 10 minutes

# Minimum fraction of recorders that must have coverage for a candidate window
# 1.0 = all recorders; 0.9 = at least 90% of recorders, etc.
MIN_FRAC_RECORDERS = 1.0

# Optionally ignore some recorders you don't trust
EXCLUDE_RECORDERS = []  # e.g., ["L1N3E3"]

# Restrict the clip window to be between 04:30 and 08:00 local time
WINDOW_START_TIME = time(4, 30)   # 4:30 AM
WINDOW_END_TIME   = time(8, 0)    # 8:00 AM


# ---------------------------------------------------------------------
# LOAD ALL RESAMPLED FILES FOR THE TARGET DATE
# ---------------------------------------------------------------------

print(f"Scanning resampled recordings in: {RESAMPLED_DIR}")
rows = []

for wav in RESAMPLED_DIR.rglob("*.wav"):
    # recorder is top folder under RESAMPLED_DIR: e.g. L1N2E3
    recorder = wav.relative_to(RESAMPLED_DIR).parts[0]

    if recorder in EXCLUDE_RECORDERS:
        continue

    # Parse start/end datetimes from filename
    start_dt, end_dt = extract_start_end(wav.name)

    # Only keep rows from the target date
    if start_dt.strftime("%Y%m%d") != TARGET_DATE:
        continue

    rows.append(
        {
            "recorder": recorder,
            "path": wav,
            "start": start_dt,
            "end": end_dt,
        }
    )

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(f"No resampled WAVs found on date {TARGET_DATE} in {RESAMPLED_DIR}")

print(f"\nFound {len(df)} resampled file(s) on {TARGET_DATE}.")
all_recorders = sorted(df["recorder"].unique())
print("Recorders:", all_recorders)

# ---------------------------------------------------------------------
# PER-RECORDER COVERAGE + GLOBAL OVERLAP
# ---------------------------------------------------------------------

per_rec = (
    df.groupby("recorder")
      .agg(first_start=("start", "min"),
           last_end=("end", "max"))
)

print("\nPer-recorder coverage on this date:")
print(per_rec.to_string())

latest_start = per_rec["first_start"].max()
earliest_end = per_rec["last_end"].min()
full_overlap_s = (earliest_end - latest_start).total_seconds()

print("\n=== Global overlap across recorders ===")
print(f"  latest_start : {latest_start}")
print(f"  earliest_end : {earliest_end}")
print(f"  full_overlap : {full_overlap_s:.1f} s")

if full_overlap_s <= 0:
    raise RuntimeError(
        f"No global overlap across recorders on {TARGET_DATE}. "
        f"latest_start={latest_start}, earliest_end={earliest_end}, "
        f"delta_seconds={full_overlap_s}"
    )

# ---------------------------------------------------------------------
# BUILD CANDIDATE clip_start TIMES
#   - We use file start times for any file long enough to host CLIP_LENGTH_S
# ---------------------------------------------------------------------

candidate_times = set()

for _, row in df.iterrows():
    start_dt = row["start"]
    end_dt = row["end"]
    dur_s = (end_dt - start_dt).total_seconds()

    if dur_s >= CLIP_LENGTH_S:
        candidate_times.add(start_dt)

candidate_times = sorted(candidate_times)

if not candidate_times:
    raise RuntimeError(
        f"No single file in the dataset is at least {CLIP_LENGTH_S} s long on {TARGET_DATE}."
    )

print(f"\nInitial number of candidate clip_start times (from file starts): {len(candidate_times)}")

# ---------------------------------------------------------------------
# FILTER CANDIDATES TO 04:30–08:00 WINDOW
# ---------------------------------------------------------------------

def in_morning_window(clip_start, clip_len_s: int) -> bool:
    """
    Returns True if:
      - clip_start >= WINDOW_START_TIME (same date)
      - clip_end   <= WINDOW_END_TIME   (same date)
    """
    clip_end = clip_start + timedelta(seconds=clip_len_s)

    # Conditions at the level of wall clock time (ignoring date since it's constant)
    t_start = clip_start.timetz()
    t_end = clip_end.timetz()

    # start_ok: clip_start >= WINDOW_START_TIME
    start_ok = (
        (t_start.hour > WINDOW_START_TIME.hour)
        or (t_start.hour == WINDOW_START_TIME.hour and t_start.minute >= WINDOW_START_TIME.minute)
    )

    # end_ok: clip_end <= WINDOW_END_TIME (allowing 00 seconds exactly)
    end_ok = (
        (t_end.hour < WINDOW_END_TIME.hour)
        or (
            t_end.hour == WINDOW_END_TIME.hour
            and t_end.minute == WINDOW_END_TIME.minute
            and t_end.second == 0
        )
    )

    return start_ok and end_ok


candidate_times = [t for t in candidate_times if in_morning_window(t, CLIP_LENGTH_S)]

if not candidate_times:
    raise RuntimeError(
        f"No {CLIP_LENGTH_S}-s window between "
        f"{WINDOW_START_TIME.strftime('%H:%M')} and {WINDOW_END_TIME.strftime('%H:%M')} "
        f"with any candidate file starts."
    )

print(
    f"Number of candidate clip_start times within "
    f"{WINDOW_START_TIME.strftime('%H:%M')}–{WINDOW_END_TIME.strftime('%H:%M')}: "
    f"{len(candidate_times)}"
)

# ---------------------------------------------------------------------
# FOR EACH CANDIDATE, CHECK HOW MANY RECORDERS HAVE COVERAGE
# ---------------------------------------------------------------------

def fraction_recorders_covering(clip_start: pd.Timestamp, clip_len_s: int) -> float:
    clip_end = clip_start + timedelta(seconds=clip_len_s)
    n_rec = 0

    for rec in all_recorders:
        df_rec = df[df["recorder"] == rec]

        # Recorder has coverage if ANY of its files fully contains [clip_start, clip_end]
        has_coverage = ((df_rec["start"] <= clip_start) & (df_rec["end"] >= clip_end)).any()

        if has_coverage:
            n_rec += 1

    return n_rec / len(all_recorders)


best_clip_start = None
best_frac = -1.0

for cand in candidate_times:
    frac = fraction_recorders_covering(cand, CLIP_LENGTH_S)
    if frac >= MIN_FRAC_RECORDERS and frac > best_frac:
        best_frac = frac
        best_clip_start = cand

if best_clip_start is None:
    raise RuntimeError(
        f"No {CLIP_LENGTH_S}-s window between "
        f"{WINDOW_START_TIME.strftime('%H:%M')} and {WINDOW_END_TIME.strftime('%H:%M')} "
        f"has at least MIN_FRAC_RECORDERS={MIN_FRAC_RECORDERS:.2f} coverage."
    )

clip_start = best_clip_start
clip_end = clip_start + timedelta(seconds=CLIP_LENGTH_S)

print("\n=== Chosen trimming window ===")
print(f"  clip_start : {clip_start}")
print(f"  clip_end   : {clip_end}")
print(f"  clip_len   : {CLIP_LENGTH_S} s")
print(f"  fraction of recorders covering this window: {best_frac:.3f}")

# ---------------------------------------------------------------------
# TRIM FILES FOR EACH RECORDER
# ---------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

n_trimmed = 0
n_skipped = 0

for rec in all_recorders:
    df_rec = df[df["recorder"] == rec].copy()

    # Find a file in this recorder that fully covers [clip_start, clip_end]
    mask = (df_rec["start"] <= clip_start) & (df_rec["end"] >= clip_end)
    if not mask.any():
        print(
            f"[WARN] No single file for {rec} fully covers "
            f"[{clip_start} to {clip_end}]. Skipping this recorder."
        )
        n_skipped += 1
        continue

    row = df_rec[mask].iloc[0]
    wav_path = row["path"]
    original_start, _ = extract_start_end(wav_path.name)

    print(f"Trimming recorder {rec} from file {wav_path.name}")

    # Load audio
    audio = Audio.from_file(wav_path)

    # Use helper to pull the correct segment
    trimmed_audio = get_audio_from_time(
        clip_start=clip_start,
        clip_length_s=CLIP_LENGTH_S,
        original_start=original_start,
        original_audio=audio,
    )

    # Naming convention: same as original trim script:
    #   <recorder>_<original_filename>.wav
    new_name = f"{rec}_{wav_path.name}"
    out_path = OUT_DIR / new_name

    trimmed_audio.save(out_path)
    print(f"  Saved trimmed audio: {out_path}")
    n_trimmed += 1

print("\n=== Done ===")
print(f"Trimmed files written: {n_trimmed}")
print(f"Recorders skipped (no coverage for chosen window): {n_skipped}")
