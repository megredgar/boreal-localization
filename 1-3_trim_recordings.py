# -*- coding: utf-8 -*-
"""
Trim a common overlapping window from resampled BAR-LT recordings,
in a way that avoids datetime assertion issues.

- Reads all *_resampled.wav under RESAMPLED_DIR
- Restricts to TARGET_DATE
- Optionally drops some recorders (e.g. L1N3E3)
- Computes per-recorder min(start) / max(end)
- Chooses a clip window within the global overlap
- For each recorder, finds a single file that covers the whole window
- Trims by seconds relative to that file using Audio.trim()

Output naming matches original trim script:
    <recorder>_<original_resampled_filename>

Author: adapted for Megan, Dec 2025
"""

from pathlib import Path
import pandas as pd
from opensoundscape.audio import Audio
from frontierlabsutils import extract_start_end

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Folder with synced/resampled recordings
RESAMPLED_DIR = Path(r"E:/BAR-LT_LocalizationProject/localizationresample")

# Folder to write trimmed clips
OUT_DIR = Path(r"E:/BAR-LT_LocalizationProject/localizationtrim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Date (YYYYMMDD)
TARGET_DATE = "20250531"

# Drop problematic recorder(s) if desired
# e.g., L1N3E3 is the oddball; you can set [] to keep all
DROP_RECORDERS = ["L1N3E3"]

# Length of clip in seconds
CLIP_LENGTH_S = 600           # 10 minutes

# Offset from start of global overlap, in seconds (0 = start at latest_start)
CLIP_OFFSET_S = 0


# ---------------------------------------------------------------------
# STEP 1: READ ALL RESAMPLED FILES FOR TARGET DATE
# ---------------------------------------------------------------------

rows = []
for wav in RESAMPLED_DIR.rglob("*.wav"):
    recorder = wav.relative_to(RESAMPLED_DIR).parts[0]
    start_dt, end_dt = extract_start_end(wav.name)

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

# Optionally drop some recorders
if DROP_RECORDERS:
    df = df[~df["recorder"].isin(DROP_RECORDERS)].copy()

print(f"Using {df['recorder'].nunique()} recorder(s) on {TARGET_DATE}")
print(sorted(df["recorder"].unique()))

# ---------------------------------------------------------------------
# STEP 2: PER-RECORDER ENVELOPES AND GLOBAL OVERLAP
# ---------------------------------------------------------------------

per_rec = (
    df.groupby("recorder")
      .agg(first_start=("start", "min"),
           last_end=("end", "max"))
)

print("\nPer-recorder coverage (kept recorders):")
print(per_rec.to_string())

latest_start = per_rec["first_start"].max()
earliest_end = per_rec["last_end"].min()
full_overlap_s = (earliest_end - latest_start).total_seconds()

print("\n=== Global overlap across kept recorders ===")
print("  latest_start :", latest_start)
print("  earliest_end :", earliest_end)
print(f"  full_overlap : {full_overlap_s:.1f} s")

if full_overlap_s <= 0:
    raise RuntimeError(
        f"No shared overlap across kept recorders on {TARGET_DATE}.\n"
        f"latest_start={latest_start}, earliest_end={earliest_end}, "
        f"delta_seconds={full_overlap_s}"
    )

# Choose clip window
if CLIP_LENGTH_S > full_overlap_s:
    raise ValueError(
        f"Requested CLIP_LENGTH_S={CLIP_LENGTH_S} > full_overlap={full_overlap_s:.1f} s"
    )

clip_start = latest_start + pd.to_timedelta(CLIP_OFFSET_S, unit="s")
clip_end = clip_start + pd.to_timedelta(CLIP_LENGTH_S, unit="s")

if clip_end > earliest_end:
    raise ValueError(
        f"Clip [{clip_start} to {clip_end}] exceeds overlap "
        f"[{latest_start} to {earliest_end}]"
    )

print("\n=== Chosen trimming window ===")
print("  clip_start :", clip_start)
print("  clip_end   :", clip_end)
print(f"  clip_len   : {CLIP_LENGTH_S} s")

# ---------------------------------------------------------------------
# STEP 3: FIND FILE COVERING WINDOW FOR EACH RECORDER & TRIM
# ---------------------------------------------------------------------

n_trimmed = 0
skipped_recorders = []

for rec in sorted(per_rec.index):
    df_r = df[df["recorder"] == rec].copy()

    # Find any one file that fully covers the window
    candidates = df_r[(df_r["start"] <= clip_start) & (df_r["end"] >= clip_end)]

    if candidates.empty:
        print(f"[WARN] No single file for {rec} fully covers [{clip_start} to {clip_end}]. Skipping.")
        skipped_recorders.append(rec)
        continue

    row = candidates.sort_values("start").iloc[0]
    wav_path = row["path"]
    original_start = row["start"]

    # 🔹 Matching the original naming convention:
    #     <recorder>_<original_resampled_filename>
    out_name = f"{rec}_{wav_path.name}"
    out_path = OUT_DIR / out_name

    if out_path.exists():
        print(f"[SKIP] {out_path.name} already exists")
        continue

    print(f"Trimming recorder {rec} from {wav_path.name} to {out_name}")

    audio = Audio.from_file(wav_path)

    # Seconds from the start of THIS file to clip_start
    offset_s = (clip_start - original_start).total_seconds()
    start_s = max(0, offset_s)
    end_s = start_s + CLIP_LENGTH_S

    # Trim directly by seconds (no get_audio_from_time, so no assertion)
    trimmed = audio.trim(start_s, end_s)
    trimmed.save(out_path)

    n_trimmed += 1

print(f"\nFinished trimming. Wrote {n_trimmed} clip(s) to {OUT_DIR}")

if skipped_recorders:
    print("Recorders skipped (no single covering file):")
    print(", ".join(skipped_recorders))
