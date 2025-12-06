from pathlib import Path
from datetime import datetime
import re
import pandas as pd

from frontierlabsutils import extract_start_end  # same helper used in your sync script

# Point at the ORIGINAL raw recordings, not the resample folder
data_dir = Path(r"D:/BBMP/2025/ARU - Breeding Season 2025/Localization")

rows = []
bad_files = []

for wav in data_dir.rglob("*.wav"):
    path_str = str(wav)

    # Try to grab recorder code like L1N2E3 from the path
    m_site = re.search(r"L\d+N\d+E\d+", path_str)
    recorder = m_site.group(0) if m_site else wav.parent.name

    # Robustly parse start/end using your existing helper
    try:
        start_dt, end_dt = extract_start_end(wav.name)
    except Exception as e:
        print(f"⚠️ Skipping {wav.name}: extract_start_end() failed ({e})")
        bad_files.append({
            "recorder": recorder,
            "filename": wav.name,
            "path": path_str,
            "error": str(e),
        })
        continue

    rows.append({
        "recorder": recorder,
        "filename": wav.name,
        "path": path_str,
        "start": start_dt,
        "end": end_dt,
        "duration_s": (end_dt - start_dt).total_seconds(),
        "date": start_dt.date(),
        "start_time": start_dt.time(),
        "end_time": end_dt.time(),
    })

df = pd.DataFrame(rows)
bad_df = pd.DataFrame(bad_files)

print("✅ Parsed files:", len(df))
print("⚠️ Skipped (bad name/parse):", len(bad_df))
print(df.head())



# Sanity: how many recorders do we see?
all_recorders = sorted(df["recorder"].unique())
print(f"Total unique recorders:", len(all_recorders))
print(all_recorders)

rows_summary = []

for d in sorted(df["date"].unique()):
    df_day = df[df["date"] == d].copy()
    day_recorders = sorted(df_day["recorder"].unique())

    if len(day_recorders) < 2:
        latest_start = pd.NaT
        earliest_end = pd.NaT
        overlap_seconds = 0
        has_overlap = False
    else:
        latest_start = df_day["start"].max()
        earliest_end = df_day["end"].min()
        overlap_seconds = (earliest_end - latest_start).total_seconds()
        has_overlap = overlap_seconds > 0

    rows_summary.append({
        "date": d,
        "n_recorders_that_day": len(day_recorders),
        "recorders_that_day": ", ".join(day_recorders),
        "latest_start": latest_start,
        "earliest_end": earliest_end,
        "overlap_seconds": overlap_seconds,
        "has_overlap": has_overlap,
        "includes_all_recorders": set(day_recorders) == set(all_recorders),
    })

summary = pd.DataFrame(rows_summary)

print("\n=== Date-wise overlap summary (original recordings) ===")
print(
    summary[
        ["date", "n_recorders_that_day", "overlap_seconds",
         "has_overlap", "includes_all_recorders"]
    ].to_string(index=False)
)


import pandas as pd

# Make sure df exists from the earlier step
# columns: recorder, filename, path, start, end, duration_s, date, start_time, end_time

# Build per-date, per-recorder ranges
per_day_rec = (
    df.groupby(["date", "recorder"])
      .agg(
          first_start=("start", "min"),
          last_end=("end", "max")
      )
      .assign(duration_s=lambda x: (x["last_end"] - x["first_start"]).dt.total_seconds())
      .reset_index()
)

print("Per-day/per-recorder summary built with", len(per_day_rec), "rows.")


from datetime import date as Date

focus_date = Date(2025, 5, 31)   # change if you want a different day

day_table = (
    per_day_rec[per_day_rec["date"] == focus_date]
    .sort_values("first_start")
)

print(f"\n=== Recorder ranges on {focus_date} ===")
print(day_table.to_string(index=False))


def inspect_overlap_for_date(per_day_rec, focus_date):
    df_day = per_day_rec[per_day_rec["date"] == focus_date].copy()
    if df_day.empty:
        print(f"No recordings found on {focus_date}")
        return

    latest_start = df_day["first_start"].max()
    earliest_end = df_day["last_end"].min()
    overlap = (earliest_end - latest_start).total_seconds()

    print(f"\n=== Global overlap for {focus_date} ===")
    print("Latest start across recorders :", latest_start)
    print("Earliest end across recorders :", earliest_end)
    print("Overlap (seconds)             :", overlap)

    # Who ends the earliest? Who starts the latest?
    earliest_end_row = df_day.loc[df_day["last_end"].idxmin()]
    latest_start_row = df_day.loc[df_day["first_start"].idxmax()]

    print("\nRecorder with EARLIEST END:")
    print(earliest_end_row.to_string())

    print("\nRecorder with LATEST START:")
    print(latest_start_row.to_string())

    # Optional: what if we ignore each recorder one at a time?
    print("\nIf you drop each recorder one by one, what’s the overlap?")
    results = []
    for rec in df_day["recorder"].unique():
        keep = df_day[df_day["recorder"] != rec]
        ls = keep["first_start"].max()
        ee = keep["last_end"].min()
        ov = (ee - ls).total_seconds()
        results.append((rec, ov, ls, ee))

    res_df = pd.DataFrame(results, columns=["dropped_recorder", "overlap_s", "latest_start_keep", "earliest_end_keep"])
    # sort to see which drop gives you positive/maximum overlap
    res_df = res_df.sort_values("overlap_s", ascending=False)
    print(res_df.to_string(index=False))


from datetime import date as Date

inspect_overlap_for_date(per_day_rec, Date(2025, 5, 31))










from datetime import timedelta

rows_best = []

for d in sorted(per_day_rec["date"].unique()):
    df_day = per_day_rec[per_day_rec["date"] == d].copy()
    if df_day["recorder"].nunique() < 2:
        continue

    # baseline (keep all)
    base_ls = df_day["first_start"].max()
    base_ee = df_day["last_end"].min()
    base_ov = (base_ee - base_ls).total_seconds()

    # best after dropping one
    best_ov = base_ov
    best_drop = None

    for rec in df_day["recorder"].unique():
        keep = df_day[df_day["recorder"] != rec]
        ls = keep["first_start"].max()
        ee = keep["last_end"].min()
        ov = (ee - ls).total_seconds()
        if ov > best_ov:
            best_ov = ov
            best_drop = rec

    rows_best.append({
        "date": d,
        "baseline_overlap_s": base_ov,
        "best_overlap_s_dropping_one": best_ov,
        "recorder_to_drop": best_drop,
    })

best_df = pd.DataFrame(rows_best)
print(best_df.to_string(index=False))




from pathlib import Path
import pandas as pd
from frontierlabsutils import extract_start_end

RESAMPLED_DIR = Path(r"E:/BAR-LT_LocalizationProject/localizationresample")
DATE_STR = "20250531"

rows = []
for wav in RESAMPLED_DIR.rglob("*.wav"):
    recorder = wav.relative_to(RESAMPLED_DIR).parts[0]
    start_dt, end_dt = extract_start_end(wav.name)
    if start_dt.strftime("%Y%m%d") != DATE_STR:
        continue
    rows.append({"recorder": recorder, "start": start_dt, "end": end_dt})

df = pd.DataFrame(rows)
print("N rows on that date:", len(df))
print("Recorders:", sorted(df["recorder"].unique()))

per_rec = (
    df.groupby("recorder")
      .agg(first_start=("start", "min"),
           last_end=("end", "max"))
)
print(per_rec.to_string())

latest_start = per_rec["first_start"].max()
earliest_end = per_rec["last_end"].min()
overlap_s = (earliest_end - latest_start).total_seconds()

print("\nResampled per-recorder overlap for 2025-05-31:")
print("  latest_start :", latest_start)
print("  earliest_end :", earliest_end)
print("  overlap_s    :", overlap_s)
