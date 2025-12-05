# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:00:23 2025, adapted Dec 12 2015 by MR Edgar

@author: AlexE

Important: modify and run the frontierlabsutils.py script first
NOTE: Make sure that it's called frontierlabsutils.py and is in the same folder as this script

This script provides code for synchronizing BAR-LT recordings for localization which has been modified from the example_sync.py document,
created by Tessa Rhinehart (https://github.com/rhine3/frontierlabsutils). Note that I cherry picked recordings from a date/time and added them
to a separate folder to be processed. Tessa's original script pulls a recordings for defined dates and times from the wider dataset.

"""
from opensoundscape.audio import Audio
from pathlib import Path
import frontierlabsutils
print(frontierlabsutils.__file__)
from time import time



"""
need to check for duplicates in recorder list - some recorders have multiple files for same date/time
"""
from pathlib import Path
import re
import pandas as pd

# define as a Path, not a string
data_dir = Path(r"D:/BBMP/2025/ARU - Breeding Season 2025/Localization")

audio_paths = list(data_dir.rglob("*.wav"))
len(audio_paths)

rows = []

for p in audio_paths:
    path_str = str(p)

    # --- SITE: first thing like L1N1E7 anywhere in the path ---
    site_match = re.search(r"L\d+N\d+E\d+", path_str)
    if not site_match:
        # If this prints for many paths, we need to tweak the pattern.
        # print("No site code found in:", p)
        continue
    site = site_match.group(0)

    # --- DATE & SESSION ("0521") from S...T... in filename ---
    # e.g., S20250531T052113.1234-0500  -> date = 20250531, session = 0521
    dt_match = re.search(r"S(\d{8})T(\d{4})", path_str)
    date = None
    session = None
    if dt_match:
        date = dt_match.group(1)
        session = dt_match.group(2)
    else:
        # Fallback: 8-digit date right after site_ in some folder name:
        # e.g., L1N1E7_20250605_Localization...
        folder_match = re.search(rf"{site}_([0-9]{{8}})", path_str)
        if folder_match:
            date = folder_match.group(1)
        # session will stay None in this fallback case

    # --- GPS from the [ +lat -lon ] bit in the folder name ---
    # matches things like [+51.34199-96.95669]
    gps_match = re.search(r"\[([+-]\d+\.\d+)[^\d+-]*([+-]\d+\.\d+)\]", path_str)
    lat = lon = None
    if gps_match:
        lat = float(gps_match.group(1))
        lon = float(gps_match.group(2))

    rows.append(
        {
            "site": site,
            "date": date,
            "session": session,
            "lat": lat,
            "lon": lon,
            "path": str(p),
        }
    )

df = pd.DataFrame(rows)
print(f"Parsed {len(df)} audio files into the table.")
df.head()

df["site"].nunique()        # should be ~49
df[["site", "date"]].drop_duplicates().shape[0]  # how many unique site+date combos
df.sample(5)                # eyeball some rows

import matplotlib.pyplot as plt

sites = (
    df.dropna(subset=["lat", "lon"])
      .groupby("site", as_index=False)[["lat", "lon"]]
      .first()
)

print(f"Number of sites with coordinates: {len(sites)}")
sites.head()

plt.figure(figsize=(6, 6))
plt.scatter(sites["lon"], sites["lat"])

for _, row in sites.iterrows():
    plt.text(row["lon"], row["lat"], row["site"], fontsize=7)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Localization sites (quick check)")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

"""
ok so that should be done now and you can now start the sync recordings script 
"""



data_dir = "D:/BBMP/2025/ARU - Breeding Season 2025/Localization" # Where your original recordings and loclog files are stored
out_dir  = "C:/Users/EdgarM/Desktop/Localization/localizationresample" # Where you want the resampled recordings to be saved

# Define recorders, boreal naming conventions
recorders = frontierlabsutils.get_recorder_list()
# check that it's pulled recording files properly
print("Recorders:", recorders[:5], "...")

# Dates you want to synchronize recordings from
dates = ["20250531"]
#times = []

resampled_folder = Path(out_dir)

t0 = time()
num_resampled = 0

for recorder in recorders:
    print(f"\nProcessing recorder: {recorder}")

    # Make sure the output folder exists for this recorder
    recorder_folder = resampled_folder.joinpath(recorder)
    recorder_folder.mkdir(parents=True, exist_ok=True)

    for date in dates:
        print(f"\nResampling data from recorder {recorder}, date: {date}")

        # Get all recordings from this date and recorder
        rec_date_recording_paths = frontierlabsutils.get_recording_path(
            recorder=recorder, 
            date=date,
            hour_minute="any", 
            data_dir=data_dir
        )

        # Avoid dates with zero recordings found
        if type(rec_date_recording_paths) != list or len(rec_date_recording_paths) == 0:
            print(f"  No recordings found for {recorder} on {date}. Skipping.")
            continue
            
        print(f"{len(rec_date_recording_paths)} recordings found")

        # Get loclog contents
        loclog_values = frontierlabsutils.get_loclog_contents(
            data_dir=data_dir,
            recorder=recorder,
            date=date
        )
        
        if len(loclog_values) < 1:
            print("  No loclog values found for recorder. Continuing.")
            continue

        # Resample all recordings
        for recording_path in rec_date_recording_paths:
            print(f"Resampling {recording_path}")
            resampled_filename = recorder_folder.joinpath(f"{Path(recording_path).stem}_resampled.wav")
            start, end = frontierlabsutils.extract_start_end(Path(recording_path).name)

            if resampled_filename.exists():
                print("  Already resampled. Continuing.")
                continue

            try:
                recording_time = recording_path.name.split('.')[0].split('T')[1][:4]
            except IndexError:
                print(f"  Recording {recording_path} not in correct format. Skipping.")
                continue

            write_times = frontierlabsutils.get_recording_write_times(
                loclog_values, date=date, start_time=recording_time)
            if len(write_times) == 0:
                print(f"  No write times logged for {recorder}. Skipping.")
                continue

            overflow_sample_indices, overflow_samples_to_insert = frontierlabsutils.get_buffer_insert_lengths(write_times)            
            audio = frontierlabsutils.insert_missing_buffers(
                recorder=recorder,
                date=date,
                time=recording_time,
                data_dir=data_dir,
                overflow_sample_indices=overflow_sample_indices,
                overflow_samples_to_insert=overflow_samples_to_insert
            )

            resampled = frontierlabsutils.interpolate_audio(start, end, audio)
            resampled.save(resampled_filename)
            num_resampled += 1

t1 = time()
print(f"\nIt took {(t1-t0)/60:.2f} minutes to resample {num_resampled} files.")
