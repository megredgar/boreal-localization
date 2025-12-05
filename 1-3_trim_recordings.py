# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:36:54 2025

@author: AlexE

#This script provides code for trimming recordings that have been synchronized (see sync_loop.py) so that they all have the same start time
#The code was modified from the example_trim script developed by Tessa Rhinehart (https://github.com/rhine3/frontierlabsutils)

"""

import pandas as pd
from pathlib import Path
from opensoundscape.audio import Audio

from frontierlabsutils import (
    get_recorder_list,
    get_latest_start_second,
    get_earliest_end_second,
    extract_start_end,
    get_audio_from_time,
)

# Input/output directories
data_dir = "C:/Users/AlexE/OneDrive - EC-EC/Robinson,Barry (il _ he, him) (ECCC)'s files - Grassland Bird Monitoring/R Projects/Localization/recordings/cclo_resample"
out_dir = "C:/Users/AlexE/OneDrive - EC-EC/Robinson,Barry (il _ he, him) (ECCC)'s files - Grassland Bird Monitoring/R Projects/Localization/recordings/cclo_trim"

# Create output directory if it doesn't exist
trimmed_recording_dir = Path(out_dir)
trimmed_recording_dir.mkdir(exist_ok=True)

# Recorder list and date of interest
recorders = get_recorder_list()
date = "20250531"

# Store found recordings and their recorder names
recordings_this_time = []
recorders_this_time = []

# Loop over all recorders and try to grab the one recording for this date
for recorder in recorders:
    recorder_dir = Path(data_dir) / recorder
    if not recorder_dir.exists():
        continue

    wav_files = list(recorder_dir.glob(f"*{date}*.wav"))
    if not wav_files:
        print(f"No recordings found for {recorder} on {date}")
        continue

    recordings_this_time.append(wav_files[0])
    recorders_this_time.append(recorder)

# If nothing found, skip the rest
if not recordings_this_time:
    print(f"No recordings found for any recorder on {date}. Skipping.")
else:
    # Calculate latest start and earliest end across all recordings
    latest_start_second = get_latest_start_second(recordings_this_time)
    earliest_end_second = get_earliest_end_second(recordings_this_time)

    # Trim and save each file
    for recorder, recording in zip(recorders_this_time, recordings_this_time):
        recorder_out_dir = trimmed_recording_dir
        recorder_out_dir.mkdir(exist_ok=True)
        
        #Add recorder name as prefix
        new_filename = f"{recorder}_{recording.name}"
        trimmed_audio_filename = recorder_out_dir / new_filename

        if trimmed_audio_filename.exists():
            print(f"Already trimmed: {trimmed_audio_filename.name}")
            continue

        print(f"Trimming {recording.name}...")

        audio = Audio.from_file(recording)
        original_start, original_end = extract_start_end(recording.name)

        clip_len = (earliest_end_second - latest_start_second).seconds

        trimmed_audio = get_audio_from_time(
            clip_start=latest_start_second,
            clip_length_s=clip_len,
            original_start=original_start,
            original_audio=audio
        )

        trimmed_audio.save(trimmed_audio_filename)
        print(f"Saved trimmed audio: {trimmed_audio_filename}")

