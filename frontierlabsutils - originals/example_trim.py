"""
example_trim.py
Tessa Rhinehart
2024-11-11

This file takes resampled files (see example_sync.py) and trims  
them so that they all contain audio starting at the same time.

This is accomplished by finding the latest start time and
earliest end time for each synchronized recording from the 
same day and recording period, then trimming all recordings
from that recording period to that time.

This approach facilitates using localization pipelines that rely on 
automated detection algorithms. This way, detection outputs 
are synchronized.
"""

import pandas as pd
from pathlib import Path
from opensoundscape.audio import Audio

from frontierlabsutils import get_recording_path, get_recorder_list, get_all_times
from frontierlabsutils import get_latest_start_second, get_earliest_end_second
from frontierlabsutils import extract_start_end, get_audio_from_time

data_dir =  "/my/recording/directory/resampled_recordings" # Directory containing resampled files
out_dir =  "/my/recording/directory/trimmed_recordings" # Directory to save trimmed recordings

# Make the trimmed recording directory
trimmed_recording_dir = Path(out_dir)
trimmed_recording_dir.mkdir(exist_ok=True)

# The list of recorders is specific to my deployment;
# you will have to rewrite this function for your own
recorders = get_recorder_list()
date = '2023-06-28'

# These are the recording periods used
earliest_start_times = [
    "0358", 
    "0428", "0458", 
    "0528", "0558",
    "0628", "0658",
    "0728", "0758", 
    "0828", "0858",
    "0928", "0958",
]

for start_time in earliest_start_times:
    try_times = get_all_times(start_time)[:3]
    recordings_this_time = []
    recorders_this_time = []
    for recorder in recorders:
        for time in try_times:
            recording = get_recording_path(
                recorder=recorder,
                date=date,
                hour_minute=time,
                data_dir=data_dir
            )
            if recording:
                recorders_this_time.append(recorder)
                recordings_this_time.append(Path(recording))
                break

    # Find the latest start time of them
    latest_start_second = get_latest_start_second(recordings_this_time)

    # Find the earliest end time of them
    earliest_end_second = get_earliest_end_second(recordings_this_time)

    # Trim all recordings from the earliest time to the end
    for recorder, recording in zip(recorders_this_time, recordings_this_time):
        recorder_dir = trimmed_recording_dir.joinpath(recorder)
        recorder_dir.mkdir(exist_ok=True)
        trimmed_audio_filename = recorder_dir.joinpath(recording.name)
        if trimmed_audio_filename.exists():
            continue
            
        audio = Audio.from_file(recording)
        
        original_start, original_end = extract_start_end(recording.name)
        
        clip_len = (earliest_end_second - latest_start_second).seconds
        
        trimmed_audio = get_audio_from_time(
            clip_start = latest_start_second,
            clip_length_s = clip_len,
            original_start = original_start,
            original_audio = audio
        )
        trimmed_audio.save(trimmed_audio_filename)
