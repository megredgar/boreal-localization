"""
example_sync.py
Tessa Rhinehart
2024-11-11

This file shows an example of resampling recordings from a given date to time-synchronize them.

The pipeline for synchronizing files is as follows:
- Get all recordings from a given date and recorder
- Get the loclog.txt file for this recorder
- Fix buffer overflows/dropped buffers (adds 0s)
- Interpolate each recording based on its start/end timestamp
- Saves the audio in a new directory

Optionally, you can then trim the files to use all the same start time (see example_trim.py)

"""

from opensoundscape.audio import Audio
from pathlib import Path
import frontierlabsutils
from time import time

data_dir = "/my/recording/directory/raw_recordings" # Where your original recordings and loclog files are stored
out_dir = "/my/recording/directory/resampled_recordings" # Where to save your resampled recordings (created below)
recorder = "A1" # Should be a subdirectory within data_dir

# Dates you want to synchronize recordings from
dates = [
    "20230521",
    "20230522",
    "20230523",
    "20230524",
    "20230525",
    "20230526",
    "20230527",
    "20230528",
    "20230529",
    "20230530",
    "20230531",
    "20230601",
    "20230602",
    "20230603",
    "20230604",
    "20230605",
    "20230606",
    "20230607",
    "20230608",
    "20230609",
    "20230610",
    "20230611",
    "20230612",
    "20230613",
    "20230614",
    "20230615",
    "20230616",
    "20230617",
    "20230618",
    "20230619",
    "20230620",
    "20230621",
    "20230622",
    "20230623",
    "20230624",
    "20230625",
    "20230626",
    "20230627",
    "20230628",
    "20230629",
]


# Set up a recorder subfolder to save resampled files in
resampled_folder = Path(out_dir)
recorder_folder = resampled_folder.joinpath(recorder)
recorder_folder.mkdir(exist_ok=True)


t0 = time()
num_resampled = 0
for date in dates:
    print(f"\n\nResampling data from recorder {recorder}, date: {date}")
    # Get all recordings from this date and recorder
    rec_date_recording_paths = frontierlabsutils.get_recording_path(
        recorder=recorder, 
        date=date,
        hour_minute="any", 
        data_dir=data_dir,
        old_style=False,
    )
    
    # Avoid dates with zero recordings found for them (e.g. recordings on old firmware)
    if type(rec_date_recording_paths) != list:
        continue
        
    print(f"{len(rec_date_recording_paths)} recordings found")

    # Get all write_times for this date and recorder
    loclog_values = frontierlabsutils.get_loclog_contents(
        data_dir=data_dir,
        recorder=recorder,
        date=date,
        old_style=False
    )
    
    # Avoid files with missing loclogs
    if len(loclog_values) < 1:
        print("  No loclog values found for recorder. Continuing.")
        continue

    # Fix overflows and interpolate each recording
    for recording_path in rec_date_recording_paths:
        print(f"Resampling {recording_path}")
        # Set up estimated save folder
        resampled_filename = recorder_folder.joinpath(f"{Path(recording_path).stem}_resampled.wav")
        start, end = frontierlabsutils.extract_start_end(Path(recording_path).name)
        if resampled_filename.exists():
            print("  Already resampled. Continuing.")
            continue

        # Get the time of this recording
        try:
            recording_time = recording_path.name.split('.')[0].split('T')[1][:4]
        except IndexError:
            print(f"  Recording {recording_path} not in correct format. Skipping.")
            continue

        # Get the write_times of all samples for this recording
        write_times = frontierlabsutils.get_recording_write_times(
            loclog_values, date=date, start_time=recording_time)
        if len(write_times) == 0:
            print(f"  No write times logged for {recorder}. Skipping")
            continue

        # Find buffer overflows and add 0s to compensate for missing buffers
        overflow_sample_indices, overflow_samples_to_insert = frontierlabsutils.get_buffer_insert_lengths(write_times)            
        audio = frontierlabsutils.insert_missing_buffers(
            recorder=recorder,
            date=date,
            time=recording_time,
            data_dir=data_dir,
            overflow_sample_indices=overflow_sample_indices,
            overflow_samples_to_insert=overflow_samples_to_insert,
            old_style=False
        )

        # Resample buffer-corrected audio
        resampled = frontierlabsutils.interpolate_audio(start, end, audio)

        # Save audio
        resampled.save(resampled_filename)
        
        # Keep track of how many we have resampled
        num_resampled += 1
        
        
t1 = time()

print(f"It took {(t1-t0)/60} minutes to resample {num_resampled} files")
