#################################################################################################################################
# embHEtools - version 1.0
# Created by: Dr. Erin Bayne
# Help from: ChatGPT and Google Gemini
# Released: July 28, 2025
#################################################################################################################################
# General description:
# Set of tools to crawl Cirrus server (or any hard drive)
# Select files to process
# Get HawkEars to look for species of interest (or all species)
# Create validation tables based on different validation rules
# Tools to see and hear audio to validate HawkEars labels

#################################################################################################################################

"""
Note: you will have to modify the audiolist_format function on line , which extracts metadata from the filenames, to match your data storage
and filenaming structure. It's important that the resulting csv has the following data columns as these will be expected by subsequent functions:
        filepath: path to recording file
        filename: name of recording file
        location: name of ARU station (prefix)
        recording_date: date 
        recording_start: start time
        latitude
        longitude
        filtype: filename extension (wav)
        
"""
# Required packages
import argparse
import csv
from datetime import datetime
import glob
import importlib
import pandas as pd
import numpy as np
import IPython.display as ipd
from IPython.display import display, Audio, clear_output, Markdown, HTML
import ipywidgets as widgets
import librosa
import librosa.display
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import os
from pathlib import Path
import random
import shutil
import soundfile as sf
import sqlite3
import subprocess
import sys

#################################################################################################################################
def normalize_path(path):
    """
    Converts a path to an absolute POSIX-style path (with forward slashes).
    """

    return Path(path).expanduser().resolve().as_posix()

#################################################################################################################################
class Tee:
    """
    Saves log files and displays them in a notebook at the same time
    """

    def __init__(self, logfilepath, mode="w", encoding="utf-8"):
        self.logfilepath = normalize_path(logfilepath) # Added July 30
        self.file = open(logfilepath, mode, encoding=encoding)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

#################################################################################################################################
def hawkears_selectspp(he_allspp_csv, spp_to_include, output_name='IGNORE.txt'):
    """
    Function to select species you want HawkEars to process. Does this by creating an IGNORE file that is used by HawkEars.
    Creates a copy of the original IGNORE file if one already exists.

    Args:
        he_allspp_csv (str): Path to the file that lists all the species that HawkEars is trained on.
        spp_to_include (list or set): Collection of species names or codes you want HawkEars to keep track of.
        output_name (str): Name of the IGNORE file to be written.
    """

    dir_path = os.path.dirname(os.path.abspath(he_allspp_csv))
    ignore_path = os.path.join(dir_path, output_name)

    # Backup existing IGNORE file if it exists
    if os.path.exists(ignore_path):
        counter = 1
        while True:
            backup_path = os.path.join(dir_path, f"IGNORE_{counter}.txt")
            if not os.path.exists(backup_path):
                shutil.move(ignore_path, backup_path)
                print(f"Renamed existing IGNORE.txt to: {backup_path}")
                break
            counter += 1

    # Handle empty spp_to_include — write a completely blank IGNORE file
    if not spp_to_include:
        open(ignore_path, 'w').close()
        print(f"⚠️ No species selected. Blank IGNORE file saved to: {ignore_path}")
        return

    # Continue with filtering
    df = pd.read_csv(he_allspp_csv)
    df_filtered = df[~df[['COMMON_NAME', 'CODE4', 'CODE6']].isin(spp_to_include).any(axis=1)]
    df_filtered = df_filtered.drop(columns=['CODE4', 'CODE6'])

    common_names = df_filtered['COMMON_NAME']
    common_names.to_csv(ignore_path, index=False, header=False)
    print(f"✅ Saved new IGNORE file to: {ignore_path}")

#################################################################################################################################
def audiolist_create(root_folder, output_csv):
    """
    Function to search your root directory of interest and find wav and wac files

    Args:
        root_folder (str): Lowest level you want to search. Recursively moves up from the root to find all files in all subdirectories
        output_csv (str): Name of the csv to store the files you find in your storage area
    """

    root_folder = normalize_path(root_folder)
    output_csv = normalize_path(output_csv)
    all_files = []

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.wac', '.wav')):
                file_path = Path(dirpath) / filename
                all_files.append(file_path.as_posix())

    if not all_files:
        print("🚫 HARD STOP: No audio files (.wav or .wac) found.")
        print(f"➡️  Checked directory: {root_folder}")
        print("Please check that your folder path is correct and contains valid audio recordings.")
        sys.exit(1)  # Immediate stop — no need to return

    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filepath'])
        writer.writerows([[f] for f in all_files])

    print(f"🧾 File list created for all subfolders. Results saved to {output_csv}")

#################################################################################################################################
def audiolist_count(input_csv, output_csv):
    """
    Reads a CSV of audio file paths, counts the number of `.wav` and `.wac` files per location,
    and saves the counts to a new CSV file.

    Args:
        input_csv (str): Path to the input CSV containing audio file paths.
        output_csv (str): Path where the summary CSV with counts per location will be saved.

    Stops:
        If input file cannot be read or no `.wav` files are found, prints a message and exits.
    """

    input_csv = normalize_path(input_csv)
    output_csv = normalize_path(output_csv)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ HARD STOP: Input file not found at {input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HARD STOP: An error occurred while reading the CSV: {e}")
        sys.exit(1)

    df['filepath'] = df['filepath'].apply(lambda x: Path(x).as_posix())
    df['location'] = df['filepath'].apply(lambda x: Path(x).parent.name)
    df['filetype'] = df['filepath'].apply(lambda x: Path(x).suffix.lower())

    file_counts = df[df['filetype'].isin(['.wac', '.wav'])] \
        .groupby(['location', 'filetype']) \
        .size() \
        .unstack(fill_value=0)

    if '.wac' not in file_counts.columns:
        file_counts['.wac'] = 0
    if '.wav' not in file_counts.columns:
        file_counts['.wav'] = 0

    file_counts = file_counts.rename(columns={'.wac': 'WACcount', '.wav': 'WAVcount'})
    file_counts['totalfiles'] = file_counts['WACcount'] + file_counts['WAVcount']

    try:
        file_counts.to_csv(output_csv)
        print(f"🧾 Number of recordings per location computed. Results saved to {output_csv}")
    except Exception as e:
        print(f"❌ HARD STOP: An error occurred while writing the CSV: {e}")
        sys.exit(1)

    if (file_counts['WAVcount'] == 0).all():
        print("⚠️ HARD STOP: No .wav files found in any location.")
        print("➡️  Check if your directories only contain .wac files.")
        sys.exit(1)

#################################################################################################################################
import re
import pandas as pd
from pathlib import Path

def audiolist_format(fileall, fileformat):
    """
    Reads a CSV containing file paths of audio recordings (with or without headers), filters for audio files,
    extracts metadata from filenames (date, start/end time, latitude, longitude, filetype),
    and uses the filename prefix as the location. Saves the formatted data to a new CSV with guaranteed column names.

    Time columns will have the letter 'T' removed if present.

    Args:
        fileall (str): Path to the input CSV file containing audio file paths.
        fileformat (str): Path to the output CSV file where formatted data will be saved.

    Returns:
        None
    """
    fileall = Path(fileall).resolve()
    fileformat = Path(fileformat).resolve()

    try:
        # Ensure the output directory exists
        fileformat.parent.mkdir(parents=True, exist_ok=True)

        # Read input CSV, try with header first, fallback to no header
        try:
            df = pd.read_csv(fileall)
            if 'filepath' not in df.columns:
                df.columns = ['filepath']
        except pd.errors.ParserError:
            df = pd.read_csv(fileall, header=None)
            df.columns = ['filepath']

        # Filter for audio files (any extension)
        audio_files_df = df[df['filepath'].str.lower().str.endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg'))].copy()
        audio_files_df['filepath'] = audio_files_df['filepath'].apply(lambda x: Path(x).as_posix())
        audio_files_df['filename'] = audio_files_df['filepath'].apply(lambda x: Path(x).name)

      # NEW: match audiolist_count – use parent folder as location
        audio_files_df['location'] = audio_files_df['filepath'].apply(lambda x: Path(x).parent.name)


        # Function to parse filenames and remove 'T' from time
        def parse_filename(filename):
            """
            Extracts date, start_time, end_time, latitude, and longitude from filename.
            Removes 'T' from time strings if present.
            Returns default values if parsing fails.
            """
            name_part = Path(filename).stem
            parts = name_part.split('_')

            # Default values
            date, start_time, end_time, latitude, longitude = None, None, None, '', ''

            if len(parts) >= 5:
                try:
                    start_ts = parts[1][1:]  # remove 'S'
                    end_ts = parts[2][1:]    # remove 'E'

                    # Remove 'T' if present
                    start_ts = start_ts.replace('T', '')
                    end_ts = end_ts.replace('T', '')

                    date = start_ts[:8]
                    start_time = start_ts[8:14]
                    end_time = end_ts[8:14]

                    # Extract latitude and longitude
                    coord_match = re.match(r'([+-]?\d+\.\d+)([+-]\d+\.\d+)', parts[3])
                    if coord_match:
                        latitude, longitude = coord_match.groups()
                except Exception:
                    pass  # keep default values if parsing fails

            return [date, start_time, end_time, latitude, longitude]

        # Apply parsing safely
        parsed_cols = audio_files_df['filename'].apply(parse_filename).tolist()
        audio_files_df[['recording_date', 'recording_time', 'end_time', 'latitude', 'longitude']] = parsed_cols

        # Assign 'filetype' dynamically based on the file extension
        audio_files_df['filetype'] = audio_files_df['filename'].apply(lambda x: Path(x).suffix.lower().lstrip('.'))

        # Ensure final column order
        final_columns = [
            'filepath', 'filename', 'location',
            'recording_date', 'recording_time', 'end_time',
            'latitude', 'longitude', 'filetype'
        ]
        audio_files_df = audio_files_df[final_columns]

        # Save CSV with guaranteed column names
        audio_files_df.to_csv(fileformat, index=False)
        print(f"🧾 File paths formatted. Results saved to {fileformat}")

    except FileNotFoundError:
        print(f"⚠️ The file '{fileall}' was not found.")
    except PermissionError:
        print(f"⚠️ Permission denied. Check that the file is not open or that you have write access to the folder.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


#################################################################################################################################
def audiolist_filter(fileall, min_mmdd, max_mmdd, min_time, max_time, filesubset, random_sample_per_day=None, random_sample_per_year=None):
    """
    Filters an audio files CSV based on recording date and time, and saves the filtered subset to a new CSV.
    If no valid rows match the date/time filters or if an error occurs, the function stops execution.
    Optionally, randomly selects a specified number of files per day or per year.
    """

    fileall = normalize_path(fileall)
    filesubset = normalize_path(filesubset)

    # Input validation for sampling options
    if random_sample_per_day is not None and random_sample_per_year is not None:
        print("❌ HARD STOP: Cannot specify both 'random_sample_per_day' and 'random_sample_per_year'.")
        sys.exit(1)

    try:
        df = pd.read_csv(fileall)
    except FileNotFoundError:
        print(f"❌ HARD STOP: The file '{fileall}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HARD STOP: An error occurred while reading '{fileall}': {e}")
        sys.exit(1)

    try:
        df['filepath'] = df['filepath'].apply(lambda x: Path(x).as_posix())

        df['recording_date'] = pd.to_numeric(df['recording_date'], errors='coerce')
        df['recording_time'] = pd.to_numeric(df['recording_time'], errors='coerce')
        df.dropna(subset=['recording_date', 'recording_time'], inplace=True)

        df['mmdd'] = df['recording_date'].astype(int) % 10000
        df['syear'] = df['recording_date'].astype(str).str[:4]
        df['sdate'] = df['recording_date'].astype(int)

        filtered_df = df[
            (df['mmdd'] >= min_mmdd) &
            (df['mmdd'] <= max_mmdd) &
            (df['recording_time'] >= min_time) &
            (df['recording_time'] <= max_time)
        ].copy()

        if filtered_df.empty:
            print("⚠️ HARD STOP: No files match the selected date and time filters.")
            print(f"➡️  Date range: {min_mmdd}–{max_mmdd}, Time range: {min_time}–{max_time}")
            sys.exit(1)

        final_filtered_df = filtered_df.copy()
        
        # Apply random sampling per day
        if random_sample_per_day is not None and random_sample_per_day > 0:
            print(f"➡️ Randomly selecting up to {random_sample_per_day} files per recording day.")
            
            sampled_list = []
            for (location, sdate), group in filtered_df.groupby(['location','sdate']):
                n_samples = min(random_sample_per_day, len(group))
                sampled_list.append(group.sample(n=n_samples, random_state=1))
            
            if sampled_list:
                final_filtered_df = pd.concat(sampled_list).reset_index(drop=True)
            else:
                final_filtered_df = pd.DataFrame()

        # Apply random sampling per year
        elif random_sample_per_year is not None and random_sample_per_year > 0:
            print(f"➡️ Randomly selecting up to {random_sample_per_year} files per year.")
            
            sampled_list = []
            for (location, syear), group in filtered_df.groupby(['location', 'syear']):
                n_samples = min(random_sample_per_year, len(group))
                sampled_list.append(group.sample(n=n_samples, random_state=1))
            
            if sampled_list:
                final_filtered_df = pd.concat(sampled_list).reset_index(drop=True)
            else:
                final_filtered_df = pd.DataFrame()

        if final_filtered_df.empty:
            print("⚠️ HARD STOP: After random sampling, no files remain.")
            sys.exit(1)

        hawkears_df = final_filtered_df[[
            'filepath', 'filename', 'location', 'latitude', 'longitude',
            'recording_date', 'mmdd', 'recording_time', 'filetype'
        ]].copy()

        hawkears_df.to_csv(filesubset, index=False)

        print(f"🔢 Original number of audio files in search: {len(df)}")
        print(f"🔢 Number of audio files per location selected based on date and time: {len(filtered_df)}")
        if random_sample_per_day or random_sample_per_year:
            print(f"🔢 Number of audio files per location after random sampling: {len(hawkears_df)}")
        print(f"🧾 Subset results saved to {filesubset}")

    except KeyError as e:
        print(f"❌ HARD STOP: Missing expected column in the data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HARD STOP: Unexpected error during filtering: {e}")
        sys.exit(1)

#################################################################################################################################
def audiolist_join(countfiles_csv, subsetfiles_csv, mergedfiles_csv):
    """
    Merges audio file count data with a subset of audio files by location and saves the merged result.

    Args:
        countfiles_csv (str): Path to the CSV file containing counts of audio files per location.
        subsetfiles_csv (str): Path to the CSV file containing a subset of audio files with metadata.
        mergedfiles_csv (str): Path to the output CSV file where the merged results will be saved.

    Stops:
        If inputs are missing, merge fails, or write fails, the workflow stops with a clear message.
    """

    countfiles_csv = normalize_path(countfiles_csv)
    subsetfiles_csv = normalize_path(subsetfiles_csv)
    mergedfiles_csv = normalize_path(mergedfiles_csv)

    try:
        count_df = pd.read_csv(countfiles_csv)
        subset_df = pd.read_csv(subsetfiles_csv)
    except FileNotFoundError:
        print(f"❌ HARD STOP: One or both input files not found:\n - {countfiles_csv}\n - {subsetfiles_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HARD STOP: Error reading input files: {e}")
        sys.exit(1)

    # 🔧 IMPORTANT: normalize subset locations to match count_df logic
    # count_df['location'] was created in audiolist_count as parent folder name.
    # We force subset_df['location'] to be the parent folder of filepath as well.
    subset_df['filepath'] = subset_df['filepath'].astype(str)
    subset_df['filepath'] = subset_df['filepath'].apply(lambda x: Path(x).as_posix())
    subset_df['location'] = subset_df['filepath'].apply(lambda x: Path(x).parent.name)

    merged_df = pd.merge(count_df, subset_df, on='location', how='inner')

    if merged_df.empty:
        print("⚠️ HARD STOP: No subset files could be merged with count data.")
        print("➡️  This usually means the folder names in the count file and subset file do not match.")
        print("➡️  After forcing subset locations from parent folder, there are still no matches.")
        sys.exit(1)

    print(f"🔢 Successfully merged {len(merged_df)} subset files to locations.")

    try:
        merged_df.to_csv(mergedfiles_csv, index=False)
        print(f"🧾 Merged results saved to {mergedfiles_csv}")
    except Exception as e:
        print(f"❌ HARD STOP: Failed to write merged CSV: {e}")
        sys.exit(1)


#################################################################################################################################
def audiolist_filelist(input_csv, output_base_dir):
    """
    Reads a merged CSV of audio file metadata, then creates and saves separate CSV filelists
    for each unique location containing selected columns. Required by HawkEars

    Args:
        input_csv (str): Path to the input CSV file with merged audio metadata.
        output_base_dir (str): Directory where per-location CSV filelists will be saved.

    Returns:
        None

    Prints:
        Error messages if the input CSV is not found or cannot be read.
        Warnings if expected columns are missing in the data for any location.
        Error messages if saving any per-location CSV fails.
        Confirmation message upon successful completion.
    """

    input_csv = normalize_path(input_csv)
    output_base_dir = normalize_path(output_base_dir)
    try:
        merged_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("❌  Input file not found.")
        #return None
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred while reading the CSV file: {e}")
        #return None
        sys.exit(1)

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    unique_locations = merged_df['location'].unique()

    for location in unique_locations:
        location_df = merged_df[merged_df['location'] == location].copy()
        required_columns = ['filename', 'latitude', 'longitude', 'recording_date']
        for col in required_columns:
            if col not in location_df.columns:
                print(f"Warning: Column '{col}' not found for location {location}.")
                location_df[col] = ''

        location_filelist_df = location_df[required_columns]
        output_filename = f"{location}_HElist.csv"
        output_filepath = Path(output_base_dir) / output_filename

        try:
            location_filelist_df.to_csv(output_filepath.as_posix(), index=False)
        except Exception as e:
            print(f"❌  An error occurred while writing the CSV for {location}: {e}")
            sys.exit(1)

    print(f"🧾 Finished creating filelists for each location. Results saved to {output_base_dir}.")

#############################################################################################
def audiolist_listoflists(merged_csv_path, listoflists_csv, inputs_folder, tag_output):
    """
    Generates a master CSV listing audio directories and associated HElist and tag output file paths by location.

    Args:
        merged_csv_path (str): Path to the merged CSV containing audio file metadata including 'filepath' and 'location'.
        listoflists_csv (str): Path where the generated master CSV listing HE filelists will be saved.
        inputs_folder (str): Directory path where HElist files are stored.
        tag_output (str): Path specifying the output location for tagging results.

    Returns:
        pandas.DataFrame: DataFrame containing unique entries for audio paths, HElist file paths, and tag output path.
                          Returns an empty DataFrame with expected columns if an error occurs.

    Prints:
        Error message if required columns are missing or if any exception occurs.
        Confirmation message upon successful creation of the master CSV.
    """
    try:
        merged_csv_path = normalize_path(merged_csv_path)
        listoflists_csv = normalize_path(listoflists_csv)
        inputs_folder = normalize_path(inputs_folder)
        tag_output = normalize_path(tag_output)

        merged_df = pd.read_csv(merged_csv_path)

        if 'filepath' not in merged_df.columns or 'location' not in merged_df.columns:
            print("❌ Required columns ('filepath' and 'location') not found in the merged CSV.")
            return pd.DataFrame(columns=['audio_path', 'filelist_path', 'filelist', 'tag_output'])

        # Normalize and extract path info using pathlib
        merged_df['filepath'] = merged_df['filepath'].apply(lambda x: normalize_path(x))
        merged_df['audio_path'] = merged_df['filepath'].apply(lambda x: str(Path(x).parent.as_posix()))
        merged_df['filelist'] = merged_df['location'].astype(str) + "_HElist.csv"
        merged_df['filelist_path'] = merged_df['filelist'].apply(lambda f: f"{inputs_folder}/{f}")
        merged_df['tag_output'] = tag_output

        listoflists_df = merged_df[['audio_path', 'filelist_path', 'filelist', 'tag_output']] \
            .drop_duplicates(subset=['filelist_path']).reset_index(drop=True)

        Path(listoflists_csv).parent.mkdir(parents=True, exist_ok=True)
        listoflists_df.to_csv(listoflists_csv, index=False)

        print(f"🧾 Master list of all HE filelists created by location. Results saved to {listoflists_csv}")
        return listoflists_df

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return pd.DataFrame(columns=['audio_path', 'filelist_path', 'filelist', 'tag_output'])

#################################################################################################################################
def hawkears_run(database_name, listoflists_path, python, hawkears_code, cutoff=0.8, overlap=1.5, merge=1):
    """
    Runs HawkEars for a list of locations and appends the results to a SQL database,
    Inserts placeholder rows when no detections are made for specific filenames
    (Not_spp is inserted if no labels are found by HawkEars).
    """

    # Normalize paths
    database_name = normalize_path(database_name)
    listoflists_path = normalize_path(listoflists_path)
    hawkears_code = normalize_path(hawkears_code)
    python = normalize_path(python)

    # >>> NEW: set HawkEars root directory (for cwd)
    hawkears_root = os.path.dirname(hawkears_code)
    print(f"[embHEtools] HawkEars root (cwd) will be: {hawkears_root}")

    # Connect to database
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    print(f"🔗 Connected to SQLite database: {database_name}")

    # Select database table
    database_table = "hawkears_results"
    insert_columns = [
        'filename', 'start_time', 'end_time', 'class_name',
        'class_code', 'score', 'original_filelist'
    ]

    # Create table if does not exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{database_table}" (
            filename TEXT,
            start_time REAL,
            end_time REAL,
            class_name TEXT,
            class_code TEXT,
            score REAL,
            original_filelist TEXT,
            UNIQUE(filename, start_time, end_time, class_code, score, original_filelist)
        );
    """)
    conn.commit()

    # >>> NEW: define insert_query once, so it's available in all branches
    insert_query = f"""
        INSERT OR IGNORE INTO "{database_table}"
        ({', '.join(insert_columns)})
        VALUES ({', '.join(['?'] * len(insert_columns))})
    """

    try:
        listoflists_df = pd.read_csv(listoflists_path)
        print(f"📄 Loaded list of lists: {listoflists_path}")
    except Exception as e:
        print(f"❌ Failed to read list of lists: {e}")
        conn.close()
        return

    for _, row in listoflists_df.iterrows():
        audio_dir = normalize_path(str(row.get("audio_path", "")))
        filelist_dir = normalize_path(str(row.get("filelist_path", "")))
        labels_dir = normalize_path(str(row.get("tag_output", "")))

        if not audio_dir or not filelist_dir or not labels_dir:
            print(f"⚠️ Missing required paths in row: {row}")
            continue

        output_csv_path = os.path.join(labels_dir, "HawkEars_labels.csv")

        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir, exist_ok=True)
            print(f"📂 Created labels directory: {labels_dir}")

        # Load filenames to process
        try:
            filelist_df = pd.read_csv(filelist_dir)
            if 'filename' in filelist_df.columns:
                filenames = filelist_df['filename'].astype(str).tolist()
            else:
                filenames = filelist_df.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            print(f"❌ Failed to read filelist {filelist_dir}: {e}")
            filenames = [os.path.basename(filelist_dir).replace('_HElist.csv', '.wav')]
            filelist_df = pd.DataFrame()

        if not filenames:
            print(f"⚠️ No filenames found in filelist {filelist_dir}")
            filenames = [os.path.basename(filelist_dir).replace('_HElist.csv', '.wav')]

        # Run HawkEars subprocess
        try:
            command = [
                python, hawkears_code,
                "-i", audio_dir,
                "-o", labels_dir,
                "--rtype", "csv",
                "--filelist", filelist_dir,
                "--overlap", str(overlap),
                "--merge", str(int(merge)),
                "-p", str(cutoff),
                "--fast"
            ]
            print(f"🚀 Running (cwd={hawkears_root}): {' '.join(command)}")
            # >>> NEW: set cwd so relative paths like data/ckpt and data/occurrence.db work
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=hawkears_root
            )
            print(result.stdout)
            if result.stderr:
                print(f"⚠️ Warnings/Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"❌ HawkEars subprocess failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            # If HawkEars itself failed, don't treat this as "no detections";
            # just continue to next location without inserting placeholders.
            continue
        except Exception as e:
            print(f"❌ Unexpected error during HawkEars run: {e}")
            continue

        # Process HawkEars output
        if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
            try:
                hawkears_output_df = pd.read_csv(output_csv_path)
                hawkears_output_df['original_filelist'] = os.path.basename(filelist_dir)

                # Ensure all expected columns exist
                for col in insert_columns:
                    if col not in hawkears_output_df.columns:
                        hawkears_output_df[col] = None

                detected_files = set(hawkears_output_df['filename'].astype(str))

                # Insert real detections
                inserted_count = 0
                for _, det_row in hawkears_output_df.iterrows():
                    values = tuple(det_row[col] for col in insert_columns)
                    try:
                        cursor.execute(insert_query, values)
                        inserted_count += cursor.rowcount
                    except Exception as e:
                        print(f"⚠️ Insert error: {e}")

                conn.commit()
                if inserted_count > 0:
                    print(f"✅ Inserted {inserted_count} unique HawkEars label(s) into database")
                else:
                    print(f"⚠️ All HawkEars labels were duplicates. No labels added to database")

                # Prepare placeholders for missing files
                missing_files = [f for f in filenames if f not in detected_files]
                if missing_files:
                    print(f"➕ Adding Not_spp for {len(missing_files)} filename(s) as selected species not detected at threshold you selected.")
                    placeholder_data = {
                        'filename': missing_files,
                        'start_time': [0] * len(missing_files),
                        'end_time': [0] * len(missing_files),
                        'class_name': ['Not_spp'] * len(missing_files),
                        'class_code': ['Not_spp'] * len(missing_files),
                        'score': [0] * len(missing_files),
                        'original_filelist': [os.path.basename(filelist_dir)] * len(missing_files)
                    }
                    placeholder_df = pd.DataFrame(placeholder_data)

                    # Query existing placeholders to avoid duplicates
                    existing_placeholders = pd.read_sql_query(f"""
                        SELECT filename, original_filelist
                        FROM "{database_table}"
                        WHERE score = 0 AND start_time = 0 AND end_time = 0
                    """, conn)

                    if not existing_placeholders.empty:
                        existing_keys = set(zip(
                            existing_placeholders['filename'],
                            existing_placeholders['original_filelist']
                        ))
                        placeholder_df = placeholder_df[
                            ~placeholder_df[['filename', 'original_filelist']].apply(tuple, axis=1).isin(existing_keys)
                        ]

                    inserted_count = 0
                    if not placeholder_df.empty:
                        for _, row2 in placeholder_df.iterrows():
                            values = tuple(row2[col] for col in insert_columns)
                            try:
                                cursor.execute(insert_query, values)
                                inserted_count += cursor.rowcount
                            except Exception as e:
                                print(f"❌ Insert error: {e}")
                        conn.commit()

                    if inserted_count > 0:
                        print(f"📝 Appended {inserted_count} placeholder row(s).")
                    elif not placeholder_df.empty:
                        print("⚠️ No new Not_spp added. You already processed this data at this cutoff.")

            except Exception as e:
                print(f"❌ Error reading HawkEars output CSV: {e}")
        else:
            print(f"⚠️ HawkEars output missing or empty: {output_csv_path}")
            # Insert placeholders for all files when output is missing or empty
            placeholder_data = {
                'filename': filenames,
                'start_time': [0] * len(filenames),
                'end_time': [0] * len(filenames),
                'class_name': ['Not_spp'] * len(filenames),
                'class_code': ['Not_spp'] * len(filenames),
                'score': [0] * len(filenames),
                'original_filelist': [os.path.basename(filelist_dir)] * len(filenames)
            }
            placeholder_df = pd.DataFrame(placeholder_data)

            # Query existing placeholders to avoid duplicates
            existing_placeholders = pd.read_sql_query(f"""
                SELECT filename, original_filelist
                FROM "{database_table}"
                WHERE score = 0 AND start_time = 0 AND end_time = 0
            """, conn)

            if not existing_placeholders.empty:
                existing_keys = set(zip(
                    existing_placeholders['filename'],
                    existing_placeholders['original_filelist']
                ))
                placeholder_df = placeholder_df[
                    ~placeholder_df[['filename', 'original_filelist']].apply(tuple, axis=1).isin(existing_keys)
                ]

            inserted_count = 0
            if not placeholder_df.empty:
                for _, row2 in placeholder_df.iterrows():
                    values = tuple(row2[col] for col in insert_columns)
                    try:
                        cursor.execute(insert_query, values)
                        inserted_count += cursor.rowcount
                    except Exception as e:
                        print(f"❌ Insert error: {e}")
                conn.commit()

            if inserted_count > 0:
                print(f"📝 Appended {inserted_count} placeholder row(s).")
            elif not placeholder_df.empty:
                print("⚠️ Not_spp was not added. You already processed this filename at these settings.")

    conn.close()
    print(f"🔒 HawkEars done. Database closed")


#################################################################################################################################
def hawkears_dbaseupdate(database_name):
    """
    Updates 'location', 'sdate', 'stime', and 'filetype' fields in the hawkears_results table,
    and initializes 'sppTF', 'filechkTF', and 'skipTF' as 0 (False) unless already present.
    """

    database_name = normalize_path(database_name)
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Add columns if they don't exist
    new_columns = {
        'location': 'TEXT',
        'sdate': 'TEXT',
        'stime': 'TEXT',
        'filetype': 'TEXT',
        'sppTF': 'BOOLEAN DEFAULT 0',
        'filechkTF': 'BOOLEAN DEFAULT 0',
        'skipTF': 'BOOLEAN DEFAULT 0'
    }

    for column, dtype in new_columns.items():
        try:
            cursor.execute(f"ALTER TABLE hawkears_results ADD COLUMN {column} {dtype}")
            print(f"🗃️ New database table created. Added columns: {column}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Read into DataFrame
    hawkears_df = pd.read_sql_query(
        "SELECT rowid, filename, location, sdate, stime, filetype, skipTF, sppTF, filechkTF FROM hawkears_results", conn)

    # Parse filename into parts
    parts = hawkears_df['filename'].str.split(r'[_\.]', expand=True)
    hawkears_df['parsed_location'] = parts[0] if parts.shape[1] > 0 else None
    hawkears_df['parsed_sdate'] = parts[1] if parts.shape[1] > 1 else None
    hawkears_df['parsed_stime'] = parts[2] if parts.shape[1] > 2 else None
    hawkears_df['parsed_filetype'] = parts.iloc[:, -1] if parts.shape[1] > 0 else None

    for _, row in hawkears_df.iterrows():
        update_fields = {}
        if not row['location']:
            update_fields['location'] = row['parsed_location']
        if not row['sdate']:
            update_fields['sdate'] = row['parsed_sdate']
        if not row['stime']:
            update_fields['stime'] = row['parsed_stime']
        if not row['filetype']:
            update_fields['filetype'] = row['parsed_filetype']

        if update_fields:
            sql = f"UPDATE hawkears_results SET {', '.join(f'{k} = ?' for k in update_fields)} WHERE rowid = ?"
            try:
                cursor.execute(sql, list(update_fields.values()) + [row['rowid']])
            except Exception as e:
                print(f"❌ Error updating rowid {row['rowid']}: {e}")

    conn.commit()
    conn.close()

#########################################################################################
def hawkears_filesrun(rootdir_subset, database_name):
    """
    Searches for '*_subsetfiles.csv' files under rootdir_subset, combines them into one DataFrame,
    and appends only unique new rows (based on 'filename') to the 'all_subsetfiles' table in the SQLite database.

    Parameters:
    - rootdir_subset (str): Path to the directory to search
    - database_name (str): Path to the SQLite database
    """
    # Search recursively for matching files
    pattern = os.path.join(rootdir_subset, "**", "*_subsetfiles.csv")
    matching_files = glob.glob(pattern, recursive=True)

    if not matching_files:
        print("⚠️ No *_subsetfiles.csv files found.")
        return

    # Read all new CSV files
    new_dfs = []
    for file in matching_files:
        try:
            df = pd.read_csv(file)
            df['source_path'] = file  # optional: track source file
            new_dfs.append(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

    if not new_dfs:
        print("⚠️ No valid CSV files could be read.")
        return

    new_combined_df = pd.concat(new_dfs, ignore_index=True)

    try:
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()

        # Load existing filenames if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='all_subsetfiles';")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            existing_filenames = pd.read_sql_query("SELECT filename FROM all_subsetfiles", conn)['filename'].unique()
            # Filter to only new filenames
            new_combined_df = new_combined_df[~new_combined_df['filename'].isin(existing_filenames)]

        if new_combined_df.empty:
            print("⚠️ No new filenames to add. They already exist in database.")
        else:
            new_combined_df.to_sql("all_subsetfiles", conn, if_exists='append', index=False)
            print(f"✅ Appended {len(new_combined_df)} new filenames to database")

        # Create index if not exists
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_all_subsetfiles_filename ON all_subsetfiles(filename);")
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"❌ Failed to update database: {e}")

#################################################################################################################################
def validate_topXdetections(df, top_n=None, min_spacing=None):
    """
    Select top N detections per filename with minimum spacing between detections.
    Always assigns a 'rank' column within each filename group by descending score.

    Args:
        df (pd.DataFrame): Input detections with 'filename', 'start_time', 'score'.
        top_n (int or None): Max number of detections per file.
        min_spacing (float): Minimum time in seconds between detections.

    Returns:
        pd.DataFrame: Filtered detections with 'rank' column.
    """
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df = df.dropna(subset=['start_time'])

    result = []

    for filename, group in df.groupby("filename"):
        group = group.sort_values("score", ascending=False).copy()

        selected = []
        selected_times = []

        for _, row in group.iterrows():
            this_time = row['start_time']
            if all(abs(this_time - t) >= min_spacing for t in selected_times):
                selected.append(row)
                selected_times.append(this_time)
                if top_n is not None and len(selected) >= top_n:
                    break

        if selected:
            selected_df = pd.DataFrame(selected)
            selected_df = selected_df.sort_values("score", ascending=False).copy()
            selected_df['rank'] = range(1, len(selected_df) + 1)
            result.append(selected_df)

    if result:
        return pd.concat(result, ignore_index=True)
    else:
        return pd.DataFrame(columns=df.columns.tolist() + ['rank'])

######################################################################################################################
def validate_maketables(database_name,
                        validation_table="vtbl",
                        top_n=None,
                        min_spacing=6.0,
                        validation_type=7,
                        labelsperbin=10,
                        target_species=None):
    """
    Create validation tables per class_code with:
      - row_order INTEGER PRIMARY KEY AUTOINCREMENT
      - UNIQUE(filename, start_time, end_time, class_code)
    Insert rows using INSERT OR IGNORE but report attempted/skipped/inserted counts.
    """
    import sqlite3
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from embHEtools import validate_topXdetections

    def normalize_path(p):
        return str(Path(p).resolve())

    database_name = normalize_path(database_name)
    conn = sqlite3.connect(database_name)

    # Load all detections
    all_detections_df = pd.read_sql_query("SELECT * FROM hawkears_results", conn)
    required_cols = {'filename', 'start_time', 'end_time', 'class_code', 'score',
                     'sppTF', 'filechkTF', 'skipTF', 'location', 'sdate', 'stime', 'filetype'}
    missing = required_cols - set(all_detections_df.columns)
    if missing:
        conn.close()
        raise ValueError(f"Missing columns in hawkears_results: {missing}")

    # Filter species if requested
    if target_species is not None:
        if isinstance(target_species, str):
            target_species = [target_species]
        all_detections_df = all_detections_df[all_detections_df["class_code"].isin(target_species)].copy()
        if all_detections_df.empty:
            conn.close()
            raise ValueError("No labels found for the specified target_species.")

    # Distinct class codes
    class_codes = all_detections_df['class_code'].dropna().unique()

    # Load filepath mapping (optional)
    try:
        filepaths_df = pd.read_sql_query("SELECT filename, filepath FROM all_subsetfiles", conn)
        filepaths_df = filepaths_df.drop_duplicates(subset=['filename'])
    except Exception:
        filepaths_df = pd.DataFrame(columns=['filename', 'filepath'])

    # Helpers

    def create_table_with_autoinc(conn, tbl, df_template):
        """Create table with autoinc row_order and UNIQUE(...) if missing."""
        cur = conn.cursor()
        cols = []
        for c in df_template.columns:
            if c == "row_order":
                continue
            ser = df_template[c]
            if pd.api.types.is_integer_dtype(ser):
                ctype = "INTEGER"
            elif pd.api.types.is_float_dtype(ser):
                ctype = "REAL"
            elif pd.api.types.is_bool_dtype(ser):
                ctype = "INTEGER"
            else:
                ctype = "TEXT"
            cols.append(f'"{c}" {ctype}')
        # ensure dedupe columns exist
        for req in ["filename", "start_time", "end_time", "class_code"]:
            if req not in df_template.columns:
                cols.append(f'"{req}" TEXT')
        cols_sql = ",\n  ".join(cols)
        create_sql = f'''
        CREATE TABLE IF NOT EXISTS "{tbl}" (
          row_order INTEGER PRIMARY KEY AUTOINCREMENT,
          {cols_sql},
          UNIQUE(filename, start_time, end_time, class_code)
        );
        '''
        cur.executescript(create_sql)
        conn.commit()

    def get_table_columns(conn, tbl):
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{tbl}")')
        rows = cur.fetchall()
        return [r[1] for r in rows]  # second element is column name

    def insert_or_ignore_with_report(conn, tbl, df):
        """
        Insert rows into tbl using INSERT OR IGNORE.
        Report attempted, skipped (already present by unique key), and inserted counts.
        """
        tbl_cols = get_table_columns(conn, tbl)
        insert_cols = [c for c in df.columns if c != 'row_order' and c in tbl_cols]
        if not insert_cols:
            print(f"⚠️ No insertable columns found for table '{tbl}'.")
            return 0, 0, 0

        unique_key_cols = ["filename", "start_time", "end_time", "class_code"]
        dedupe_cols = [c for c in unique_key_cols if c in tbl_cols]
        attempted = len(df)

        try:
            count_before = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
        except Exception:
            count_before = 0

        if count_before == 0 or not dedupe_cols:
            df_to_insert = df.copy()
        else:
            existing_df = pd.read_sql_query(f'SELECT {",".join(dedupe_cols)} FROM "{tbl}"', conn).drop_duplicates()
            merged = df.merge(existing_df, on=dedupe_cols, how='left', indicator=True)
            df_to_insert = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

        skipped = attempted - len(df_to_insert)
        if len(df_to_insert) == 0:
            print(f"ℹ️ No new labels to add for table '{tbl}'. (attempted {attempted}, skipped {skipped}, inserted 0)")
            return attempted, skipped, 0

        rows = [tuple(None if pd.isna(x) else x for x in r) for r in df_to_insert[insert_cols].itertuples(index=False, name=None)]
        placeholders = ", ".join(["?"] * len(insert_cols))
        colnames_quoted = ", ".join([f'"{c}"' for c in insert_cols])
        sql = f'INSERT OR IGNORE INTO "{tbl}" ({colnames_quoted}) VALUES ({placeholders})'
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()

        count_after = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
        inserted = count_after - count_before

        print(f"✅ Added {inserted} new labels to '{tbl}' (attempted {attempted}, skipped {skipped})")
        return attempted, skipped, inserted

    # Main loop per class_code
    for class_code in class_codes:
        safe_class_code = str(class_code).replace(" ", "_").replace("-", "_")
        validation_table_name = f"{validation_table}_{safe_class_code}"

        df_class = all_detections_df[all_detections_df["class_code"] == class_code].copy()
        df_top = validate_topXdetections(df_class, top_n=top_n, min_spacing=min_spacing)

        # Sampling / sorting logic
        if validation_type == 1:
            df = df_top.sort_values(['location', 'score'], ascending=[True, False])
        elif validation_type == 2:
            df = df_top.sort_values(['location', 'rank', 'sdate'], ascending=[True, True, True])
        elif validation_type == 3:
            df = df_top.sort_values(['filename', 'rank'], ascending=[True, True])
        elif validation_type == 4:
            df = df_top.sort_values(['filename', 'sdate', 'stime', 'start_time'], ascending=True)
        elif validation_type == 5:
            max_idx = df_class.groupby("filename")["score"].idxmax()
            min_idx = df_class.groupby("filename")["score"].idxmin()
            df_max = df_class.loc[max_idx].copy(); df_max['rank'] = 1
            df_min = df_class.loc[min_idx].copy(); df_min['rank'] = 2
            df = pd.concat([df_max, df_min], ignore_index=True)
            df = df.drop_duplicates(subset=["filename", "start_time", "score"])
            df = df.sort_values(["location", "sdate", "stime", "rank"]).copy()
        elif validation_type == 6:
            bins = np.linspace(0.01, 1.0, 100)
            df_top['score_bin'] = pd.cut(df_top['score'], bins=bins, include_lowest=True)
            df = df_top.groupby('score_bin').head(labelsperbin)
        elif validation_type == 7:
            df = df_top.sort_values(['location', 'sdate', 'stime'])
        else:
            conn.close()
            raise ValueError(f"Unknown validation_type: {validation_type}")

        df['validation_type'] = validation_type
        df = df.merge(filepaths_df, on='filename', how='left')

        # Ensure table exists (creates with AUTOINC and UNIQUE if missing)
        create_table_with_autoinc(conn, validation_table_name, df)

        # Insert and report
        attempted, skipped, inserted = insert_or_ignore_with_report(conn, validation_table_name, df)

        if inserted == 0 and skipped == 0:
            print(f"Class '{class_code}': nothing to insert into '{validation_table_name}'.")
        elif inserted == 0 and skipped > 0:
            print(f"ℹ️ Class '{class_code}': attempted {attempted}, skipped {skipped}, inserted {inserted} into '{validation_table_name}'")
        else:
            print(f"✅ Class '{class_code}': successfully inserted {inserted} new rows into '{validation_table_name}'. Attempted: {attempted}, Skipped: {skipped}")

    conn.close()

######################################################################################################################
def OLDvisualize_spectrogram(
    database_name,
    validation_table,
    buffer=0.0,
    min_freq=1000,
    max_freq=10000,
    n_fft=512,
    hop_length=256,
    target_sr=16000,
    use_setduration=False,
    set_duration=10.0,
    shared_state=None,
    filename_changed_callback=None
):
    import sqlite3
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, Audio, Markdown, clear_output, HTML
    import librosa
    import librosa.display
    import IPython.display as ipd

    status_output = widgets.Output()

    database_name = str(Path(database_name).resolve())
    conn = sqlite3.connect(database_name)

    # Determine validation type first (single value expected)
    try:
        raw_df = pd.read_sql_query(f'SELECT * FROM "{validation_table}" ORDER BY row_order', conn)
        vt_unique = raw_df['validation_type'].dropna().unique()
        if len(vt_unique) == 0:
            validation_type = 7
            print("⚠️ No 'validation_type' listed in table; defaulting to 7 (alllabels).")
        else:
            validation_type = int(vt_unique[0])
            if len(vt_unique) > 1:
                print("⚠️ Multiple validation_type values found; using the first:", validation_type)
    except Exception as e:
        validation_type = 7
        print("⚠️ Failed to read validation_type; defaulting to 7. Error:", e)
        raw_df = pd.read_sql_query(f'SELECT * FROM "{validation_table}" ORDER BY row_order', conn)

    # Build SQL query with exclusion based on validation type
    if validation_type in [1, 2]:
        sql = f"""
        SELECT * FROM "{validation_table}"
        WHERE location NOT IN (
            SELECT location FROM "{validation_table}"
            GROUP BY location
            HAVING SUM(filechkTF) = COUNT(*)
        )
        ORDER BY row_order
        """
    elif validation_type in [3, 4]:
        sql = f"""
        SELECT * FROM "{validation_table}"
        WHERE filename NOT IN (
            SELECT filename FROM "{validation_table}"
            GROUP BY filename
            HAVING SUM(filechkTF) = COUNT(*)
        )
        ORDER BY row_order
        """
    else:
        sql = f'SELECT * FROM "{validation_table}" ORDER BY row_order'

    df = pd.read_sql_query(sql, conn)

    if df.empty:
        print(f"❌ No more labels to validate.")
        conn.close()
        return

    # Further filter in pandas based on validation type
    if validation_type in [1, 2]:
        validated = df[df['sppTF'] == 1]['location'].unique()
        df = df[~df['location'].isin(validated)]
    elif validation_type in [3, 4]:
        validated = df[df['sppTF'] == 1]['filename'].unique()
        df = df[~df['filename'].isin(validated)]
    elif validation_type in [5, 6, 7]:
        pass
    else:
        conn.close()
        raise ValueError(f"Unsupported validation_type: {validation_type}")

    df = df.reset_index(drop=True)

    if df.empty:
        print("✅ No labels to validate.")
        conn.close()
        return

    total_rows = len(df)

    # Tracking widgets
    idx_box = widgets.BoundedIntText(value=0, min=0, max=total_rows - 1, step=1, description="Label index:")
    # Row box shows *row_order* values (DB IDs)
    row_box = widgets.BoundedIntText(value=int(df.loc[0, 'row_order']), 
                                     min=int(df['row_order'].min()), 
                                     max=int(df['row_order'].max()), step=1, description="Row order #:")
    output = widgets.Output()

    # Busy flag to avoid re-entrant callbacks
    busy = {'flag': False}

    def plot_spectrogram_and_audio(i):
        # guard: i must be in-range for current df
        output.clear_output()
        with output:
            if len(df) == 0:
                print("✅ No more labels to validate.")
                return

            if i is None or i < 0 or i >= len(df):
                print("⚠️ Index out of range for current DataFrame.")
                return

            # Capture row data early so later mutations don't change what we display
            row = df.iloc[i].copy()
            try:
                # Temporarily set busy while updating the displayed row_order
                busy['flag'] = True
                row_box.value = int(row['row_order'])
            finally:
                busy['flag'] = False

            filepath = row['filepath']
            start_time = float(row['start_time']) if pd.notnull(row['start_time']) else 0.0
            end_time = float(row['end_time']) if pd.notnull(row['end_time']) else start_time + 1.0

            # Determine audio length in a lightweight way if needed
            audio_length = None
            if use_setduration:
                try:
                    import soundfile as sf
                    info = sf.info(filepath)
                    audio_length = info.frames / info.samplerate
                except Exception:
                    # fallback to librosa for length, but only load header if possible
                    try:
                        full_y, full_sr = librosa.load(filepath, sr=target_sr, duration=0.1)
                        audio_length = librosa.get_duration(filename=filepath)
                    except Exception:
                        audio_length = None

            if use_setduration:
                duration = min(set_duration, audio_length if audio_length is not None else set_duration)
                offset = 0.0
            else:
                duration = end_time - start_time
                offset = max(0, start_time - buffer)
                duration = duration + 2 * buffer

            # Clip duration to audio length if available
            if audio_length is not None:
                if offset + duration > audio_length:
                    duration = max(0, audio_length - offset)

            # To create counters
            loc = row['location']
            fn = row['filename']
            df_loc = df[df['location'] == loc]
            df_fn = df[df['filename'] == fn]

            # Calculate 1-based row number inside location group (relative to df)
            try:
                # find position in df_loc of the row with this row_order
                row_in_loc = int(df_loc.reset_index().reset_index().loc[lambda x: x['index'] == i, 'level_0'].values[0]) + 1
            except Exception:
                # fallback simpler approach
                row_in_loc = None

            try:
                row_in_fn = int(df_fn.reset_index().reset_index().loc[lambda x: x['index'] == i, 'level_0'].values[0]) + 1
            except Exception:
                row_in_fn = None

            n_loc = len(df_loc)
            v_loc = int(df_loc['sppTF'].sum()) if 'sppTF' in df_loc.columns else 0
            n_fn = len(df_fn)
            v_fn = int(df_fn['sppTF'].sum()) if 'sppTF' in df_fn.columns else 0

            # Print statements on status of processing
            display(Markdown(f"""
  📍 **Location:** `{loc}` — Filename {row_in_loc if row_in_loc is not None else '?'} of {n_loc}\n
  📄 **Filename:** `{Path(fn).name}` — Label {row_in_fn if row_in_fn is not None else '?'} of {n_fn}
- Validation type: `{row.get('validation_type', validation_type)}`
- Species validating: `{row.get('class_code', '')}`
- HawkEars score: `{row.get('score', 0.0):.2f}`
- Detection time: `{start_time:.2f}s – {end_time:.2f}s`
- Audio segment: `{offset:.2f}s – {offset + duration:.2f}s`
- Label rank: `{row.get('rank', '')}`
"""))

            # Plot spectrogram + audio
            try:
                y, sr = librosa.load(filepath, sr=target_sr, offset=offset, duration=duration)
                S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                plt.figure(figsize=(6, 3))
                librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
                plt.ylim(min_freq, max_freq)
                plt.colorbar(format="%+2.0f dB")
                plt.title("Spectrogram")

                if not use_setduration:
                    # show detection window
                    plt.axvline(x=buffer, color='lime', linestyle='--', linewidth=1.5)
                    plt.axvline(x=buffer + (end_time - start_time), color='red', linestyle='--', linewidth=1.5)

                plt.tight_layout()
                plt.show()

                display(HTML("<br><br>"))
                ipd.display(ipd.Audio(y, rate=sr))

            except Exception as e:
                print(f"⚠️ Failed to load/plot audio: {e}")

    def update_widget_bounds():
        # call whenever df changes
        idx_box.max = max(0, len(df) - 1)
        if len(df) > 0 and 'row_order' in df.columns:
            try:
                row_box.min = int(df['row_order'].min())
                row_box.max = int(df['row_order'].max())
            except Exception:
                row_box.min = 0
                row_box.max = max(0, len(df) - 1)

    def update_row_tag(sppTF_val, filechkTF_val, skip_to="next"):
        nonlocal df
        i = idx_box.value
        if i >= len(df):
            return

        # Capture current row fields *before* we mutate df
        row = df.iloc[i].copy()
        current_row_order = int(row['row_order'])
        current_location = row['location']
        current_filename = row['filename']

        cursor = conn.cursor()

        with status_output:
            status_output.clear_output(wait=True)

            # Use row_order for updates (exact match)
            try:
                cursor.execute(f"""
                    UPDATE "{validation_table}"
                    SET sppTF = ?, filechkTF = ?
                    WHERE row_order = ?
                """, (sppTF_val, filechkTF_val, current_row_order))
                conn.commit()
            except Exception as e:
                print("⚠️ DB update failed:", e)
                return

            try:
                cursor.execute(f"""
                    SELECT sppTF, filechkTF FROM "{validation_table}" WHERE row_order = ?
                """, (current_row_order,))
                updated_values = cursor.fetchone()
            except Exception as e:
                updated_values = None
                print("⚠️ DB fetch after update failed:", e)

            if updated_values:
                print(f"🔗 The previous label with row_order={current_row_order} in {validation_table} was set to: sppTF={updated_values[0]}, filechkTF={updated_values[1]}")
            else:
                print(f"❌ row_order={current_row_order} not found in table {validation_table}")

            # Decide whether skip-ahead should drop groups (only when sppTF == 1)
            should_drop = (validation_type in [1, 2, 3, 4] and sppTF_val == 1)

            if should_drop:
                # mutate df to remove the whole location/filename group
                if validation_type in [1, 2]:
                    df = df[df['location'] != current_location].reset_index(drop=True)
                elif validation_type in [3, 4]:
                    df = df[df['filename'] != current_filename].reset_index(drop=True)

                update_widget_bounds()

                if df.empty:
                    output.clear_output()
                    with output:
                        print("✅ No more labels to validate.")
                    conn.close()
                    return

                # Jump to the first row of the next location group (or next filename)
                next_loc = df.iloc[0]['location']
                new_index = df[df['location'] == next_loc].index[0]
                # use safe setter to avoid re-entrancy
                safe_set_idx(new_index)

            elif skip_to == "next":
                # If we're at last row and user clicked next/wrong, finish
                if i >= idx_box.max:
                    output.clear_output()
                    with output:
                        print("✅ No more labels to validate.")
                    conn.close()
                    return
                else:
                    safe_set_idx(min(i + 1, idx_box.max))

            elif skip_to == "prev":
                safe_set_idx(max(i - 1, 0))

            elif skip_to is None:
                # redraw current index
                plot_spectrogram_and_audio(idx_box.value)

    # Safe setter to avoid observer re-entry
    def safe_set_idx(v):
        busy['flag'] = True
        try:
            idx_box.value = v
        finally:
            busy['flag'] = False
            plot_spectrogram_and_audio(idx_box.value)

    # Set up functions that buttons run
    def on_prev(b): safe_set_idx(max(idx_box.value - 1, 0))
    def on_next(b): 
        if idx_box.value >= idx_box.max:
            output.clear_output()
            with output:
                print("✅ No more labels to validate.")
            conn.close()
            return
        safe_set_idx(min(idx_box.value + 1, idx_box.max))
    def on_correct(b): update_row_tag(1, 1, skip_to="next")
    def on_wrong(b): update_row_tag(0, 1, skip_to="next")

    btn_prev = widgets.Button(description="◀️", tooltip="Previous label")
    btn_next = widgets.Button(description="▶️", tooltip="Next label")
    btn_correct = widgets.Button(description="Correct ID", button_style="success")
    btn_wrong = widgets.Button(description="Wrong ID", button_style="danger")
    btn_quit = widgets.Button(description="Quit", button_style="warning")

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_correct.on_click(on_correct)
    btn_wrong.on_click(on_wrong)

    # Quit handler: close DB and disable UI
    def on_quit(b):
        try:
            conn.close()
        except Exception:
            pass
        btn_prev.disabled = btn_next.disabled = btn_correct.disabled = btn_wrong.disabled = btn_quit.disabled = True
        with output:
            clear_output()
            print("Session closed. DB connection closed.")
        with status_output:
            clear_output()

    btn_quit.on_click(on_quit)

    # Layout tweaks
    btn_prev.layout = widgets.Layout(width='50px')
    btn_next.layout = widgets.Layout(width='50px')
    btn_correct.layout = widgets.Layout(width='100px')
    btn_wrong.layout = widgets.Layout(width='100px')
    btn_quit.layout = widgets.Layout(width='80px')
    idx_box.layout = widgets.Layout(width='200px')

    # Observe idx_box safely
    def _idx_observer(change):
        if busy['flag']:
            return
        # only respond to value changes
        if change['name'] == 'value':
            plot_spectrogram_and_audio(change['new'])

    idx_box.observe(_idx_observer, names='value')

    # on_row_box_change: robust to df updates, uses safe_set_idx
    def on_row_box_change(change):
        if busy['flag']:
            return
        if change['name'] == 'value' and change['new'] is not None:
            target_order = change['new']
            matches = df.index[df['row_order'] == target_order].tolist()
            if matches:
                safe_set_idx(matches[0])
            else:
                print(f"❌ row_order {target_order} not found in DataFrame.")

    row_box.observe(on_row_box_change, names='value')

    # Setup user interface
    ui = widgets.VBox([
        widgets.HBox([btn_prev, btn_next, btn_correct, btn_wrong, btn_quit, row_box]),
        status_output,
        output
    ])

    # initial bounds update and first plot
    update_widget_bounds()
    plot_spectrogram_and_audio(idx_box.value)
    display(ui)

######################################################################################################################    
def visualize_scores(database_name, validation_table, class_code):
    # This is how you share data between two data
    shared_state = {'df_val': None}

    # Load data
    conn = sqlite3.connect(database_name)
    df_valmeta = pd.read_sql_query(
        f'SELECT * FROM "{validation_table}" WHERE class_code = ? ORDER BY filename',
        conn, params=(class_code,)
    )
    df_valmeta = df_valmeta.set_index("row_order")
    filenames_df = pd.read_sql_query(
        "SELECT DISTINCT filename FROM hawkears_results WHERE class_code = ? ORDER BY filename",
        conn, params=(class_code,)
    )
    conn.close()

    # Ensure no duplicate columns
    df_valmeta = df_valmeta.loc[:, ~df_valmeta.columns.duplicated()]

    filenames_list = filenames_df['filename'].tolist()
    if not filenames_list:
        print(f"No filenames found in hawkears_results for species code: {class_code}.")
        return

    # Widgets
    idx_box = widgets.BoundedIntText(value=0, min=0, max=len(filenames_list) - 1, description="File index:", layout=widgets.Layout(width='150px'))
    btn_prev = widgets.Button(description="◀️ Prev", layout=widgets.Layout(width='80px'))
    btn_next = widgets.Button(description="Next ▶️", layout=widgets.Layout(width='80px'))
    btn_load_audio = widgets.Button(description="🎧 Listen to entire audio file", button_style='primary', layout=widgets.Layout(width='200px'))
    label = widgets.HTML(value="", layout=widgets.Layout(height='25px', margin='5px 0', min_width='300px'))
    audio_output = widgets.Output(layout=widgets.Layout(border='1px solid #ccc', min_height='80px', width='100%', margin='10px 0'))
    plot_output = widgets.Output(layout=widgets.Layout(border='2px solid green', min_height='300px', width='100%'))

    # Plot function
    def plot_for_index(index):
        fname = filenames_list[index]
        conn = sqlite3.connect(database_name)
        df_all = pd.read_sql_query(
            "SELECT filename, start_time, score FROM hawkears_results WHERE filename = ? AND class_code = ?",
            conn, params=(fname, class_code)
        )
        df_val = pd.read_sql_query(
            f'SELECT filename, start_time, score, sppTF, filechkTF, rank, filepath FROM "{validation_table}" WHERE filename = ? AND class_code = ?',
            conn, params=(fname, class_code)        
        )
        shared_state['df_val'] = df_val
        conn.close()

        with plot_output:
            clear_output(wait=True)
            if df_all.empty:
                print(f"⚠️ No detections for {fname} and species code: {class_code}")
                return

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(df_all['start_time'], df_all['score'], color='lightgray', label='All detections')

            if not df_val.empty:
                unchecked = df_val[df_val['filechkTF'] == 0]
                wrong = df_val[(df_val['filechkTF'] == 1) & (df_val['sppTF'] == 0)]
                correct = df_val[(df_val['filechkTF'] == 1) & (df_val['sppTF'] == 1)]

                if not unchecked.empty:
                    ax.scatter(unchecked['start_time'], unchecked['score'], color='blue', s=100)
                    for _, row in unchecked.iterrows():
                        if pd.notna(row['rank']):
                            ax.text(row['start_time'] + 10, row['score'], f"R{int(row['rank'])}", fontsize=9)

                if not wrong.empty:
                    ax.scatter(wrong['start_time'], wrong['score'], color='red', marker='x', s=100)
                    for _, row in wrong.iterrows():
                        if pd.notna(row['rank']):
                            ax.text(row['start_time'] + 10, row['score'], f"R{int(row['rank'])}", fontsize=9)

                for _, row in correct.iterrows():
                    ax.text(row['start_time'], row['score'], '✓', fontsize=12, color='green', ha='center')
                    if pd.notna(row['rank']):
                        ax.text(row['start_time'] + 10, row['score'], f"R{int(row['rank'])}", fontsize=9)

            ax.set_title(f"{Path(fname).name} ({class_code})")
            ax.set_xlabel("Start Time (s)")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            plt.tight_layout()
            plt.show()        

    # Audio load button callback
    def on_load_audio_clicked(b):
        with audio_output:
            clear_output(wait=True)
            df_val = shared_state.get('df_val')
            if df_val is not None and not df_val.empty and 'filepath' in df_val.columns:
                audio_file = df_val.iloc[0]['filepath']
                if audio_file and Path(audio_file).is_file():
                    print("⏳ Loading audio, please wait...")
                    clear_output(wait=True)
                    display(Audio(str(audio_file)))
                else:
                    print(f"⚠️ Audio file not found:\n{audio_file}")
            else:
                print("⚠️ No valid filepath found for audio.")

    # Navigation callbacks
    def on_prev(b): idx_box.value = max(idx_box.value - 1, 0)
    def on_next(b): idx_box.value = min(idx_box.value + 1, idx_box.max)
    
    # Set button functionality
    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_load_audio.on_click(on_load_audio_clicked)
    idx_box.observe(lambda change: plot_for_index(change['new']), names='value')

    # Cascading filter update functions
    def update_sdate_options():
        selected_locations = location_select.value
        if not selected_locations:
            sdate_select.options = []
            return

        filtered = df_valmeta[df_valmeta['location'].isin(selected_locations)]
        sdate_options = sorted(filtered['sdate'].dropna().unique())
        sdate_select.options = sdate_options

        if sdate_options:
            sdate_select.value = (sdate_options[0],)

    def update_stime_options():
        selected_locations = location_select.value
        selected_sdates = sdate_select.value

        if not selected_locations or not selected_sdates:
            stime_select.options = []
            return

        filtered = df_valmeta[
            (df_valmeta['location'].isin(selected_locations)) &
            (df_valmeta['sdate'].isin(selected_sdates))
        ]
        stime_options = sorted(filtered['stime'].dropna().unique())
        stime_select.options = stime_options

        if stime_options:
            stime_select.value = (stime_options[0],)

    def jump_to_filename(*args):
        selected_locs = location_select.value
        selected_dates = sdate_select.value
        selected_times = stime_select.value
        if selected_locs and selected_dates and selected_times:
            match = df_valmeta[
                df_valmeta['location'].isin(selected_locs) &
                df_valmeta['sdate'].isin(selected_dates) &
                df_valmeta['stime'].isin(selected_times)
            ]
            if not match.empty:
                fname = match.iloc[0]['filename']
                if fname in filenames_list:
                    idx_box.value = filenames_list.index(fname)

     # Select multiple cascading filters with explicit layout and initial values
    location_options = sorted(df_valmeta['location'].dropna().unique())
    location_select = widgets.SelectMultiple(
        options=location_options,
        value=(location_options[0],) if location_options else (),
        description='Location:',
        rows=6,
        layout=widgets.Layout(border='1px solid gray', min_height='120px', width='250px')
    )
    sdate_select = widgets.SelectMultiple(
        options=[],
        description='Date:',
        rows=6,
        layout=widgets.Layout(border='1px solid gray', min_height='120px', width='250px')
    )
    stime_select = widgets.SelectMultiple(
        options=[],
        description='Time:',
        rows=6,
        layout=widgets.Layout(border='1px solid gray', min_height='120px', width='250px')
    )

    # Attach observers for cascading
    location_select.observe(lambda change: [update_sdate_options(), update_stime_options(), jump_to_filename()], names='value')
    sdate_select.observe(lambda change: [update_stime_options(), jump_to_filename()], names='value')
    stime_select.observe(lambda change: jump_to_filename(), names='value')

    # UI layout
    filter_box = widgets.HBox([location_select, sdate_select, stime_select], layout=widgets.Layout(margin='10px 0'))
    nav_box = widgets.HBox([btn_prev, btn_next, btn_load_audio], layout=widgets.Layout(margin='10px 0', align_items='center'))
    main_vbox = widgets.VBox([filter_box, nav_box, plot_output, audio_output], layout=widgets.Layout(width='900px', min_height='400px', border='1px solid #ddd', padding='10px', overflow='visible'))

    # Initialize cascading filters and plot
    update_sdate_options()
    update_stime_options()
    jump_to_filename()
    plot_for_index(idx_box.value)

    # Display UI
    display(main_vbox)

######################################################################################################################
def visualize_spectrogram(
    database_name,
    validation_table,
    buffer=0.0,
    min_freq=1000,
    max_freq=10000,
    n_fft=512,
    hop_length=256,
    target_sr=16000,
    use_setduration=False,
    set_duration=10.0,
    shared_state=None,
    filename_changed_callback=None
):
    import sqlite3
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, Audio, Markdown, clear_output, HTML
    import librosa
    import librosa.display
    import IPython.display as ipd

    status_output = widgets.Output()

    database_name = str(Path(database_name).resolve())
    conn = sqlite3.connect(database_name)

    # Determine validation type first (single value expected)
    try:
        raw_df = pd.read_sql_query(f'SELECT * FROM "{validation_table}" ORDER BY row_order', conn)
        vt_unique = raw_df['validation_type'].dropna().unique()
        if len(vt_unique) == 0:
            validation_type = 7
            print("⚠️ No 'validation_type' listed in table; defaulting to 7 (alllabels).")
        else:
            validation_type = int(vt_unique[0])
            if len(vt_unique) > 1:
                print("⚠️ Multiple validation_type values found; using the first:", validation_type)
    except Exception as e:
        validation_type = 7
        print("⚠️ Failed to read validation_type; defaulting to 7. Error:", e)
        raw_df = pd.read_sql_query(f'SELECT * FROM "{validation_table}" ORDER BY row_order', conn)

    # Build SQL query with exclusion based on validation type
    if validation_type in [1, 2]:
        sql = f"""
        SELECT * FROM "{validation_table}"
        WHERE location NOT IN (
            SELECT location FROM "{validation_table}"
            GROUP BY location
            HAVING SUM(filechkTF) = COUNT(*)
        )
        ORDER BY row_order
        """
    elif validation_type in [3, 4]:
        sql = f"""
        SELECT * FROM "{validation_table}"
        WHERE filename NOT IN (
            SELECT filename FROM "{validation_table}"
            GROUP BY filename
            HAVING SUM(filechkTF) = COUNT(*)
        )
        ORDER BY row_order
        """
    else:
        sql = f'SELECT * FROM "{validation_table}" ORDER BY row_order'

    df = pd.read_sql_query(sql, conn)

    if df.empty:
        print(f"❌ No more labels to validate.")
        conn.close()
        return

    # Further filter in pandas based on validation type
    if validation_type in [1, 2]:
        validated = df[df['sppTF'] == 1]['location'].unique()
        df = df[~df['location'].isin(validated)]
    elif validation_type in [3, 4]:
        validated = df[df['sppTF'] == 1]['filename'].unique()
        df = df[~df['filename'].isin(validated)]
    elif validation_type in [5, 6, 7]:
        pass
    else:
        conn.close()
        raise ValueError(f"Unsupported validation_type: {validation_type}")

    df = df.reset_index(drop=True)

    if df.empty:
        print("✅ No labels to validate.")
        conn.close()
        return

    total_rows = len(df)

    # Tracking widgets
    idx_box = widgets.BoundedIntText(value=0, min=0, max=total_rows - 1, step=1, description="Label index:")
    # Row box shows *row_order* values (DB IDs)
    row_box = widgets.BoundedIntText(value=int(df.loc[0, 'row_order']),
                                     min=int(df['row_order'].min()),
                                     max=int(df['row_order'].max()), step=1, description="Row order #:")
    output = widgets.Output()

    # Busy flag to avoid re-entrant callbacks
    busy = {'flag': False}

    def plot_spectrogram_and_audio(i):
        # guard: i must be in-range for current df
        output.clear_output()
        with output:
            if len(df) == 0:
                print("✅ No more labels to validate.")
                return

            if i is None or i < 0 or i >= len(df):
                print("⚠️ Index out of range for current DataFrame.")
                return

            # Capture row data early so later mutations don't change what we display
            row = df.iloc[i].copy()
            try:
                # Temporarily set busy while updating the displayed row_order
                busy['flag'] = True
                row_box.value = int(row['row_order'])
            finally:
                busy['flag'] = False

            filepath = row['filepath']
            start_time = float(row['start_time']) if pd.notnull(row['start_time']) else 0.0
            end_time = float(row['end_time']) if pd.notnull(row['end_time']) else start_time + 1.0

            # Determine audio length in a lightweight way if needed
            audio_length = None
            if use_setduration:
                try:
                    import soundfile as sf
                    info = sf.info(filepath)
                    audio_length = info.frames / info.samplerate
                except Exception:
                    # fallback to librosa for length, but only load header if possible
                    try:
                        full_y, full_sr = librosa.load(filepath, sr=target_sr, duration=0.1)
                        audio_length = librosa.get_duration(filename=filepath)
                    except Exception:
                        audio_length = None

            if use_setduration:
                duration = min(set_duration, audio_length if audio_length is not None else set_duration)
                offset = 0.0
            else:
                duration = end_time - start_time
                offset = max(0, start_time - buffer)
                duration = duration + 2 * buffer

            # Clip duration to audio length if available
            if audio_length is not None:
                if offset + duration > audio_length:
                    duration = max(0, audio_length - offset)

            # To create counters
            loc = row['location']
            fn = row['filename']
            df_loc = df[df['location'] == loc]
            df_fn = df[df['filename'] == fn]

            # Calculate 1-based row number inside location group (relative to df)
            try:
                pos = (df_loc['row_order'] == row['row_order']).to_numpy().nonzero()[0]
                row_in_loc = int(pos[0]) + 1 if pos.size > 0 else None
            except Exception:
                row_in_loc = None

            try:
                posf = (df_fn['row_order'] == row['row_order']).to_numpy().nonzero()[0]
                row_in_fn = int(posf[0]) + 1 if posf.size > 0 else None
            except Exception:
                row_in_fn = None

            n_loc = len(df_loc)
            v_loc = int(df_loc['sppTF'].sum()) if 'sppTF' in df_loc.columns else 0
            n_fn = len(df_fn)
            v_fn = int(df_fn['sppTF'].sum()) if 'sppTF' in df_fn.columns else 0

            # Print statements on status of processing
            display(Markdown(f"""
  📍 **Location:** `{loc}` — Filename {row_in_loc if row_in_loc is not None else '?'} of {n_loc}\n
  📄 **Filename:** `{Path(fn).name}` — Label {row_in_fn if row_in_fn is not None else '?'} of {n_fn}
- Validation type: `{row.get('validation_type', validation_type)}`
- Species validating: `{row.get('class_code', '')}`
- HawkEars score: `{row.get('score', 0.0):.2f}`
- Detection time: `{start_time:.2f}s – {end_time:.2f}s`
- Audio segment: `{offset:.2f}s – {offset + duration:.2f}s`
- Label rank: `{row.get('rank', '')}`
"""))

            # Plot spectrogram + audio
            try:
                y, sr = librosa.load(filepath, sr=target_sr, offset=offset, duration=duration)
                S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
                S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

                plt.figure(figsize=(6, 3))
                librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
                # Ensure ylim is within sensible bounds
                plt.ylim(max(0, min_freq), min(max_freq, sr / 2))
                plt.colorbar(format="%+2.0f dB")
                plt.title("Spectrogram")

                if not use_setduration:
                    # show detection window
                    plt.axvline(x=buffer, color='lime', linestyle='--', linewidth=1.5)
                    plt.axvline(x=buffer + (end_time - start_time), color='red', linestyle='--', linewidth=1.5)

                plt.tight_layout()
                plt.show()

                display(HTML("<br><br>"))
                ipd.display(ipd.Audio(y, rate=sr))

            except Exception as e:
                print(f"⚠️ Failed to load/plot audio: {e}")

    def update_widget_bounds():
        # call whenever df changes
        idx_box.max = max(0, len(df) - 1)
        if len(df) > 0 and 'row_order' in df.columns:
            try:
                row_box.min = int(df['row_order'].min())
                row_box.max = int(df['row_order'].max())
            except Exception:
                row_box.min = 0
                row_box.max = max(0, len(df) - 1)

    def update_row_tag(sppTF_val, filechkTF_val, skip_to="next"):
        nonlocal df
        i = idx_box.value
        if i >= len(df):
            return

        # Capture current row fields *before* we mutate df
        row = df.iloc[i].copy()
        current_row_order = int(row['row_order'])
        current_location = row['location']
        current_filename = row['filename']

        cursor = conn.cursor()

        # Take a snapshot of the current ordering as row_order list (preserves visible order)
        current_ordered_row_orders = list(df['row_order'].astype(int).tolist())

        # Also map groups in order so we can pick the "next group" reliably
        if validation_type in [1, 2]:
            current_group_key = 'location'
            current_group_val = current_location
            ordered_groups = list(pd.Index(df['location']).unique())
        else:
            current_group_key = 'filename'
            current_group_val = current_filename
            ordered_groups = list(pd.Index(df['filename']).unique())

        post_action_idx = None
        finished = False

        with status_output:
            status_output.clear_output(wait=True)

            # DB update
            try:
                cursor.execute(f"""
                    UPDATE "{validation_table}"
                    SET sppTF = ?, filechkTF = ?
                    WHERE row_order = ?
                """, (sppTF_val, filechkTF_val, current_row_order))
                conn.commit()
            except Exception as e:
                print("⚠️ DB update failed:", e)
                return

            # Confirm update
            try:
                cursor.execute(f"""
                    SELECT sppTF, filechkTF FROM "{validation_table}" WHERE row_order = ?
                """, (current_row_order,))
                updated_values = cursor.fetchone()
            except Exception as e:
                updated_values = None
                print("⚠️ DB fetch after update failed:", e)

            if updated_values:
                print(f"🔗 The previous label with row_order={current_row_order} in {validation_table} was set to: sppTF={updated_values[0]}, filechkTF={updated_values[1]}")
            else:
                print(f"❌ row_order={current_row_order} not found in table {validation_table}")

            # Decide whether skip-ahead should drop groups (only when sppTF == 1)
            should_drop = (validation_type in [1, 2, 3, 4] and sppTF_val == 1)

            # --- Determine target row_order index BEFORE mutating df ---
            try:
                pos_in_order = current_ordered_row_orders.index(current_row_order)
            except ValueError:
                pos_in_order = None

            if should_drop:
                # Determine next group according to original ordering
                next_group_val = None
                if pos_in_order is not None:
                    try:
                        current_group_pos = ordered_groups.index(current_group_val)
                    except ValueError:
                        current_group_pos = None

                    if current_group_pos is not None and (current_group_pos + 1) < len(ordered_groups):
                        next_group_val = ordered_groups[current_group_pos + 1]
                    else:
                        next_group_val = None

                # Now mutate df to remove the whole group
                if validation_type in [1, 2]:
                    df = df[df['location'] != current_location].reset_index(drop=True)
                else:
                    df = df[df['filename'] != current_filename].reset_index(drop=True)

                update_widget_bounds()

                if df.empty:
                    output.clear_output()
                    with output:
                        print("✅ No more labels to validate.")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    finished = True
                else:
                    # Find the first row_order in the NEW df that belongs to next_group_val
                    if next_group_val is not None and next_group_val in df[current_group_key].values:
                        # choose first row of that group
                        candidate_ro = int(df[df[current_group_key] == next_group_val].iloc[0]['row_order'])
                        # map candidate row_order to its current df index
                        try:
                            post_action_idx = int(df.index[df['row_order'] == candidate_ro][0])
                        except Exception:
                            post_action_idx = None
                    else:
                        # fallback: try to pick the next row_order after current_row_order in the remaining df
                        remaining_row_orders = sorted([int(x) for x in df['row_order'].astype(int).tolist()])
                        # pick the smallest remaining row_order that is greater than current_row_order, else first available
                        larger = [r for r in remaining_row_orders if r > current_row_order]
                        candidate_ro = larger[0] if larger else remaining_row_orders[0]
                        try:
                            post_action_idx = int(df.index[df['row_order'] == candidate_ro][0])
                        except Exception:
                            post_action_idx = 0

            elif skip_to == "next":
                # We didn't drop a group (Wrong ID or simple next). Advance by the next row in the original ordering.
                # Determine the row_order of the next item in the snapshot ordering
                candidate_ro = None
                if pos_in_order is not None and (pos_in_order + 1) < len(current_ordered_row_orders):
                    candidate_ro = current_ordered_row_orders[pos_in_order + 1]
                else:
                    candidate_ro = None

                # Now, after DB update, map candidate_ro to new df index (df hasn't changed here)
                if candidate_ro is not None and candidate_ro in df['row_order'].values:
                    post_action_idx = int(df.index[df['row_order'] == candidate_ro][0])
                else:
                    # fallback: if candidate_ro isn't present (rare), then try advancing by position i+1
                    if i + 1 < len(df):
                        post_action_idx = i + 1
                    else:
                        # at end -> finish
                        finished = True

            elif skip_to == "prev":
                # Go to previous row in the original ordering
                candidate_ro = None
                if pos_in_order is not None and pos_in_order - 1 >= 0:
                    candidate_ro = current_ordered_row_orders[pos_in_order - 1]
                if candidate_ro is not None and candidate_ro in df['row_order'].values:
                    post_action_idx = int(df.index[df['row_order'] == candidate_ro][0])
                else:
                    post_action_idx = max(i - 1, 0)

            elif skip_to is None:
                # redraw current index (no navigation)
                post_action_idx = i

        # End of status_output context — perform navigation AFTER printing
        if finished:
            return

        if post_action_idx is not None:
            # Final safety: clamp
            post_action_idx = max(0, min(post_action_idx, max(0, len(df) - 1)))
            # If setting to same index, try to bump by one (defensive)
            if post_action_idx == idx_box.value and idx_box.value < max(0, len(df) - 1):
                post_action_idx = idx_box.value + 1
            safe_set_idx(post_action_idx)
        else:
            # If we have nothing to go to, redraw current
            plot_spectrogram_and_audio(idx_box.value)

    # Safe setter to avoid observer re-entry
    def safe_set_idx(v):
        busy['flag'] = True
        try:
            idx_box.value = v
        finally:
            busy['flag'] = False
            plot_spectrogram_and_audio(idx_box.value)

    # Set up functions that buttons run
    def on_prev(b): safe_set_idx(max(idx_box.value - 1, 0))
    def on_next(b):
        if idx_box.value >= idx_box.max:
            output.clear_output()
            with output:
                print("✅ No more labels to validate.")
            conn.close()
            return
        safe_set_idx(min(idx_box.value + 1, idx_box.max))
    def on_correct(b): update_row_tag(1, 1, skip_to="next")
    def on_wrong(b): update_row_tag(0, 1, skip_to="next")

    btn_prev = widgets.Button(description="◀️", tooltip="Previous label")
    btn_next = widgets.Button(description="▶️", tooltip="Next label")
    btn_correct = widgets.Button(description="Correct ID", button_style="success")
    btn_wrong = widgets.Button(description="Wrong ID", button_style="danger")
    btn_quit = widgets.Button(description="Quit", button_style="warning")

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_correct.on_click(on_correct)
    btn_wrong.on_click(on_wrong)

    # Quit handler: close DB and disable UI
    def on_quit(b):
        try:
            conn.close()
        except Exception:
            pass
        btn_prev.disabled = btn_next.disabled = btn_correct.disabled = btn_wrong.disabled = btn_quit.disabled = True
        with output:
            clear_output()
            print("Session closed. DB connection closed.")
        with status_output:
            clear_output()

    btn_quit.on_click(on_quit)

    # Layout tweaks
    btn_prev.layout = widgets.Layout(width='50px')
    btn_next.layout = widgets.Layout(width='50px')
    btn_correct.layout = widgets.Layout(width='100px')
    btn_wrong.layout = widgets.Layout(width='100px')
    btn_quit.layout = widgets.Layout(width='80px')
    idx_box.layout = widgets.Layout(width='200px')

    # Observe idx_box safely
    def _idx_observer(change):
        if busy['flag']:
            return
        # only respond to value changes
        if change['name'] == 'value':
            plot_spectrogram_and_audio(change['new'])

    idx_box.observe(_idx_observer, names='value')

    # on_row_box_change: robust to df updates, uses safe_set_idx
    def on_row_box_change(change):
        if busy['flag']:
            return
        if change['name'] == 'value' and change['new'] is not None:
            target_order = change['new']
            matches = df.index[df['row_order'] == target_order].tolist()
            if matches:
                safe_set_idx(matches[0])
            else:
                print(f"❌ row_order {target_order} not found in DataFrame.")

    row_box.observe(on_row_box_change, names='value')

    # Setup user interface
    ui = widgets.VBox([
        widgets.HBox([btn_prev, btn_next, btn_correct, btn_wrong, btn_quit, row_box]),
        status_output,
        output
    ])

    # initial bounds update and first plot
    update_widget_bounds()
    plot_spectrogram_and_audio(idx_box.value)
    display(ui)
######################################################################################################################
