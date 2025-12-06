# -*- coding: utf-8 -*-
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
Tip: be mindful about where you save your input/output files to, there are many of them and it will make your life much easier if you know where they
are and they are organized well.

See Erin's google collab notebook for additional details.

Important: Run the embHEtools.py script FIRST
"""
#Run HawkEars via embHEtools ##########################################################################################################
# Import required packages
import librosa
import sys # To work with directory of your computer
import os # To work with operating system of your computer
#Set working directory where you have put HawkEars
#os.chdir("C:/Users/bayne/HawkEars_v108")
os.chdir("C:/Users/EdgarM/Desktop/Localization/boreal-localization")


import sqlite3 # Database used to store results
import pandas as pd # Query tools
import embHEtools # Functions built by Erin Bayne to run validation code


# Name project and select species you want HawkEars to detect
projectname = 'boreallocalization_ewpw'
spp_to_include = ['EWPW'] # NOTE in spp_to_include YOU COULD USE ANY 1 OF['OVEN', 'Ovenbird', 'ovenbi1'])  # can be COMMON_NAME, CODE4, or CODE6 [indicate a dictionary and are required]
# If spp_to_include is empty then all species HawkEars knows are processed. If you want a subset type "OVEN", "WTSP", "CHSP" etc)

# Desired recordings to process from Cirrus
#orgid = "CWS"
#projid = "EGA"
#yearid = 2025
#siteid = "SNAS"
runid = f"{projectname}" # Name of processing run
audio_dir ="E:/BAR-LT_LocalizationProject/localizationtrim" #Run all 49 ARUs

# MEG PUT THESE INSANE TIME
# Desired dates and times of recordings to select from server
min_mmdd = 101 # Do not include leading zeros Example: 101 = January 1st
max_mmdd = 1231 # Do not include leading zeros Example: 1231 = December 31st
min_time = 0 # Note do NOT include the leading zero that comes off an ARU for time. 0 is midnight not 000000. 50000 is 5 AM
max_time = 235959 # Note do NOT include the leading zero that comes off an ARU for time. 235999 is millisecond right before midnight

# Desired HawkEars settings. You can further select when doing validation
cutoff = 0.7 # Cutoff score you want HawkEars to run at. Default = 0.8 in HawkEars. I used 0.01 as I want every 3 seconds to have a score
overlap = 0 # Amount sliding window overlaps when searching for signal. Searches 3 second windows with X amount of amount in window. Less overlap, faster. Potentially less accurate for some questions. Default = 1.5
merge = 0 # If 1 (true) then merge is on. If 0 (false) then you get individual tags.# Merge example: With default threshold of .8 and your species score >= .8 in every scanned segment for 30 seconds and one label has score = .97 it creates a long label that is 30-seconds long with a score of .97.

# Locations of code
#python = fr"C:/Users/bayne/HawkEars_v108/venv/Scripts/python.exe" # To tell Colab or Jupyter notebook where Python is located because we have to call subprocess
#hawkears_code = fr"C:/Users/bayne/HawkEars_v108/analyze.py"
#he_allspp_cv = "C:/Users/bayne/HawkEars_v108/data/species_codes_morethanbirds.csv"
python = "C:/Users/EdgarM/.conda/envs/boreal_loc/python.exe" # To tell Colab or Jupyter notebook where Python is located because we have to call subprocess
hawkears_code = "C:/Users/EdgarM/Desktop/Localization/boreal-localization/HawkEars/analyze.py"
he_allspp_cv = "C:/Users/EdgarM/Desktop/Localization/boreal-localization/HawkEars/data/species_codes.csv"

# Locations of input files required by HawkEars
input_dir = f"C:/Users/EdgarM/Desktop/Localization/hawkears_inputs/{runid}" #After crawling server this is the folder where text files from each run are going to be stored
fileall = f"{input_dir}/{runid}_files.csv" # Location of all files found in the selected audio_dir
filecount = f"{input_dir}/{runid}_countfiles.csv" # Location of counts of wac/wav files per location
fileformat = f"{input_dir}/{runid}_formatfiles.csv" # Location of all files, formatted for selection
filesubset = f"{input_dir}/{runid}_subsetfiles.csv" # Location of selected recordings based on date/time
filemerge = f"{input_dir}/{runid}_mergefiles.csv" # Location of merged count and subset data
filelistoflists = fr"{input_dir}/{runid}_listoflists.csv" #Location of the list of files you are going to send to HawkEars the directory they are in

# Locations of output files from HawkEars
tag_dir = f"C:/Users/EdgarM/Desktop/Localization/hawkears_tags/{runid}" # Tags created by HawkEars in raw CSV
foldertags = f"{tag_dir}/{runid}" # Location of where HawkEars writes tags in CSV format
database_name = f"C:/Users/EdgarM/Desktop/Localization/hawkears_{projectname}_database.db" # Where results are stored in addition to raw CSV
rootdir_subset = "C:/Users/EdgarM/Desktop/Localization/hawkears_inputs" # Loops through this folder structure to find every file processed you selected for HawkEars to run and writes the path to database

# Location of logs
os.makedirs(foldertags, exist_ok=True)
log_file_path = fr"{foldertags}/{runid}_output_log.txt"  # Log file location
log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")  # Line-buffered

# Workflow
# Send logs to screen and text file
#from embHEtools import Tee
#Tee = Tee(log_file_path)
#sys.stdout = Tee
#sys.stderr = Tee

# Settings used for this run that are tracked in the log
print(fr"Focal species are: {spp_to_include}")
print(fr"Minimum date of year is: {min_mmdd}")
print(fr"Maximum date of year is: {max_mmdd}")
print(fr"Minimum time of day is: {min_time}")
print(fr"Maximum time of day is: {max_time}")
print(fr"HE score cutoff value is: {cutoff}")
print(fr"Overlap value is: {overlap}")
print(fr"Merge value is: {merge}")



# Each of these steps can be used independently as you see fit

# 0. Sets species you want HawkEars to scan for in recordings
embHEtools.hawkears_selectspp(he_allspp_cv, spp_to_include)

# 1. Scans all folders on your storage space recursively and writes paths to wav and wac files. Only wav processed currently. Stored as CSV.
embHEtools.audiolist_create(audio_dir, fileall)

# 2. Create a list of all locations and count # of wac vs wav files. Only wav processed currently. Stored as CSV.
embHEtools.audiolist_count(fileall, filecount)

# 3. Formats the full list of audio files to allow selection of recordings with certain properties (retains path). Stored as CSV
embHEtools.audiolist_format(fileall, fileformat)

# 4. Select the mmdd (aka recording_date without year) and recording_time(s) you want. This subset stored as CSV
embHEtools.audiolist_filter(fileformat, min_mmdd, max_mmdd, min_time, max_time, filesubset)

# 5. Joins count data to subset data and saves to filemerge. Stored as CSV
embHEtools.audiolist_join(filecount, filesubset, filemerge)

# 6. Create individual HawkEars lists to run for each location. Stored as CSV
embHEtools.audiolist_filelist(filemerge, input_dir)

# 7. Make a master list of filelists. Filelists are the recordings that you select for Hawk Ears to process. Stored as CSV
embHEtools.audiolist_listoflists(filemerge, filelistoflists, input_dir, foldertags)

# 8. Run HawkEars and populates a SQL database
embHEtools.hawkears_run(database_name, filelistoflists, python, hawkears_code, cutoff, overlap, merge)

# 9. Create/update SQL database table that has all required fields for fields created by Hawk Ears. Called hawkears_results in your SQL database
embHEtools.hawkears_dbaseupdate(database_name)

#10. Create/ update database table that has a list of all files that you attempted to run and the path to those files. Called all_subsetfiles in your SQLdatabase
embHEtools.hawkears_filesrun(rootdir_subset, database_name)


# Shut down log system
#sys.stdout.flush()
#sys.stderr.flush()
#Tee.close()

# Completed automated recognition workflow
print("✅ Processing complete! Log file saved to:", log_file_path)

#Create validation tables ##################################################################################################################################
# Validation tables are created if they do not exist. If they exist, they are updated with data that is not duplicates.
# If duplicates (i.e you reran same files using same settings the labels are not written to the database because they exist)

# Database and table names
database_name = r"C:/Users/EdgarM/Desktop/Localization/hawkears_boreallocalization_ewpw_database.db" # Name of the SQLite database to call hawkears_results from
conn = sqlite3.connect(database_name)

# See what tables exist
print(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn))

# How many labels did HawkEars write?
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM hawkears_results;", conn))

# How many subset files are tracked?
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM all_subsetfiles;", conn))

conn.close()

# Validation settings
#top_n = 1 # Number of best tags to keep in a recording (filename) for validation. All tags above cutoff are kept in hawkears_results table, this is just for validation. Default validation setting is all labels
min_spacing = 10 # Approximate number of seconds between the tags selected for validation. Goal here is to make sure validated songs are unique and not HawkEars scoring same song as the window slides.
validation_type = 3  # Choose from: 1=bestlocationlabel, 2=firstlocationlabel, 3=bestfilelabel, 4=firstfilelabel, 5=minmaxfilelabel, 6=gradientfilelabel, 7=alllabels, 8=firstlocationlabel_alltop_n
validation_table = (fr"vtbl{validation_type}") # Automatic naming
labelsperbin = 5 # For gradientfilelabel method (# 6)

# Create validation dataset
embHEtools.validate_maketables(database_name=database_name, validation_table=validation_table, min_spacing = min_spacing, validation_type=validation_type, labelsperbin=5)

# Validate labels ########################################################################################################################################

#Note: Erica (and ChatGPT) adapted this code from Erin's notebook so that it functions similarly in Spyder. It is a work in progress
import pandas as pd
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import sounddevice as sd  # pip install sounddevice

database_name = r"C:/Users/AlexE/Localization/hawkears_cclofollow_database.db"
class_code = "CCLO"      #species to validate
vtype = 3                #validation type. Choose from: 1=bestlocationlabel, 2=firstlocationlabel, 3=bestfilelabel, 4=firstfilelabel, 5=minmaxfilelabel, 6=gradientfilelabel, 7=alllabels, 8=firstlocationlabel_allranks

validation_table = f"vtbl{vtype}_{class_code}"

#Settings for visualizing spectrograms
buffer = 1          # seconds before/after label
min_freq = 2000     # Hz
max_freq = 7000     # Hz
n_fft = 1024        # Amount of information used to make spectrograms. Bigger numbers better resolution, slower draw time.
hop_length = 512    #
target_sr = 22000   #Sampling rate, based on max frequency you want to observe on spectrogram
use_setduration = False  #False means you use the start_time and end_time of individual HawkEars labels. True is intended to show longer periods, for example when HawkEars merge is on
set_duration = 60 # Show X second spectrogram if use_setduration==True. Otherwise shows HawkEars labels +/- buffer size

#Load validation data
conn = sqlite3.connect(database_name)
df = pd.read_sql_query(f"SELECT * FROM {validation_table}", conn)
conn.close()

# Add validation column if missing
if 'validation' not in df.columns:
    df['validation'] = ''

#Iterate through labels, interactively validate. Audio will play and then the spectrogram will appear. Type 'r' in the console
#no replay audio

print(f"\nStarting iterative validation of {len(df)} labels...\n")
print("Type 'y' = yes, 'n' = no, 'skip' = skip label, 'r' = replay, 'q' = quit.\n")

for idx, row in df.iterrows():
    print(f"\nLabel {idx+1}/{len(df)}")
    print(f"File: {row['filepath']}")
    print(f"Start: {row.get('start_time', 'N/A')}, End: {row.get('end_time', 'N/A')}")

    audio_path = Path(row['filepath'])

    if audio_path.exists():
        try:
            # Load WAV file
            fs, data = wavfile.read(str(audio_path))

            # If stereo, use first channel
            if len(data.shape) > 1:
                data = data[:, 0]

            # Determine audio segment boundaries
            start_time = row.get('start_time', 0)
            end_time   = row.get('end_time', None)

            if use_setduration:
                start_time_segment = start_time
                end_time_segment = start_time + set_duration
            else:
                start_time_segment = max(0, start_time - buffer)
                end_time_segment = end_time + buffer if end_time is not None else start_time + buffer

            start_sample = int(start_time_segment * fs)
            end_sample   = int(end_time_segment * fs)
            data_segment = data[start_sample:end_sample]

            # --- Plot spectrogram ---
            plt.figure(figsize=(10, 4))
            plt.specgram(
                data_segment,
                NFFT=n_fft,
                Fs=fs,
                noverlap=hop_length,
                cmap='viridis'
            )
            plt.ylim(min_freq, max_freq)
            plt.title(f"Spectrogram: {audio_path.name}")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.colorbar(label='Intensity [dB]')
            plt.show()

            #Prompt loop
            while True:
                print("Options: y = yes, n = no, skip = skip, r = replay, q = quit")
                user_input = input("Your choice: ").lower()

                if user_input == 'r':
                    print("▶️ Replaying audio...")
                    sd.play(data_segment, fs)
                    sd.wait()

                elif user_input in ['y', 'n', 'skip']:
                    df.at[idx, 'validation'] = user_input
                    break

                elif user_input == 'q':
                    print("\n🛑 Validation stopped by user.")
                    break

                else:
                    print("Invalid input. Use y, n, skip, r, or q.")

            if user_input == 'q':
                break

            # Play segment initially
            print("▶️ Playing audio...")
            sd.play(data_segment, fs)
            sd.wait()

        except Exception as e:
            print(f"⚠️ Audio error: {e}")

    else:
        print(f"⚠️ File not found: {audio_path}")

    if user_input == 'q':
        break

#Save results to csv
save_folder = Path(r"C:/Users/AlexE/Localization/validated_results")
save_folder.mkdir(parents=True, exist_ok=True)
save_path = save_folder / f"validated_labels_{class_code}_v{vtype}.csv"

df.to_csv(save_path, index=False)
print(f"\nResults saved to {save_path}")