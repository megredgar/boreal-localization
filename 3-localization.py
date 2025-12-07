# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:52:47 2025
@author: AlexE, adapted by Megan Edgar on December 10, 2025

# This script provides the code for localizing sounds from audio recordings using the opensoundscape library, developed by the Kitzes lab at the University of Pittsburg.
# For the source code and more detailed instructions, see their tutorial page (https://opensoundscape.org/en/latest/tutorials/acoustic_localization.html)
# See my RMarkdown document for some additional tips and tricks!
"""

import opensoundscape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import csv with ARU coordinates
aru_coords = pd.read_csv(
    "C:/Users/AlexE/OneDrive - EC-EC/Robinson,Barry (il _ he, him) (ECCC)'s files - Grassland Bird Monitoring/R Projects/Localization/data/aru_coords/gsa_noB1.csv",
    index_col=0)

#initialize a recorder array
from opensoundscape.localization import SynchronizedRecorderArray

array = SynchronizedRecorderArray(aru_coords)

#load csv of HawkEars detections
detections = pd.read_csv("C:/Users/AlexE/OneDrive - EC-EC/Localization/data/gsa_grsp_follow.csv")

#Add the start timestamp, adjusted to match the trimmed recordings (it should be the start time of your latest recording)
import pytz
from datetime import datetime, timedelta

local_timestamp = datetime(2025, 6, 18, 8, 3, 58)
local_timezone = pytz.timezone("America/Regina")
detections["start_timestamp"] = [
    local_timezone.localize(local_timestamp) + timedelta(seconds=s)
    for s in detections["start_time"]
]

#set four column multi-index expected for localization
detections = detections.set_index(['file', 'start_time', 'end_time', 'start_timestamp'])

#set parameters for localization
min_n_receivers = 3 #min number of recievers with detection
max_receiver_dist = 80 #max distance between recorders for estimating TDOA

#localize detections
position_estimates = array.localize_detections(
    detections,
    min_n_receivers=min_n_receivers,
    max_receiver_dist=max_receiver_dist
)


# Data Exploration ==========================================================================================================

#filter to a single species
#DEJU_positions = [
#    e
#    for e in position_estimates
#    if e.class_name == "DEJU"]

# Examine individual position estimates
example = position_estimates[1]
print(f"The start time of the detection: {example.start_timestamp}")
print(f"This is a detection of the class/species: {example.class_name}")
print(
    f"The duration of the time-window in which the sound was detected: {example.duration}"
)
print(f"The estimated location of the sound: {example.location_estimate}")
print(f"The receivers on which our species was detected: \n{example.receiver_files}")
print(f"The estimated time-delays of arrival: \n{example.tdoas}")
print(f"The normalized Cross-Correlation scores: \n{example.cc_maxs}")

plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.scatter(
    x=example.location_estimate[0],
    y=example.location_estimate[1],
    color="red",
    label=f"{example.class_name}",
)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.show()

#Assess the quality of the localization based on how well the spectrogram lines up
from opensoundscape import Spectrogram

audio_segments = example.load_aligned_audio_segments()
specs = [Spectrogram.from_audio(a).bandpass(8000, 12000) for a in audio_segments]
plt.pcolormesh(np.vstack([s.spectrogram for s in specs]), cmap="Greys")

example.residual_rms #check residual

#See all the position estimates for this localized event
#each estimate is generated using a different recorder as the reference unit

grsp = [
    e
    for e in position_estimates
    if e.class_name == example.class_name
    and e.start_timestamp == example.start_timestamp
]

# get the x-coordinates of the estimated locations
x_coords = [e.location_estimate[0] for e in grsp]
# get the y-coordinates of the estimated locations
y_coords = [e.location_estimate[1] for e in grsp]
# get the rms of residuals per event
rms = [e.residual_rms for e in grsp]
# plot the estimated locations, colored by the residuals
plt.scatter(
    x_coords,
    y_coords,
    c=rms,
    label="GRSP",
    alpha=0.4,
    edgecolors="black",
    cmap="jet",
)
cbar = plt.colorbar()
cbar.set_label("residual rms (meters)")
# plot the ARU locations
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
# make the legend appear outside of the plot
plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")

#Filter out bad localizations with high error (residul rms)

#to see a summary of the residuals:
residuals = [e.residual_rms for e in grsp]
min_rms = min(residuals)
max_rms = max(residuals)
mean_rms = np.mean(residuals)
median_rms = np.median(residuals)
lqt_rms = np.quantile(residuals, 0.25)
uqt_rms = np.quantile(residuals, 0.75)

print(f"Residual RMS (GRSP events):")
print(f"Min: {min_rms:.2f} m")
print(f"Max: {max_rms:.2f} m")
print(f"Mean: {mean_rms:.2f} m")
print(f"Median: {median_rms:.2f} m")
print(f"First Quartile (Q1): {lqt_rms}")
print(f"Third Quartile (Q3): {uqt_rms}")

#select threshold and filter
low_rms = [
    e for e in grsp if e.residual_rms < 35]  # get only the events with low residual rms

#plot the ARU locations
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
#plot the estimated locations
plt.scatter(
    [e.location_estimate[0] for e in low_rms],
    [e.location_estimate[1] for e in low_rms],
   edgecolor="black",
   label="GRSP",
)
#make the legend appear outside the plot
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")


#Plot average position for each time window =================================================================================================================
from collections import defaultdict

# filter to error threshold
rms_threshold = 20 # meters, residual RMS threshold

#get events
grsp_events = [
    e for e in position_estimates
    if e.class_name == "GRSP" and e.residual_rms < rms_threshold
]

#group by time bin
grouped_by_time = defaultdict(list)
for event in grsp_events:
    grouped_by_time[event.start_timestamp].append(event)

# can also set a minimum number of valid events per time window
min_events_per_window = 3

filtered_groups = {
    ts: events for ts, events in grouped_by_time.items()
    if len(events) >= min_events_per_window
}

#get average position per time window
avg_locations = []
timestamps = []

for timestamp, events in filtered_groups.items():
    x_avg = np.mean([e.location_estimate[0] for e in events])
    y_avg = np.mean([e.location_estimate[1] for e in events])
    avg_locations.append((x_avg, y_avg))
    timestamps.append(timestamp)

#split coordinate pairs into x and y positions
x_avg_coords, y_avg_coords = zip(*avg_locations)
    
#Optional: overlay with coordinates of known locations
import pandas as pd

#load csv
csv_path = "C:/Users/AlexE/OneDrive - EC-EC/Localization/data/GRSP_coords.csv"  # replace with your file path
known_locations = pd.read_csv(csv_path)

#extract coordinates
x_known = known_locations['x']
y_known = known_locations['y']

#define plot
plt.figure(figsize=(10, 8))
#plot ARU locations
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARUs", color="black")
#plot average positions
plt.scatter(x_avg_coords, y_avg_coords, color="red", label="Average Location/Time", alpha=0.6)
#plot known positions
plt.scatter(x_known, y_known, color='blue', marker='X', s=100, label='Known GRSP Points')
#set axis limits
plt.xlim(-109.334, -109.3305)
plt.ylim(50.6660,50.66925)
#labels
plt.xlabel("X coordinate (m)")
plt.ylabel("Y coordinate (m)")
plt.title("Average GRSP Sound Localization per Time Window")
#legend
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
#remove grid lines
plt.grid(False)
plt.show()

# == Save objects from your environment ==================================
import shelve

# List only the variables you want to save
variables_to_save = ['position_estimates']  # Replace with your variable names

with shelve.open("C:/Users/AlexE/OneDrive - EC-EC/Localization/python_output/grsp_gsa.out", 'n') as my_shelf:
    for key in variables_to_save:
        my_shelf[key] = globals()[key]
        
#Load objects in your next session
import shelve
filename = "C:/Users/AlexE/OneDrive - EC-EC/Robinson,Barry (il _ he, him) (ECCC)'s files - Grassland Bird Monitoring/R Projects/Localization/python_output/grsp_gsa.out"
with shelve.open(filename) as my_shelf:
   # Loop through the saved keys and restore them to the global namespace
    for key in my_shelf:
        globals()[key] = my_shelf[key]       
        

