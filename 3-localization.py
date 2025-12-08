# -*- coding: utf-8 -*-
"""
Localization of [[Veery (VEER)]] calls using opensoundscape
Adapted from Erica Alex's script by Megan Edgar, Dec 10 2025
"""

import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import shelve

from opensoundscape.localization import SynchronizedRecorderArray
from opensoundscape import Spectrogram

# =============================================================================
# Paths – EDIT THESE IF NEEDED
# =============================================================================
aru_coords_path   = r"C:/Users/EdgarM/Desktop/Localization/aru_coords.csv"
detections_path   = r"C:/Users/EdgarM/Desktop/Localization/detections_VEER.csv"
shelf_out_path    = r"C:/Users/EdgarM/Desktop/Localization/pythonoutput/veer.out"

# make sure the output folder exists
os.makedirs(os.path.dirname(shelf_out_path), exist_ok=True)

# =============================================================================
# Load ARU coordinates and initialize array
# aru_coords.csv should have columns: file, x (lon), y (lat)
# =============================================================================
aru_coords = pd.read_csv(aru_coords_path, index_col=0)

array = SynchronizedRecorderArray(aru_coords)

# =============================================================================
# Load detections
# detections_VEER.csv should have: file, start_time, end_time, VEER
# =============================================================================
detections = pd.read_csv(detections_path)

# -----------------------------------------------------------------------------
# Add start_timestamp based on the recording start time parsed from filename
# Assumes filenames contain 'SYYYYMMDDTHHMMSS' (e.g. S20250531T052140...)
# and that start_time is offset in seconds from that.
# Time zone: America/Winnipeg (CDT in May, UTC-5) to match '-0500' in filename.
# -----------------------------------------------------------------------------
#Add the start timestamp, adjusted to match the trimmed recordings (it should be the start time of your latest recording) 
import pytz
from datetime import datetime, timedelta

# start time of the latest trimmed recording (L1N5E1)
local_timestamp = datetime(2025, 5, 31, 5, 23, 41)
local_timezone = pytz.timezone("America/Winnipeg")

detections["start_timestamp"] = [
    local_timezone.localize(local_timestamp) + timedelta(seconds=s)
    for s in detections["start_time"]
]

detections = detections.set_index(
    ["file", "start_time", "end_time", "start_timestamp"]
)
# =============================================================================
# Localization
# =============================================================================
min_n_receivers = 3   # minimum number of ARUs with detection
max_receiver_dist = 40  # maximum distance (m) between recorders for TDOA

position_estimates = array.localize_detections(
    detections,
    min_n_receivers=min_n_receivers,
    max_receiver_dist=max_receiver_dist,
)

print(f"Number of position estimates returned: {len(position_estimates)}")

if len(position_estimates) == 0:
    raise SystemExit("No position estimates – check detections / thresholds.")

# =============================================================================
# Explore a single example event
# =============================================================================
example = position_estimates[0]  # first event
print(f"The start time of the detection: {example.start_timestamp}")
print(f"This is a detection of the class/species: {example.class_name}")
print(
    f"The duration of the time-window in which the sound was detected: {example.duration}"
)
print(f"The estimated location of the sound: {example.location_estimate}")
print(f"The receivers on which VEER was detected: \n{example.receiver_files}")
print(f"The estimated time-delays of arrival: \n{example.tdoas}")
print(f"The normalized Cross-Correlation scores: \n{example.cc_maxs}")

# Plot ARUs + this single location estimate
plt.figure()
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.scatter(
    x=example.location_estimate[0],
    y=example.location_estimate[1],
    label=f"{example.class_name} example",
)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Example VEER localization")
plt.show()

# =============================================================================
# Check spectrogram alignment for this example
# =============================================================================
audio_segments = example.load_aligned_audio_segments()
specs = [Spectrogram.from_audio(a).bandpass(8000, 12000) for a in audio_segments]
plt.figure()
plt.pcolormesh(np.vstack([s.spectrogram for s in specs]), cmap="Greys")
plt.title("Aligned spectrograms for example VEER event")
plt.show()

print(f"Residual RMS for example event: {example.residual_rms:.2f} m")

# =============================================================================
# All VEER estimates for the same time window as the example
# (different reference ARUs, same event)
# =============================================================================
veer_same_event = [
    e
    for e in position_estimates
    if e.class_name == example.class_name
    and e.start_timestamp == example.start_timestamp
]

x_coords = [e.location_estimate[0] for e in veer_same_event]
y_coords = [e.location_estimate[1] for e in veer_same_event]
rms = [e.residual_rms for e in veer_same_event]

plt.figure()
plt.scatter(
    x_coords,
    y_coords,
    c=rms,
    alpha=0.4,
    edgecolors="black",
    cmap="jet",
    label="VEER event estimates",
)
cbar = plt.colorbar()
cbar.set_label("Residual RMS (m)")
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("VEER localization (single time window)")
plt.show()

# =============================================================================
# Residual RMS summary across all VEER estimates
# =============================================================================
veer_all = [e for e in position_estimates if e.class_name == "VEER"]

residuals = [e.residual_rms for e in veer_all]
min_rms = min(residuals)
max_rms = max(residuals)
mean_rms = np.mean(residuals)
median_rms = np.median(residuals)
lqt_rms = np.quantile(residuals, 0.25)
uqt_rms = np.quantile(residuals, 0.75)

print("Residual RMS (VEER events):")
print(f"  Min:    {min_rms:.2f} m")
print(f"  Max:    {max_rms:.2f} m")
print(f"  Mean:   {mean_rms:.2f} m")
print(f"  Median: {median_rms:.2f} m")
print(f"  Q1:     {lqt_rms:.2f} m")
print(f"  Q3:     {uqt_rms:.2f} m")

# Filter to good localizations
rms_cutoff = 35  # m
low_rms = [e for e in veer_all if e.residual_rms < rms_cutoff]

plt.figure()
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARUs")
plt.scatter(
    [e.location_estimate[0] for e in low_rms],
    [e.location_estimate[1] for e in low_rms],
    edgecolor="black",
    alpha=0.6,
    label=f"VEER (RMS < {rms_cutoff} m)",
)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("VEER localizations with low residual error")
plt.show()

# =============================================================================
# Average VEER position per time window (for good localizations)
# =============================================================================
rms_threshold = 20  # meters, residual RMS threshold for “high-confidence” events

veer_events = [
    e for e in position_estimates
    if e.class_name == "VEER" and e.residual_rms < rms_threshold
]

grouped_by_time = defaultdict(list)
for event in veer_events:
    grouped_by_time[event.start_timestamp].append(event)

# require at least N estimates per window
min_events_per_window = 3

filtered_groups = {
    ts: events
    for ts, events in grouped_by_time.items()
    if len(events) >= min_events_per_window
}

avg_locations = []
timestamps = []

for timestamp, events in filtered_groups.items():
    x_avg = np.mean([e.location_estimate[0] for e in events])
    y_avg = np.mean([e.location_estimate[1] for e in events])
    avg_locations.append((x_avg, y_avg))
    timestamps.append(timestamp)

if len(avg_locations) == 0:
    print("No time windows passed the RMS / count filters.")
else:
    x_avg_coords, y_avg_coords = zip(*avg_locations)

    # Optional: overlay with known locations (here using same ARU coords by default)
    known_locations = pd.read_csv(aru_coords_path, index_col=0)
    x_known = known_locations["x"]
    y_known = known_locations["y"]

    plt.figure(figsize=(8, 7))
    plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARUs")
    plt.scatter(
        x_avg_coords,
        y_avg_coords,
        label="Average VEER location per time window",
        alpha=0.7,
    )
    plt.scatter(
        x_known,
        y_known,
        marker="X",
        s=80,
        label="Known points",
    )

    # Axis limits based on your ARU grid, with small padding
    padding = 0.0003
    x_min = min(aru_coords["x"].min(), min(x_avg_coords)) - padding
    x_max = max(aru_coords["x"].max(), max(x_avg_coords)) + padding
    y_min = min(aru_coords["y"].min(), min(y_avg_coords)) - padding
    y_max = max(aru_coords["y"].max(), max(y_avg_coords)) + padding
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Average VEER localization per time window")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(False)
    plt.show()

# =============================================================================
# Save objects for later use
# =============================================================================
variables_to_save = ["position_estimates"]

with shelve.open(shelf_out_path, "n") as my_shelf:
    for key in variables_to_save:
        my_shelf[key] = globals()[key]

print(f"Saved {variables_to_save} to {shelf_out_path}")
