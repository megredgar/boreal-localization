# -*- coding: utf-8 -*-
"""
Localization of Connecticut Warbler (CONW) calls using opensoundscape
Adapted from Erica Alex's script by Megan Edgar, Dec 10 2025

"""

import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import shelve

from pyproj import Transformer
from opensoundscape.localization import SynchronizedRecorderArray
from opensoundscape import Spectrogram

# =============================================================================
# Paths – EDIT THESE IF NEEDED
# =============================================================================
aru_coords_path = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/aru_coords.csv"
detections_path = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/detections_CONW.csv"
shelf_out_path  = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW/pythonoutput/conw.out"

os.makedirs(os.path.dirname(shelf_out_path), exist_ok=True)

# =============================================================================
# Load ARU coordinates and convert to UTM
# aru_coords.csv columns: file, x (lon), y (lat)
# =============================================================================
aru_coords = pd.read_csv(aru_coords_path, index_col=0)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
aru_coords["x"], aru_coords["y"] = transformer.transform(
    aru_coords["x"].values, aru_coords["y"].values
)

array = SynchronizedRecorderArray(aru_coords)

# =============================================================================
# Load detections
# detections_CONW.csv columns: file, start_time, end_time, CONW
# =============================================================================
detections = pd.read_csv(detections_path)

# Start timestamp: must match the start time of your latest trimmed recording
local_timestamp = datetime(2025, 5, 31, 5, 23, 42)
local_timezone  = pytz.timezone("America/Winnipeg")

detections = detections.reset_index()
detections["start_timestamp"] = [
    local_timezone.localize(local_timestamp) + timedelta(seconds=s)
    for s in detections["start_time"]
]

detections = detections.set_index(
    ["file", "start_time", "end_time", "start_timestamp"]
)

# =============================================================================
# Quick sanity checks
# =============================================================================
det_files   = set(detections.index.get_level_values("file").unique())
coord_files = set(array.file_coords.index)

print("unique files in detections:", len(det_files))
print("unique files in coords:   ", len(coord_files))
print("detections missing coords:", len(det_files - coord_files))
print("coords not used by detections (extra):", len(coord_files - det_files))

missing = sorted(det_files - coord_files)
if missing:
    print("First 10 missing files:", missing[:10])

d = detections.reset_index()
print("\nSample files:", d["file"].head(3).tolist())
print(d["file"].str.extract(r"_S(\d{8})T", expand=False).value_counts().head(5))
print("start_timestamp min:", d["start_timestamp"].min())
print("start_timestamp max:", d["start_timestamp"].max())

# =============================================================================
# Localization
# =============================================================================
min_n_receivers  = 5    # minimum ARUs with detection in a time bin
max_receiver_dist = 80  # maximum inter-recorder distance (m) for TDOA

position_estimates = array.localize_detections(
    detections,
    min_n_receivers=min_n_receivers,
    max_receiver_dist=max_receiver_dist,
)

print(f"\nNumber of position estimates returned: {len(position_estimates)}")

if len(position_estimates) == 0:
    raise SystemExit("No position estimates — check detections / thresholds.")

# =============================================================================
# Explore a single example event
# =============================================================================
example = position_estimates[15]
print(f"\nExample event:")
print(f"  Start time:    {example.start_timestamp}")
print(f"  Class/species: {example.class_name}")
print(f"  Duration:      {example.duration}")
print(f"  Location est:  {example.location_estimate}")
print(f"  Receivers:\n{example.receiver_files}")
print(f"  TDOAs:\n{example.tdoas}")
print(f"  CC maxs:\n{example.cc_maxs}")

# Plot ARUs + single location estimate
plt.figure()
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.scatter(
    x=example.location_estimate[0],
    y=example.location_estimate[1],
    label=f"{example.class_name} example",
)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Example CONW localization")
plt.show()

# =============================================================================
# Check spectrogram alignment for example event
# NOTE: CONW song energy is roughly 2500–6000 Hz; adjust bandpass as needed
# =============================================================================
audio_segments = example.load_aligned_audio_segments()
specs = [Spectrogram.from_audio(a).bandpass(2500, 6000) for a in audio_segments]
plt.figure()
plt.pcolormesh(np.vstack([s.spectrogram for s in specs]), cmap="Greys")
plt.title("Aligned spectrograms for example CONW event")
plt.show()

print(f"Residual RMS for example event: {example.residual_rms:.2f} m")

# =============================================================================
# All CONW estimates for the same time window as the example
# =============================================================================
conw_same_event = [
    e for e in position_estimates
    if e.class_name == example.class_name
    and e.start_timestamp == example.start_timestamp
]

x_coords = [e.location_estimate[0] for e in conw_same_event]
y_coords = [e.location_estimate[1] for e in conw_same_event]
rms      = [e.residual_rms for e in conw_same_event]

plt.figure()
plt.scatter(
    x_coords, y_coords, c=rms, alpha=0.4,
    edgecolors="black", cmap="jet", label="CONW event estimates",
)
cbar = plt.colorbar()
cbar.set_label("Residual RMS (m)")
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("CONW localization (single time window)")
plt.show()

# Same event, RMS-filtered
conw_same_event_filt = [
    e for e in position_estimates
    if e.class_name == example.class_name
    and e.start_timestamp == example.start_timestamp
    and e.residual_rms <= 30
]

x_coords_f = [e.location_estimate[0] for e in conw_same_event_filt]
y_coords_f = [e.location_estimate[1] for e in conw_same_event_filt]
rms_f      = [e.residual_rms for e in conw_same_event_filt]

plt.figure()
plt.scatter(
    x_coords_f, y_coords_f, c=rms_f, alpha=0.4,
    edgecolors="black", cmap="jet", label="CONW event estimates",
)
cbar = plt.colorbar()
cbar.set_label("Residual RMS (m)")
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARU")
plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("CONW localization (single time window, RMS ≤ 30 m)")
plt.show()

# =============================================================================
# Residual RMS summary across all CONW estimates
# =============================================================================
conw_all  = [e for e in position_estimates if e.class_name == "CONW"]
residuals = [e.residual_rms for e in conw_all]

print("\nResidual RMS (all CONW estimates):")
print(f"  Min:    {min(residuals):.2f} m")
print(f"  Max:    {max(residuals):.2f} m")
print(f"  Mean:   {np.mean(residuals):.2f} m")
print(f"  Median: {np.median(residuals):.2f} m")
print(f"  Q1:     {np.quantile(residuals, 0.25):.2f} m")
print(f"  Q3:     {np.quantile(residuals, 0.75):.2f} m")

# Plot low-RMS localizations
rms_cutoff = 35  # m
low_rms = [e for e in conw_all if e.residual_rms < rms_cutoff]

plt.figure()
plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARUs")
plt.scatter(
    [e.location_estimate[0] for e in low_rms],
    [e.location_estimate[1] for e in low_rms],
    edgecolor="black", alpha=0.6,
    label=f"CONW (RMS < {rms_cutoff} m)",
)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("CONW localizations with low residual error")
plt.show()

# =============================================================================
# Average CONW position per time window (high-confidence events)
# =============================================================================
rms_threshold         = 25   # meters
min_events_per_window = 2    # require at least N estimates per bin

conw_events = [
    e for e in position_estimates
    if e.class_name == "CONW" and e.residual_rms < rms_threshold
]

grouped_by_time = defaultdict(list)
for event in conw_events:
    grouped_by_time[event.start_timestamp].append(event)

filtered_groups = {
    ts: events for ts, events in grouped_by_time.items()
    if len(events) >= min_events_per_window
}

avg_locations = []
timestamps    = []

for timestamp, events in filtered_groups.items():
    x_avg = np.mean([e.location_estimate[0] for e in events])
    y_avg = np.mean([e.location_estimate[1] for e in events])
    avg_locations.append((x_avg, y_avg))
    timestamps.append(timestamp)

if len(avg_locations) == 0:
    print("\nNo time windows passed the RMS / count filters.")
else:
    x_avg_coords, y_avg_coords = zip(*avg_locations)

    # Padding in metres (coordinates are UTM now)
    padding = 80
    x_min = min(aru_coords["x"].min(), min(x_avg_coords)) - padding
    x_max = max(aru_coords["x"].max(), max(x_avg_coords)) + padding
    y_min = min(aru_coords["y"].min(), min(y_avg_coords)) - padding
    y_max = max(aru_coords["y"].max(), max(y_avg_coords)) + padding

    plt.figure(figsize=(8, 7))
    plt.plot(aru_coords["x"], aru_coords["y"], "^", label="ARUs")
    plt.scatter(
        x_avg_coords, y_avg_coords,
        label="Avg CONW location per time window", alpha=0.7,
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Average CONW localization per time window")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(False)
    plt.show()

    print(f"\n{len(avg_locations)} time windows with ≥{min_events_per_window} "
          f"estimates and RMS < {rms_threshold} m")

# =============================================================================
# Save position estimates for later use
# =============================================================================
with shelve.open(shelf_out_path, "n") as my_shelf:
    my_shelf["position_estimates"] = position_estimates

print(f"\nSaved position_estimates to {shelf_out_path}")