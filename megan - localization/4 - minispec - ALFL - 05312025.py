# -*- coding: utf-8 -*-
"""
Minimum spectrogram filtering + HawkEars re-confirmation for ALFL
Adapted from Erica Alex's script by Megan Edgar

Workflow:
  1. Generate min-spec clips from localized positions
  2. Run HawkEars on those clips (command line, outside Python)
  3. Parse HawkEars output and keep only confirmed ALFL detections
"""

import numpy as np
import pandas as pd
from pathlib import Path

import librosa
from joblib import Parallel, delayed

from opensoundscape import Audio, Spectrogram
from opensoundscape.localization.position_estimate import positions_to_df

# =============================================================================
# Paths – EDIT THESE IF NEEDED
# =============================================================================
species_code = "ALFL"

base_dir  = Path(r"D:/BARLT Localization Project/localization_05312025/hawkears_0_7_ALFL")
clip_dir  = base_dir / "minspec_clips"
label_dir = base_dir / "minspec_output"

clip_dir.mkdir(exist_ok=True)
label_dir.mkdir(exist_ok=True)

confirmed_shelf_path = str(
    base_dir / "pythonoutput" / "alfl_confirmed.out"
)
Path(confirmed_shelf_path).parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper functions
# =============================================================================
def spec_to_audio(spec, sr):
    """Invert spectrogram back to audio via Griffin-Lim."""
    y_inv = librosa.griffinlim(spec.spectrogram, hop_length=256, win_length=512)
    return Audio(y_inv, sr)


def distances_to_receivers(p, dims=2):
    """Euclidean distance from estimated location to each receiver."""
    return [
        np.linalg.norm(p.location_estimate[:dims] - r[:dims])
        for r in p.receiver_locations
    ]


def min_spec_to_audio(position, discard_over_distance=80):
    """
    Build a minimum-spectrogram clip:
      - load aligned audio from each receiver
      - discard receivers beyond `discard_over_distance` metres
      - take the element-wise min across spectrograms
      - invert back to audio
    """
    clips     = position.load_aligned_audio_segments()
    distances = distances_to_receivers(position)

    clips = [c for i, c in enumerate(clips) if distances[i] < discard_over_distance]
    if len(clips) == 0:
        raise ValueError("No receivers within distance threshold")

    specs   = [Spectrogram.from_audio(c, dB_scale=False) for c in clips]
    minspec = specs[0]._spawn(
        spectrogram=np.min(np.array([s.spectrogram for s in specs]), axis=0)
    )
    max_val = np.max([c.samples.max() for c in clips])

    return (
        spec_to_audio(minspec, clips[0].sample_rate)
        .normalize(max_val)
        .extend_to(clips[0].duration)
    )

# =============================================================================
# 1. Filter positions and generate min-spec clips
#    NOTE: `position_estimates` must already exist in your environment
#    (from the localization script or loaded from a shelf file)
# =============================================================================
rms_cutoff = 30  # metres

positions = [p for p in position_estimates if p.residual_rms < rms_cutoff]
print(f"Positions with RMS < {rms_cutoff} m: {len(positions)}")


def process(p, i):
    """Generate and save one min-spec clip. Returns 0 on success, 1 on failure."""
    try:
        min_spec_to_audio(p, discard_over_distance=35).save(str(clip_dir / f"{i}.wav"))
        return 0
    except Exception as e:
        print(f"  clip {i} failed: {e}")
        return 1


results = Parallel(n_jobs=4)(
    delayed(process)(p, i) for i, p in enumerate(positions)
)
print(f"Clips generated: {len(results) - sum(results)} / {len(results)}")
print(f"Failures: {sum(results)}")

# =============================================================================
# 2. Run HawkEars on the min-spec clips (do this in a separate terminal)
#
# python C:/Users/megre/HawkEars/analyze.py -i "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_ALFL/minspec_clips" -o "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_ALFL/minspec_output"
#
# Then come back and run the rest of this script.
# =============================================================================

# =============================================================================
# 3. Load HawkEars output
# =============================================================================
hawkears_results_path = label_dir / "hawkears_labels.csv"

combined = pd.read_csv(hawkears_results_path)
print(f"Total HawkEars labels: {len(combined)}")

# =============================================================================
# 4. Filter to confirmed ALFL detections
# =============================================================================
alfl_confirmed = combined[
    combined["class_code"].str.startswith(species_code, na=False)
].copy()

# Strip extension and '_HawkEars' suffix to recover the original clip index
alfl_confirmed["clip_id"] = (
    alfl_confirmed["filename"]
    .str.replace(".wav", "", regex=False)
    .str.replace("_HawkEars", "", regex=False)
)

print(f"{species_code} labels in HawkEars output: {len(alfl_confirmed)}")

# =============================================================================
# 5. Cross-reference with clips that actually exist on disk
# =============================================================================
existing_clips = {int(f.stem) for f in clip_dir.glob("*.wav")}
failed_clips   = set(range(len(positions))) - existing_clips

print(f"Clips on disk: {len(existing_clips)}, Failed: {len(failed_clips)}")

confirmed_indices = (
    set(alfl_confirmed["clip_id"].astype(int).unique()) & existing_clips
)

confirmed_positions = [
    p for i, p in enumerate(positions) if i in confirmed_indices
]

print(f"HawkEars-confirmed {species_code} positions: {len(confirmed_indices)}")

# =============================================================================
# 6. Save confirmed positions
# =============================================================================
import shelve

with shelve.open(confirmed_shelf_path, "n") as db:
    db["position_estimates"] = confirmed_positions

print(f"Saved {len(confirmed_positions)} confirmed positions to {confirmed_shelf_path}")
