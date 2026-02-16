# -*- coding: utf-8 -*-
"""
Batch minimum spectrogram filtering + HawkEars re-confirmation.

For each species:
  1. Load position estimates from shelf
  2. Generate min-spec clips
  3. (Run HawkEars externally)
  4. Parse HawkEars CSV output
  5. Save confirmed positions to shelf

Usage:
  - Run this script FIRST to generate clips (steps 1-2)
  - Run HawkEars on each species' minspec_clips folder
  - Run this script AGAIN to parse results and save confirmed positions (steps 3-5)
    (It will skip clip generation if clips already exist)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shelve

import librosa
from joblib import Parallel, delayed

from opensoundscape import Audio, Spectrogram
from opensoundscape.localization.position_estimate import positions_to_df

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(r"D:/BARLT Localization Project/localization_05312025")

SPECIES_LIST = ["RCKI", "TEWA", "RWBL", "CHSP", "YEWA", "WTSP", "AMRO", "YBFL", "AMRE"]

RMS_CUTOFF = 30  # metres — filter positions before generating clips
DISCARD_OVER_DISTANCE = 35  # metres — drop distant receivers in minspec
N_JOBS = 4  # parallel workers for clip generation

# =============================================================================
# HELPER FUNCTIONS
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
    """Build a minimum-spectrogram clip from aligned receiver audio."""
    clips = position.load_aligned_audio_segments()
    distances = distances_to_receivers(position)

    clips = [c for i, c in enumerate(clips) if distances[i] < discard_over_distance]
    if len(clips) == 0:
        raise ValueError("No receivers within distance threshold")

    specs = [Spectrogram.from_audio(c, dB_scale=False) for c in clips]
    minspec = specs[0]._spawn(
        spectrogram=np.min(np.array([s.spectrogram for s in specs]), axis=0)
    )
    max_val = np.max([c.samples.max() for c in clips])

    return (
        spec_to_audio(minspec, clips[0].sample_rate)
        .normalize(max_val)
        .extend_to(clips[0].duration)
    )


def process_clip(p, i, clip_dir, discard_over_distance):
    """Generate and save one min-spec clip. Returns 0 on success, 1 on failure."""
    try:
        min_spec_to_audio(p, discard_over_distance=discard_over_distance).save(
            str(clip_dir / f"{i}.wav")
        )
        return 0
    except Exception as e:
        print(f"      clip {i} failed: {e}")
        return 1


# =============================================================================
# MAIN
# =============================================================================

def main():
    for species_code in SPECIES_LIST:
        print(f"\n{'='*60}")
        print(f"  {species_code}")
        print(f"{'='*60}")

        species_dir = BASE_DIR / f"hawkears_0_7_{species_code}"
        clip_dir = species_dir / "minspec_clips"
        label_dir = species_dir / "minspec_output"
        shelf_in = str(species_dir / "pythonoutput" / f"{species_code.lower()}.out")
        shelf_out = str(species_dir / "pythonoutput" / f"{species_code.lower()}_confirmed.out")

        clip_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # --- Load positions ---
        try:
            with shelve.open(shelf_in, "r") as db:
                position_estimates = db["position_estimates"]
        except Exception as e:
            print(f"    Could not load shelf: {e}. Skipping.")
            continue

        positions = [p for p in position_estimates if p.residual_rms < RMS_CUTOFF]
        print(f"    Positions with RMS < {RMS_CUTOFF} m: {len(positions)}")

        if len(positions) == 0:
            print(f"    No positions pass filter. Skipping.")
            continue

        # --- Step 1: Generate clips (skip if already done) ---
        existing_clips = list(clip_dir.glob("*.wav"))
        if len(existing_clips) >= len(positions):
            print(f"    Clips already exist ({len(existing_clips)}). Skipping generation.")
        else:
            print(f"    Generating min-spec clips...")
            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_clip)(p, i, clip_dir, DISCARD_OVER_DISTANCE)
                for i, p in enumerate(positions)
            )
            print(f"    Clips: {len(results) - sum(results)} / {len(results)}, "
                  f"Failures: {sum(results)}")

        # --- Print HawkEars command ---
        print(f"\n    HawkEars command:")
        print(f'    python C:/Users/megre/HawkEars/analyze.py -i "{clip_dir}" -o "{label_dir}" --rtype csv')



        # --- Step 2: Parse HawkEars output (if it exists) ---
        hawkears_path = label_dir / "HawkEars_labels.csv"
        if not hawkears_path.exists():
            print(f"\n    HawkEars output not found yet. Run the command above, then re-run this script.")
            continue

        combined = pd.read_csv(hawkears_path)
        print(f"\n    Total HawkEars labels: {len(combined)}")

        # Filter to confirmed detections of this species
        confirmed = combined[
            combined["class_code"].str.startswith(species_code, na=False)
        ].copy()

        confirmed["clip_id"] = (
            confirmed["filename"]
            .str.replace(".wav", "", regex=False)
        )

        print(f"    {species_code} confirmed labels: {len(confirmed)}")

        # Cross-reference with clips on disk
        existing_clips = {int(f.stem) for f in clip_dir.glob("*.wav")}
        confirmed_indices = set(confirmed["clip_id"].astype(int).unique()) & existing_clips

        confirmed_positions = [
            p for i, p in enumerate(positions) if i in confirmed_indices
        ]

        print(f"    HawkEars-confirmed positions: {len(confirmed_positions)}")

        if len(confirmed_positions) == 0:
            print(f"    No confirmed positions for {species_code}.")
            continue

        # --- Save confirmed shelf ---
        Path(shelf_out).parent.mkdir(parents=True, exist_ok=True)
        with shelve.open(shelf_out, "n") as db:
            db["position_estimates"] = confirmed_positions

        print(f"    Saved {len(confirmed_positions)} confirmed positions to {shelf_out}")

    print(f"\n{'='*60}")
    print("  BATCH MINSPEC COMPLETE")
    print(f"{'='*60}")
    print("\nIf any species are missing HawkEars output, run the printed commands")
    print("and then re-run this script to finish confirmation.")


if __name__ == "__main__":
    main()
