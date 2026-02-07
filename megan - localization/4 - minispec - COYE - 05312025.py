from opensoundscape import Audio, Spectrogram
from opensoundscape.localization.position_estimate import positions_to_df
import numpy as np
from pathlib import Path
import librosa
from joblib import Parallel, delayed

# paths
out_dir = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_output"
clip_dir = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_clips"

Path(out_dir).mkdir(exist_ok=True)
Path(clip_dir).mkdir(exist_ok=True)

def spec_to_audio(spec, sr):
    y_inv = librosa.griffinlim(spec.spectrogram, hop_length=256, win_length=512)
    return Audio(y_inv, sr)

def distances_to_receivers(p, dims=2):
    return [
        np.linalg.norm(p.location_estimate[:dims] - r[:dims])
        for r in p.receiver_locations
    ]

def min_spec_to_audio(position, discard_over_distance=50):
    clips = position.load_aligned_audio_segments()
    distances = distances_to_receivers(position)
    clips = [c for i, c in enumerate(clips) if distances[i] < discard_over_distance]
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

# Filter positions
positions = [p for p in position_estimates if p.residual_rms < 20]
print(f"Processing {len(positions)} positions")

# Generate clips
def process(p, i):
    try:
        min_spec_to_audio(p, discard_over_distance=35).save(f"{clip_dir}/{i}.wav")
        return 0
    except:
        return 1

results = Parallel(n_jobs=4)(delayed(process)(p, i) for i, p in enumerate(positions))
print(f"Failures: {sum(results)} of {len(results)}")

# run hawkears in command prompt 
# python C:\Users\megre\HawkEars\analyze.py -i "D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_clips" -o "D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_output"


label_dir = Path(r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_output")

all_labels = []
for label_file in label_dir.glob("*.txt"):
    df = pd.read_csv(label_file, sep='\t', header=None, names=['start', 'end', 'label'])
    df['file'] = label_file.stem
    all_labels.append(df)

combined = pd.concat(all_labels, ignore_index=True)
combined.to_csv(label_dir / "hawkears_minspec_results.csv", index=False)




import pandas as pd
from pathlib import Path

label_dir = Path(r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_output")

all_labels = []
for label_file in label_dir.glob("*.txt"):
    df = pd.read_csv(label_file, sep='\t', header=None, names=['start', 'end', 'label'])
    df['file'] = label_file.stem
    all_labels.append(df)

combined = pd.concat(all_labels, ignore_index=True)
coye_confirmed = combined[combined['label'].str.startswith('COYE', na=False)]
# Also strip '_HawkEars' from file column to get the original clip index
coye_confirmed = coye_confirmed.copy()
coye_confirmed['file'] = coye_confirmed['file'].str.replace('_HawkEars', '')

print(f"COYE confirmed: {len(coye_confirmed)}")



from pathlib import Path

clip_dir = Path(r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/minspec_clips")

# Which clips exist on disk = succeeded
existing_clips = {int(f.stem) for f in clip_dir.glob("*.wav")}
failed = set(range(len(positions))) - existing_clips

print(f"Clips on disk: {len(existing_clips)}, Failed: {len(failed)}")

# Confirmed = exists on disk AND HawkEars detected COYE
confirmed_indices = set(coye_confirmed['file'].astype(int).unique()) & existing_clips

confirmed_positions = [p for i, p in enumerate(positions) if i in confirmed_indices]

print(f"HawkEars confirmed COYE: {len(confirmed_indices)}")

import shelve

confirmed_shelf = r"D:/BARLT Localization Project/localization_05312025/hawkears_0_4thresh_COYE/pythonoutput/coye_confirmed.out"
with shelve.open(confirmed_shelf, "n") as db:
    db["position_estimates"] = confirmed_positions

print(f"Saved {len(confirmed_positions)} confirmed positions to {confirmed_shelf}")

