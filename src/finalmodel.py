# model6FINAL.py uses saved feature cache and voxel_corrs.npy, fixed alpha=100000
# DATA_DIR points to local NSD folder

import nibabel as nib
import numpy as np
import pandas as pd
import os
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("NSD_DATA_DIR", "C:/NSDProject")

def dpath(filename):
    return os.path.join(DATA_DIR, filename)

# masks
brainmask = nib.load(dpath('brainmask.nii')).get_fdata()
visualroimask = nib.load(dpath('prf-visualrois.nii')).get_fdata()

v1_mask = (visualroimask == 1) | (visualroimask == 2)
final_mask = (brainmask > 0) & v1_mask
mask_flat = final_mask.reshape(-1)

# load feature cache
CACHE_PATH = dpath("feature_cache.npz")
assert os.path.exists(CACHE_PATH), f"Feature cache not found at {CACHE_PATH}"

print("Loading feature cache")
loaded = np.load(CACHE_PATH, allow_pickle=True)
feature_cache = loaded['cache'].item()
print(f"Loaded {len(feature_cache)} cached features.")

# HRF parameters (TR = 1.6 s)
HRF_SHIFT_START = 3 # 3 * 1.6 = 4.8 s
HRF_SHIFT_END = 5 # 5 * 1.6 = 8.0 s

def extract_responses(v1_voxels, design_vector):
    responses = []
    image_ids = []
    for t, stim in enumerate(design_vector):
        if stim > 0:
            tr_range = range(t + HRF_SHIFT_START, t + HRF_SHIFT_END + 1)
            valid = [tr for tr in tr_range if tr < v1_voxels.shape[1]]
            if valid:
                responses.append(v1_voxels[:, valid].mean(axis=1))
                image_ids.append(int(stim))
    return np.array(responses), image_ids

# main loop
num_runs = 120
all_Y = []
all_ids = []

for i in range(1, num_runs + 1):
    session = f"{(i - 1) // 12 + 1:02d}"
    run = f"{(i - 1) % 12 + 1:02d}"

    t0 = time.time()
    print(f"Processing session {session} run {run}...")

    fmri_file = f"timeseries_session{session}_run{run}.nii"
    design_file = f"design_session{session}_run{run}.txt"

    data = np.asarray(nib.load(dpath(fmri_file)).dataobj, dtype=np.float32)
    T = data.shape[3]
    v1_voxels = data.reshape(-1, T)[mask_flat]
    del data

    design_vector = pd.read_csv(dpath(design_file), header=None)[0].values
    v1_voxels = v1_voxels[:, :len(design_vector)]

    run_mean = v1_voxels.mean(axis=1, keepdims=True)
    v1_voxels = (v1_voxels - run_mean) / (run_mean + 1e-8) * 100

    responses, image_ids = extract_responses(v1_voxels, design_vector)

    df = pd.DataFrame(responses)
    df["image_id"] = image_ids
    grouped = df.groupby("image_id").mean()

    all_Y.append(grouped.values)
    all_ids.extend(grouped.index.values)

    print(f"Done in {time.time() - t0:.1f}s")

# average cross-run repeats
Y_raw = np.vstack(all_Y)
df_final = pd.DataFrame(Y_raw)
df_final["image_id"] = all_ids
grouped_final = df_final.groupby("image_id").mean()

Y = grouped_final.values
unique_ids = grouped_final.index.values
X = np.array([feature_cache[i] for i in unique_ids])

print(f"Final X shape: {X.shape}")
print(f"Final Y shape: {Y.shape}")

# z-score Y
Y_scaler = StandardScaler()
Y_scaled = Y_scaler.fit_transform(Y)

# cross-validated ridge regression, fixed alpha=100000
# Alpha determined from prior RidgeCV run (100000 selected every fold)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=100000))
])

predictions = np.zeros_like(Y_scaled)
print("Running cross-validation...")

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    pipeline.fit(X[train_idx], Y_scaled[train_idx])
    predictions[test_idx] = pipeline.predict(X[test_idx])
    print(f"  Fold {fold}/5 done")

# evaluate
voxel_corrs = np.array([
    np.corrcoef(Y_scaled[:, v], predictions[:, v])[0, 1]
    for v in range(Y_scaled.shape[1])
])

print(f"\nMean voxel correlation : {np.nanmean(voxel_corrs):.4f}")
print(f"Median voxel correlation: {np.nanmedian(voxel_corrs):.4f}")
print(f"Max voxel correlation   : {np.nanmax(voxel_corrs):.4f}")
print(f"Voxels > 0.0 : {np.sum(voxel_corrs > 0.0)}")
print(f"Voxels > 0.1 : {np.sum(voxel_corrs > 0.1)}")
print(f"Voxels > 0.2 : {np.sum(voxel_corrs > 0.2)}")

# save Y_scaled, predictions, unique_ids alongside voxel_corrs
# these are needed by export_demo_data.py
print("Saving voxel_corrs.npy...")
np.save(dpath("voxel_corrs.npy"),  voxel_corrs)

print("Saving Y_scaled.npy...")
np.save(dpath("Y_scaled.npy"), Y_scaled)
print("Saving predictions.npy...")
np.save(dpath("predictions.npy"), predictions)
print("Saving unique_ids.npy...")
np.save(dpath("unique_ids.npy"), unique_ids)
print("Saved voxel_corrs.npy, Y_scaled.npy, predictions.npy, unique_ids.npy")

plt.hist(voxel_corrs, bins=40)
plt.xlabel("Prediction correlation")
plt.ylabel("Number of voxels")
plt.title("ResNet layer1 -> V1 prediction accuracy")
plt.tight_layout()
plt.savefig(dpath("voxel_corr_hist.png"), dpi=150)
plt.show()
