"""
export_demo_data.py
-------------------
Generates all web-friendly assets for the Google AI Studio interactive demo.

Prerequisites (run model5.py first to produce these):
  C:/NSDProject/voxel_corrs.npy
  C:/NSDProject/Y_scaled.npy
  C:/NSDProject/predictions.npy
  C:/NSDProject/unique_ids.npy
  C:/NSDProject/nsd_stimuli_subset.hdf5
  C:/NSDProject/prf-visualrois.nii
  C:/NSDProject/brainmask.nii
  C:/NSDProject/voxel_corr_map.nii   (optional — generated here if missing)

Outputs written to C:/NSDProject/demo_export/:
  images/img_XXXXX.jpg               25 best-predicted images, 300x300
  correlations.json                  per-voxel r, mean, median, histogram
  voxel_coords.json                  [x,y,z] for every V1 voxel (MNI mm)
  image_responses.json               predicted + ground truth vectors per image
  brain_heatmap_guide.md             how to map correlations onto your GLB mesh
"""

import os
import json
import numpy as np
import nibabel as nib
import h5py
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "C:/NSDProject"
EXPORT_DIR = os.path.join(DATA_DIR, "demo_export")
IMG_DIR    = os.path.join(EXPORT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

def dpath(f): return os.path.join(DATA_DIR, f)
def epath(f): return os.path.join(EXPORT_DIR, f)

N_IMAGES = 25   # number of best-predicted images to export

# ── load model outputs ─────────────────────────────────────────────────────────
print("Loading model outputs...")
voxel_corrs = np.load(dpath("voxel_corrs.npy"))
Y_scaled    = np.load(dpath("Y_scaled.npy"))
predictions = np.load(dpath("predictions.npy"))
unique_ids  = np.load(dpath("unique_ids.npy"))

# ── load masks (needed for coords and NIfTI back-projection) ──────────────────
print("Loading masks...")
brainmask     = nib.load(dpath('brainmask.nii'))
visualroimask = nib.load(dpath('prf-visualrois.nii')).get_fdata()

v1_mask    = (visualroimask == 1) | (visualroimask == 2)
final_mask = (brainmask.get_fdata() > 0) & v1_mask
affine     = brainmask.affine

# ── load stimuli ───────────────────────────────────────────────────────────────
print("Loading stimuli HDF5...")
h5_file   = h5py.File(dpath("nsd_stimuli_subset.hdf5"), "r")
imgBrick  = h5_file["imgBrick"]
image_ids = list(h5_file["image_ids"][:])
id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PER-IMAGE MEAN CORRELATION — rank images by how well the model predicted
#     their response pattern across all V1 voxels
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nRanking images by prediction accuracy...")

# For each image (row), compute the mean absolute correlation across voxels
# between its predicted and ground-truth response vector.
# This tells you which images the model understood best.
image_scores = np.array([
    np.corrcoef(Y_scaled[i, :], predictions[i, :])[0, 1]
    for i in range(Y_scaled.shape[0])
])

top_indices  = np.argsort(image_scores)[::-1][:N_IMAGES]
top_img_ids  = unique_ids[top_indices]

print(f"  Top {N_IMAGES} image scores: {image_scores[top_indices][:5].round(3)} ...")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  SAVE TOP-25 IMAGES AS 300×300 JPEGs
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving {N_IMAGES} images to {IMG_DIR}...")

saved_image_keys = []   # keep ordered list for JSON

for rank, img_id in enumerate(top_img_ids):
    img_array = np.array(imgBrick[id_to_idx[img_id]])
    pil_img   = Image.fromarray(img_array).resize((300, 300), Image.LANCZOS)
    key       = f"img_{img_id:05d}"
    out_path  = os.path.join(IMG_DIR, f"{key}.jpg")
    pil_img.save(out_path, "JPEG", quality=92)
    saved_image_keys.append(key)
    if rank < 5:
        print(f"  Saved {key}.jpg  (score={image_scores[top_indices[rank]]:.3f})")

print(f"  ... and {N_IMAGES - 5} more.")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  IMAGE RESPONSES JSON
#     { "img_XXXXX": { "predicted": [...], "ground_truth": [...] }, ... }
#     Vectors are per-voxel values for that image, rounded to 4 dp.
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding image_responses.json...")

image_responses = {}
for rank, (img_id, key) in enumerate(zip(top_img_ids, saved_image_keys)):
    row = top_indices[rank]
    image_responses[key] = {
        "predicted":    [round(float(v), 4) for v in predictions[row, :]],
        "ground_truth": [round(float(v), 4) for v in Y_scaled[row, :]],
        "image_score":  round(float(image_scores[top_indices[rank]]), 4),
    }

with open(epath("image_responses.json"), "w") as f:
    json.dump(image_responses, f)
print(f"  Written: image_responses.json  ({len(image_responses)} images, "
      f"{Y_scaled.shape[1]} voxels each)")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  VOXEL COORDINATES JSON
#     Order EXACTLY matches the voxel vectors in image_responses.json.
#     Coordinates are in MNI mm space (affine-transformed from voxel indices).
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding voxel_coords.json...")

# Get voxel ijk indices where the V1 mask is True, in the same flat order
# as mask_flat — this is the order used throughout the pipeline.
voxel_ijk = np.array(np.where(final_mask)).T   # (N_voxels, 3)

# Convert to MNI mm coords using the NIfTI affine
# affine @ [i, j, k, 1]^T → [x, y, z, 1]^T in mm
ones      = np.ones((voxel_ijk.shape[0], 1))
ijk_hom   = np.hstack([voxel_ijk, ones])        # (N, 4)
xyz_mm    = (affine @ ijk_hom.T).T[:, :3]       # (N, 3)

voxel_coords_list = [[round(float(x), 2),
                      round(float(y), 2),
                      round(float(z), 2)]
                     for x, y, z in xyz_mm]

voxel_coords_json = {"voxel_coords": voxel_coords_list}

with open(epath("voxel_coords.json"), "w") as f:
    json.dump(voxel_coords_json, f)
print(f"  Written: voxel_coords.json  ({len(voxel_coords_list)} voxels)")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  CORRELATIONS JSON
#     Per-voxel r, summary stats, pre-computed histogram bins
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding correlations.json...")

corrs_clean = voxel_corrs[~np.isnan(voxel_corrs)]

# Pre-compute histogram so the frontend does zero work
hist_counts, hist_edges = np.histogram(corrs_clean, bins=40)
hist_bin_centers = ((hist_edges[:-1] + hist_edges[1:]) / 2).tolist()

correlations_json = {
    "correlations": [round(float(v), 4) for v in voxel_corrs.tolist()],
    "mean":         round(float(np.nanmean(voxel_corrs)), 4),
    "median":       round(float(np.nanmedian(voxel_corrs)), 4),
    "max":          round(float(np.nanmax(voxel_corrs)), 4),
    "min":          round(float(np.nanmin(voxel_corrs)), 4),
    "n_voxels":     int(len(voxel_corrs)),
    "n_above_0":    int(np.sum(voxel_corrs > 0.0)),
    "n_above_0_1":  int(np.sum(voxel_corrs > 0.1)),
    "n_above_0_2":  int(np.sum(voxel_corrs > 0.2)),
    "histogram": {
        "counts":      hist_counts.tolist(),
        "bin_centers": [round(v, 4) for v in hist_bin_centers],
        "bin_edges":   [round(float(v), 4) for v in hist_edges.tolist()],
    }
}

with open(epath("correlations.json"), "w") as f:
    json.dump(correlations_json, f)
print(f"  Written: correlations.json")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VOXEL CORRELATION MAP NIfTI  (back-project for GLB mesh mapping)
# ══════════════════════════════════════════════════════════════════════════════
nii_out_path = dpath("voxel_corr_map.nii")
if not os.path.exists(nii_out_path):
    print("\nGenerating voxel_corr_map.nii...")
    corr_vol          = np.full(final_mask.shape, np.nan, dtype=np.float32)
    corr_vol[final_mask] = voxel_corrs
    nib.save(nib.Nifti1Image(corr_vol, affine), nii_out_path)
    print(f"  Written: {nii_out_path}")
else:
    print(f"\nvoxel_corr_map.nii already exists, skipping.")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  GLB BRAIN MESH MAPPING GUIDE
# ══════════════════════════════════════════════════════════════════════════════
guide = """
# Mapping the Voxel Heatmap onto Your GLB Brain Mesh
# =====================================================

## The Core Problem
Your GLB mesh is a surface (vertices in mm space). Your voxel
correlations live in a 3D volume (voxels in MNI mm space from
voxel_coords.json). You need to colour each mesh vertex by the
correlation of the nearest V1 voxel.

## Recommended Approach: Nearest-Neighbour Vertex Colouring

### Step 1 — Extract mesh vertex positions
Use Python + trimesh (pip install trimesh):

    import trimesh, numpy as np, json
    mesh = trimesh.load("your_brain.glb")
    # If the GLB has multiple sub-meshes, concatenate:
    vertices = mesh.vertices   # shape (N_vertices, 3), already in mm

### Step 2 — Load voxel coords and correlations
    with open("voxel_coords.json") as f:
        coords = np.array(json.load(f)["voxel_coords"])  # (N_voxels, 3)
    with open("correlations.json") as f:
        d = json.load(f)
    corrs = np.array(d["correlations"])                   # (N_voxels,)

### Step 3 — Nearest-neighbour query (fast with scipy)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # For each mesh vertex, find the closest V1 voxel
    dist, idx = tree.query(vertices, k=1)
    # Only colour vertices within 8mm of a V1 voxel (others stay grey)
    MAX_DIST_MM = 8.0
    vertex_corrs = np.full(len(vertices), np.nan)
    close = dist < MAX_DIST_MM
    vertex_corrs[close] = corrs[idx[close]]

### Step 4 — Map correlations to RGB colours
    import matplotlib.pyplot as plt
    cmap   = plt.get_cmap("RdYlGn")          # red=low, green=high
    vmin, vmax = 0.0, np.nanmax(corrs)
    normed = np.clip((vertex_corrs - vmin) / (vmax - vmin), 0, 1)
    normed = np.nan_to_num(normed, nan=0.5)  # NaN → mid-grey
    colors = (cmap(normed)[:, :3] * 255).astype(np.uint8)

### Step 5 — Re-export as GLB with vertex colours
    mesh.visual.vertex_colors = np.hstack(
        [colors, np.full((len(vertices), 1), 255, dtype=np.uint8)]
    )
    mesh.export("brain_heatmap.glb")

## Using the Coloured GLB in Google AI Studio / Three.js
The exported brain_heatmap.glb will load directly in Three.js with
vertex colours intact:

    const loader = new THREE.GLTFLoader();
    loader.load("brain_heatmap.glb", (gltf) => {
        scene.add(gltf.scene);
    });

No shader or texture work is needed — vertex colours are rendered
automatically by Three.js with the default MeshStandardMaterial when
vertexColors: true is set, which GLTF loader handles automatically.

## Dynamic Colouring (no re-export needed)
If you want the heatmap to update interactively (e.g. when a user
selects an image and you want to show that image's activation pattern
rather than correlations), pass the per-image voxel vector from
image_responses.json into the same nearest-neighbour colouring pipeline
at runtime in JavaScript using a pre-built vertex-to-voxel index array.

Pre-build the index once in Python and save it:

    import json
    vertex_to_voxel = idx.tolist()           # list of length N_vertices
    close_mask      = close.tolist()         # which vertices are near V1
    with open("vertex_to_voxel.json", "w") as f:
        json.dump({"vertex_to_voxel": vertex_to_voxel,
                   "close_mask": close_mask}, f)

Then in JavaScript, for any activation vector `actVec`:

    const vtvMap = vertexToVoxelData.vertex_to_voxel;
    const colors = new Float32Array(nVertices * 3);
    for (let i = 0; i < nVertices; i++) {
        const corr = closeMask[i] ? actVec[vtvMap[i]] : 0;
        const [r,g,b] = colormap(corr, vmin, vmax);  // your JS colormap fn
        colors[i*3]=r; colors[i*3+1]=g; colors[i*3+2]=b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    material.vertexColors = true;
"""
with open(epath("brain_heatmap_guide.md"), "w", encoding="utf-8") as f:
    f.write(guide)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)
print(f"Output directory: {EXPORT_DIR}")
print(f"  images/           {N_IMAGES} JPEGs (300x300)")
print(f"  image_responses.json  {N_IMAGES} images × {Y_scaled.shape[1]} voxels")
print(f"  voxel_coords.json     {len(voxel_coords_list)} voxel MNI coords")
print(f"  correlations.json     per-voxel r + histogram")
print(f"  brain_heatmap_guide.md  GLB mesh colouring instructions")
print(f"\nAlso generated/confirmed:")
print(f"  {nii_out_path}")
print(f"\nNext step: run brain_heatmap_guide.md Step 1-5 to produce")
print(f"  brain_heatmap.glb  — ready to drop into Three.js / AI Studio")
