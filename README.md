# Capture Face

Face detection & identification system using OpenCV Haar Cascades + Supervision.
Detects faces in photos/videos, crops them, deduplicates, and clusters by person.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `opencv-python` — face detection (Haar cascades) & image processing
- `supervision` — bounding box visualization & annotation
- `numpy` — array operations
- `scipy` — clustering (agglomerative hierarchy)
- `tqdm` — progress bars
- `gdown` — Google Drive download

---

## Workflow Overview

```
Step 1: detect_faces.py        Step 2: face_identifier.py
┌─────────────────────┐       ┌──────────────────────────┐
│  Photos / Videos    │──────▶│  Cropped faces           │
│  (or Google Drive)  │  faces│  → Clustered by person   │
│                     │       │  → person_1, person_2 .. │
└─────────────────────┘       └──────────────────────────┘
```

---

## 1. `detect_faces.py` — Detect & Crop Faces

Scans photos and videos, detects human faces, crops them, removes duplicates,
and saves all face images into a flat `faces/` folder.

### Basic Usage

```bash
# Scan a local folder
python detect_faces.py --input ./photos --output C:\output

# Scan from Google Drive
python detect_faces.py --google-drive FOLDER_URL --output C:\output

# Show help
python detect_faces.py --help
```

### All Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | — | Local folder with photos/videos |
| `--google-drive` | `-gd` | — | Google Drive shared folder URL or file ID |
| `--output` | `-o` | `./output` | Output folder |
| `--scale-factor` | | `1.1` | Haar cascade scale factor (smaller = more accurate, slower) |
| `--min-neighbors` | | `6` | Min neighbors per detection (higher = fewer false positives) |
| `--min-size` | | `50 50` | Minimum face size in pixels (W H) |
| `--sample-rate` | | `5` | Process every N-th frame in videos |
| `--no-dedup` | | off | Disable deduplication (save every face, even duplicates) |
| `--dedup-sensitivity` | | `20` | Hash distance for dedup (lower = stricter) |

### Output Structure

```
C:\output\
├── faces\                  ← All cropped face images
│   ├── photo1_f000001.jpg
│   ├── photo2_f000003.jpg
│   └── ...
├── annotated\              ← Photos with face boxes drawn
│   ├── photo1.jpg
│   └── ...
└── face_counts.csv         ← Per-file face counts
```

### How It Works

1. **Detection** — Dual Haar cascade (frontal + profile) with NMS
2. **Validation** — Skin-tone check (HSV) + aspect ratio filter to reduce false positives
3. **Deduplication** — Perceptual hashing (aHash) skips near-identical faces
4. **Annotation** — Uses `supervision` library to draw bounding boxes on original images

---

## 2. `face_identifier.py` — Cluster Faces by Person

Groups similar face images into person clusters using rich feature extraction
(~4040 dimensions) and two-pass agglomerative clustering.

### Basic Usage

```bash
# Mode A: From already-cropped faces (recommended workflow)
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped

# Mode B: From raw photos/videos (detects faces first)
python face_identifier.py --input ./photos --output C:\output\grouped

# Mode C: From Google Drive
python face_identifier.py --google-drive FOLDER_URL --output C:\output\grouped

# Flat mode (no person subfolders)
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped --flat

# Gallery-only mode (just samples, no person folders)
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped --gallery-only

# Show help
python face_identifier.py --help
```

### All Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | — | Local folder with photos/videos (detects faces first) |
| `--pre-cropped` | `-pc` | — | Folder of already-cropped face images (skip detection) |
| `--google-drive` | `-gd` | — | Google Drive shared folder URL or ID |
| `--output` | `-o` | `./output` | Output folder |
| `--cluster-threshold` | `-ct` | `0.35` | Cosine distance threshold (lower = stricter, more persons) |
| `--flat` | `-f` | off | Save all faces in one `faces/` folder (no person subfolders) |
| `--gallery-only` | `-g` | off | Only create gallery with 1 sample per person |
| `--scale-factor` | | `1.1` | Haar cascade scale factor (detection mode only) |
| `--min-neighbors` | | `6` | Min neighbors per detection (detection mode only) |
| `--min-size` | | `50 50` | Minimum face size in pixels (detection mode only) |
| `--sample-rate` | | `5` | Process every N-th frame in videos (detection mode only) |

### Output Modes

#### Normal Mode (default)
```
output/
├── gallery/                ← 1 sample photo per person
│   ├── person_1.jpg
│   ├── person_2.jpg
│   └── ...
├── person_1/               ← All faces of person 1
│   ├── person_1_0000.jpg
│   └── ...
├── person_2/               ← All faces of person 2
│   └── ...
├── not_face/               ← Images rejected as non-face (pre-cropped mode only)
├── summary.csv
└── person_counts.csv
```

#### Flat Mode (`--flat` / `-f`)
```
output/
├── gallery/                ← 1 sample per person
│   └── ...
├── faces/                  ← ALL faces in one folder
│   ├── person_1_0000.jpg
│   ├── person_2_0000.jpg
│   └── ...
├── not_face/
├── summary.csv
└── person_counts.csv
```

#### Gallery-Only Mode (`--gallery-only` / `-g`)
```
output/
├── gallery/                ← 1 sample per person (that's it!)
│   ├── person_1.jpg
│   ├── person_2.jpg
│   └── ...
├── summary.csv
└── person_counts.csv
```

### Face Validation (`not_face/` folder)

In `--pre-cropped` mode, each image is validated before clustering:
- **Eye detection** — checks for at least 1 eye (Haar cascade)
- **Face cascade** — checks overall face shape
- **Skin-tone ratio** — if >15% skin pixels, likely a face

Images that fail ALL checks are saved to `output/not_face/` for review.

### Feature Extraction (~4040 dimensions)

| Feature | Description | Dims |
|---------|-------------|------|
| aHash | Average perceptual hash | 256 |
| Spatial gray histogram | 4x4 grid, 64 bins each | 1024 |
| LBP texture | Local Binary Patterns (24-point) | 256 |
| HOG shape | Histogram of Oriented Gradients | 2304 |
| HSV color | Full-face color histogram | 48 |
| Eye region | Color histogram of top 1/3 | 48 |
| Mouth region | Color histogram of bottom 1/3 | 48 |
| Forehead region | Color histogram of top 20% (hair) | 48 |
| Geometry | Eye gap, position, size, tilt, mouth metrics | 7 |
| Symmetry | Left-right face correlation | 1 |

### Clustering

- **Pass 1**: Agglomerative clustering with cosine distance
- **Pass 2**: Centroid-based refinement (re-assigns outliers to nearest cluster)

### Tuning `--cluster-threshold`

| Value | Effect |
|-------|--------|
| `0.20` | Very strict — many persons, fewer merges |
| `0.30` | Strict |
| `0.35` | **Default** — balanced |
| `0.45` | Relaxed — fewer persons, more merges |
| `0.60` | Very relaxed — may merge different people |

---

## Complete Workflow Example

```bash
# Step 1: Detect and crop all faces from your photo collection
python detect_faces.py --input D:\MyPhotos --output C:\output

# Step 2: Cluster the cropped faces by person (gallery-only for quick preview)
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped --gallery-only

# Review gallery/ folder to see how many people were identified.
# If results look good, run full clustering:
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped

# Or use flat mode if you just want everything in one folder:
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped --flat
```

### Google Drive Workflow

```bash
# Step 1: Download and detect from Google Drive
python detect_faces.py --google-drive https://drive.google.com/drive/folders/YOUR_FOLDER_ID --output C:\output

# Step 2: Cluster the results
python face_identifier.py --pre-cropped C:\output\faces --output C:\output\grouped
```

---

## Tips

- **Too many false positives?** Increase `--min-neighbors` (e.g., `8`) or `--min-size` (e.g., `80 80`)
- **Missing faces?** Decrease `--min-neighbors` (e.g., `4`) or `--min-size` (e.g., `30 30`)
- **Duplicate faces saved?** Decrease `--dedup-sensitivity` (e.g., `10`)
- **Too many persons in clustering?** Increase `--cluster-threshold` (e.g., `0.45`)
- **Different people merged together?** Decrease `--cluster-threshold` (e.g., `0.25`)
- **Slow on videos?** Increase `--sample-rate` (e.g., `10` = every 10th frame)
