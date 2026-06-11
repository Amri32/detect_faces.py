"""
Face Identifier & Clustering
============================
Groups similar faces together and saves them into labelled folders:
  output/person_1/
  output/person_2/
  ...

Also creates a gallery folder with sample photos from each person:
  output/gallery/person_1_sample_1.jpg
  output/gallery/person_1_sample_2.jpg
  output/gallery/person_2_sample_1.jpg
  ... (no need to open each folder individually!)

In --pre-cropped mode, images that do NOT contain a valid face are
saved to output/not_face/ for review.

Uses facial landmark features (eyes, mouth, forehead, nose region, face
symmetry) for highly accurate person clustering (~4040 feature dims).

Two modes:
  1. --input FOLDER       Scan photos/videos, detect faces, then cluster.
  2. --pre-cropped FOLDER Read already-cropped face images and cluster them.

Usage:
    python face_identifier.py --input ./photos --output ./output
    python face_identifier.py --pre-cropped C:\\output\\faces --output ./grouped
    python face_identifier.py --google-drive FOLDER_URL --output ./output
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator, Optional

import cv2
import gdown
import numpy as np
import supervision as sv
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

# ── Supported file extensions ────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── Google Drive helpers ─────────────────────────────────────────────────────

_DRIVE_FOLDER_RE = re.compile(
    r"(?:https?://)?drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)"
)
_DRIVE_FILE_RE = re.compile(
    r"(?:https?://)?drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"
)


def _extract_drive_id(url_or_id: str) -> tuple[str, str]:
    m = _DRIVE_FOLDER_RE.search(url_or_id)
    if m:
        return m.group(1), "folder"
    m = _DRIVE_FILE_RE.search(url_or_id)
    if m:
        return m.group(1), "file"
    if len(url_or_id) >= 20:
        return url_or_id.strip(), "folder"
    return url_or_id.strip(), "file"


def download_from_google_drive(url_or_id: str, dest: Path) -> Path:
    ensure_dir(dest)
    drive_id, kind = _extract_drive_id(url_or_id)
    print(f"[GDRIVE] Downloading from Google Drive ({kind}: {drive_id}) ...")
    if kind == "folder":
        gdown.download_folder(id=drive_id, output=str(dest), quiet=False, use_cookies=False)
    else:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, output=str(dest), quiet=False, fuzzy=True)
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return dest


# ── Skin-tone face validation ────────────────────────────────────────────────

_SKIN_LOWER = np.array([0, 20, 50], dtype=np.uint8)
_SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)


def _is_real_face(crop_bgr: np.ndarray, min_skin_ratio: float = 0.15) -> bool:
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return False
    ratio = w / h
    if ratio < 0.5 or ratio > 1.5:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _SKIN_LOWER, _SKIN_UPPER)
    return cv2.countNonZero(mask) / (h * w) >= min_skin_ratio


# ── Face detector ────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 6,
                 min_size: tuple[int, int] = (50, 50)):
        self._frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self._profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if self._frontal.empty() or self._profile.empty():
            raise RuntimeError("Failed to load Haar cascade XML files.")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        fh, fw = frame_bgr.shape[:2]

        frontal_rects = self._frontal.detectMultiScale(
            gray, scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors, minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        profile_rects = self._profile.detectMultiScale(
            gray, scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors, minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        all_rects: list[tuple[int, int, int, int]] = []
        if len(frontal_rects):
            all_rects.extend((int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in frontal_rects)
        if len(profile_rects):
            all_rects.extend((int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in profile_rects)

        if not all_rects:
            return sv.Detections.empty()

        keep = cv2.dnn.NMSBoxes(
            bboxes=all_rects, scores=[1.0] * len(all_rects),
            score_threshold=0.0, nms_threshold=0.4,
        )
        if keep is None or len(keep) == 0:
            return sv.Detections.empty()

        valid_xyxy: list[list[float]] = []
        for i in keep.flatten():
            x1, y1, w, h = all_rects[i]
            x2, y2 = x1 + w, y1 + h
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            crop = frame_bgr[max(0, y1 - pad_y):min(fh, y2 + pad_y),
                             max(0, x1 - pad_x):min(fw, x2 + pad_x)]
            if _is_real_face(crop):
                valid_xyxy.append([float(x1), float(y1), float(x2), float(y2)])

        if not valid_xyxy:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(valid_xyxy, dtype=np.float32),
            confidence=np.ones(len(valid_xyxy), dtype=np.float32),
            class_id=np.zeros(len(valid_xyxy), dtype=np.int64),
        )


# ── Feature extraction for face clustering ───────────────────────────────────

_HASH_SIZE = 16   # 16x16 aHash
_GRID = 4         # 4x4 spatial histogram grid (finer spatial detail)
_HIST_BINS = 64
_LBP_RADIUS = 3
_LBP_POINTS = 24  # LBP with 24 points for richer texture
_HOG_CELL = 8     # HOG cell size

# Facial feature cascade classifiers (loaded once)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
_SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)


def _is_valid_face(face_bgr: np.ndarray) -> bool:
    """
    Validate whether an image crop actually contains a human face.
    Checks for eyes, face cascade detection, or skin-tone ratio.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 1) Eye detection (most reliable indicator)
    eyes = _EYE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(5, 5), maxSize=(w // 2, h // 2),
    )
    if len(eyes) >= 1:
        return True

    # 2) Face cascade check on the crop itself
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(w // 4, h // 4),
    )
    if len(faces) >= 1:
        return True

    # 3) Skin-tone ratio check (fallback)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(mask) / (h * w)
    if skin_ratio > 0.15:
        return True

    return False


def _lbp_hist(gray: np.ndarray, radius: int = _LBP_RADIUS,
              points: int = _LBP_POINTS) -> np.ndarray:
    """Compute LBP (Local Binary Pattern) histogram for texture."""
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint16)
    for i in range(points):
        angle = 2 * np.pi * i / points
        dx = int(round(radius * np.cos(angle)))
        dy = int(round(radius * np.sin(angle)))
        shifted = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)
        lbp += ((gray <= shifted).astype(np.uint16) << i)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 65536))
    return hist.astype(np.float32) / (hist.sum() + 1e-10)


def _hog_descriptor(gray: np.ndarray, cell_size: int = _HOG_CELL) -> np.ndarray:
    """Compute a simple HOG (Histogram of Oriented Gradients) feature vector."""
    img = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

    n_cells = 128 // cell_size
    hog_features: list[float] = []
    for r in range(n_cells):
        for c in range(n_cells):
            cell_mag = magnitude[r * cell_size:(r + 1) * cell_size,
                                 c * cell_size:(c + 1) * cell_size]
            cell_ang = angle[r * cell_size:(r + 1) * cell_size,
                             c * cell_size:(c + 1) * cell_size]
            hist, _ = np.histogram(cell_ang, bins=9, range=(0, 180),
                                   weights=cell_mag)
            hog_features.extend(hist)
    arr = np.array(hog_features, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-10)


def _region_hist(region: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute normalised HSV colour histogram of a face region."""
    if region.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([h, s, v]).astype(np.float32)
    return hist / (hist.sum() + 1e-10)


def _facial_landmark_features(face_bgr: np.ndarray) -> np.ndarray:
    """
    Detect eyes, mouth, nose within a normalised 128x128 face crop.
    Returns features:
      - Eye region colour hist (48)
      - Mouth region colour hist (48)
      - Forehead region colour hist (48)
      - Geometric features: eye_y, eye_gap, mouth_y, face_symmetry (8)
      Total: ~152 extra dims
    """
    face = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    fh, fw = face.shape[:2]

    # ── Detect eyes ──────────────────────────────────────────────────────
    eyes = _EYE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(10, 10), maxSize=(50, 50),
    )
    # Sort left-to-right by x
    eyes = sorted(eyes, key=lambda r: r[0]) if len(eyes) else []

    # ── Detect mouth/smile ───────────────────────────────────────────────
    # Search only lower half for mouth
    lower_half = gray[fh // 2:, :]
    mouths = _SMILE_CASCADE.detectMultiScale(
        lower_half, scaleFactor=1.3, minNeighbors=5,
        minSize=(15, 8), maxSize=(80, 40),
    )
    mouths = sorted(mouths, key=lambda r: r[2] * r[3], reverse=True)[:1]

    # ── Region colour histograms ─────────────────────────────────────────
    # Eye region (top third)
    eye_region = face[:fh // 3, :]
    eye_hist = _region_hist(eye_region)

    # Mouth region (bottom third)
    mouth_region = face[2 * fh // 3:, :]
    mouth_hist = _region_hist(mouth_region)

    # Forehead region (top 20%)
    forehead_region = face[:fh // 5, :]
    forehead_hist = _region_hist(forehead_region)

    # ── Geometric features ───────────────────────────────────────────────
    geo: list[float] = []

    if len(eyes) >= 2:
        # Two eyes found
        left, right = eyes[0], eyes[1]
        left_cx = left[0] + left[2] / 2
        right_cx = right[0] + right[2] / 2
        left_cy = left[1] + left[3] / 2
        right_cy = right[1] + right[3] / 2
        eye_gap = abs(right_cx - left_cx) / fw
        eye_y = (left_cy + right_cy) / 2 / fh
        eye_size = (left[2] * left[3] + right[2] * right[3]) / (2 * fw * fh)
        eye_tilt = (right_cy - left_cy) / fh  # tilt angle proxy
        geo.extend([eye_gap, eye_y, eye_size, eye_tilt])
    elif len(eyes) == 1:
        eye = eyes[0]
        eye_gap = 0.0
        eye_y = (eye[1] + eye[3] / 2) / fh
        eye_size = (eye[2] * eye[3]) / (fw * fh)
        geo.extend([eye_gap, eye_y, eye_size, 0.0])
    else:
        geo.extend([0.0, 0.35, 0.0, 0.0])  # defaults

    if len(mouths) >= 1:
        mouth = mouths[0]
        mouth_y = (fh // 2 + mouth[1] + mouth[3] / 2) / fh
        mouth_w = mouth[2] / fw
        mouth_size = (mouth[2] * mouth[3]) / (fw * fh)
        geo.extend([mouth_y, mouth_w, mouth_size])
    else:
        geo.extend([0.75, 0.4, 0.0])  # defaults

    # Face symmetry (left half vs right half gray histogram correlation)
    left_gray = gray[:, :fw // 2]
    right_gray = cv2.flip(gray[:, fw // 2:], 1)
    min_h = min(left_gray.shape[0], right_gray.shape[0])
    min_w = min(left_gray.shape[1], right_gray.shape[1])
    h_left = cv2.calcHist([left_gray[:min_h, :min_w]], [0], None, [32], [0, 256])
    h_right = cv2.calcHist([right_gray[:min_h, :min_w]], [0], None, [32], [0, 256])
    symmetry = float(cv2.compareHist(h_left, h_right, cv2.HISTCMP_CORREL))
    geo.append(symmetry)

    geo_arr = np.array(geo, dtype=np.float32)
    return np.concatenate([eye_hist, mouth_hist, forehead_hist, geo_arr])


def _extract_features(face_bgr: np.ndarray) -> np.ndarray:
    """
    Build a rich combined feature vector for a face crop:
      [aHash (256)] + [spatial gray hist (1024)]
    + [LBP texture (256)] + [HOG shape (2304)]
    + [HSV color hist (48)]
    + [facial landmarks: eye/mouth/forehead regions + geometry (~152)]
    Total ~4040 dimensions for high accuracy.
    """
    face = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    # 1) Average hash (256 bits)
    small = cv2.resize(gray, (_HASH_SIZE, _HASH_SIZE), interpolation=cv2.INTER_AREA)
    ahash = (small > small.mean()).astype(np.float32).flatten()

    # 2) Spatial grayscale histogram (4x4 grid, 64 bins each = 1024 floats)
    cell_h, cell_w = gray.shape[0] // _GRID, gray.shape[1] // _GRID
    spat_hist_parts: list[np.ndarray] = []
    for r in range(_GRID):
        for c in range(_GRID):
            cell = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            h = cv2.calcHist([cell], [0], None, [_HIST_BINS], [0, 256]).flatten()
            spat_hist_parts.append(h / (h.sum() + 1e-10))
    spat_hist = np.concatenate(spat_hist_parts).astype(np.float32)

    # 3) LBP texture histogram (256 bins)
    lbp = _lbp_hist(gray)

    # 4) HOG shape descriptor (9 bins x 16x16 cells = 2304 floats)
    hog = _hog_descriptor(gray)

    # 5) HSV color histogram (H:16 + S:16 + V:16 = 48 bins)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color_hist /= (color_hist.sum() + 1e-10)

    # 6) Facial landmark features (eyes, mouth, forehead regions + geometry)
    landmarks = _facial_landmark_features(face_bgr)

    return np.concatenate([ahash, spat_hist, lbp, hog, color_hist, landmarks])


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_media(folder: Path) -> Generator[Path, None, None]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_face(frame: np.ndarray, x1: float, y1: float,
              x2: float, y2: float, padding: float = 0.15) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    px, py = int(fw * padding), int(fh * padding)
    crop = frame[max(0, int(y1) - py):min(h, int(y2) + py),
                 max(0, int(x1) - px):min(w, int(x2) + px)]
    return crop if crop.size > 0 else None


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_faces(
    features: list[np.ndarray],
    threshold: float = 0.35,
) -> list[int]:
    """
    Cluster face feature vectors using agglomerative clustering
    with a two-pass refinement for higher accuracy:
      Pass 1: initial clustering with cosine distance
      Pass 2: for each cluster, compute centroid and re-assign outliers
              that are far from their cluster centroid.
    Returns a list of integer labels (0-based person IDs).
    """
    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0]

    matrix = np.stack(features)

    # ── Pass 1: initial clustering ───────────────────────────────────────
    dist_vec = pdist(matrix, metric="cosine")
    Z = linkage(dist_vec, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    # ── Pass 2: refine – reassign outliers ───────────────────────────────
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        centroids = {}
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() > 0:
                centroids[lbl] = matrix[mask].mean(axis=0)

        refined = labels.copy()
        for i in range(n):
            best_lbl = labels[i]
            best_dist = 1.0
            for lbl, centroid in centroids.items():
                d = float(1.0 - np.dot(
                    matrix[i] / (np.linalg.norm(matrix[i]) + 1e-10),
                    centroid / (np.linalg.norm(centroid) + 1e-10),
                ))
                if d < best_dist:
                    best_dist = d
                    best_lbl = lbl
            refined[i] = best_lbl
        labels = refined

    # Convert to 0-based contiguous IDs
    unique, remapped = np.unique(labels, return_inverse=True)
    return remapped.tolist()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group similar faces together into person_1/, person_2/ ..."
    )
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="Local folder with photos/videos (will detect faces).")
    parser.add_argument("--pre-cropped", "-pc", type=Path, default=None,
                        metavar="FOLDER",
                        help="Folder of already-cropped face images (skip detection).")
    parser.add_argument("--google-drive", "-gd", type=str, default=None,
                        metavar="URL_OR_ID",
                        help="Google Drive shared folder URL or ID.")
    parser.add_argument("--output", "-o", type=Path, default=Path("./output"),
                        help="Output folder (default: ./output).")
    parser.add_argument("--cluster-threshold", "-ct", type=float, default=0.35,
                        metavar="T",
                        help=(
                            "Cosine distance threshold for clustering. "
                            "Lower = stricter (more persons). Default: 0.35."
                        ))
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=6)
    parser.add_argument("--min-size", type=int, nargs=2, default=[50, 50],
                        metavar=("W", "H"))
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Process every N-th frame in videos.")
    parser.add_argument("--flat", "-f", action="store_true",
                        help="Save all faces in one flat folder (no person_X/ subfolders). "
                             "Files are still named person_X_0001.jpg.")
    parser.add_argument("--gallery-only", "-g", action="store_true",
                        help="Only create gallery folder with 1 sample per person. "
                             "No person folders at all.")
    args = parser.parse_args()

    output_dir: Path = args.output
    ensure_dir(output_dir)

    # ── Resolve input ────────────────────────────────────────────────────
    gdrive_temp: Optional[Path] = None
    pre_cropped_mode = args.pre_cropped is not None

    if args.pre_cropped:
        input_dir = args.pre_cropped
    elif args.google_drive:
        gdrive_temp = Path(tempfile.mkdtemp(prefix="gdrive_"))
        input_dir = download_from_google_drive(args.google_drive, gdrive_temp)
        print(f"[GDRIVE] Downloaded to: {gdrive_temp}\n")
    elif args.input:
        input_dir = args.input
    else:
        parser.error("Provide --input, --pre-cropped, or --google-drive.")

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Each entry: (source_file, face_crop_bgr, feature_vector)
    all_faces: list[tuple[str, np.ndarray, np.ndarray]] = []
    not_face_dir = output_dir / "not_face"  # only used in pre-cropped mode

    if pre_cropped_mode:
        # ── Pre-cropped mode: read face images directly ──────────────────
        print("[MODE] Pre-cropped faces (no detection, direct clustering)\n")
        face_images = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        print(f"[INFO] Found {len(face_images)} face image(s) in: {input_dir}\n")

        if not face_images:
            print("[WARN] No images found. Exiting.")
            sys.exit(0)

        ensure_dir(not_face_dir)
        not_face_count = 0

        for img_path in tqdm(face_images, desc="Extracting features", unit="face"):
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue

            # Validate: does this image actually contain a face?
            if not _is_valid_face(img):
                # Not a face — save to not_face folder
                cv2.imwrite(str(not_face_dir / img_path.name), img)
                not_face_count += 1
                continue

            feat = _extract_features(img)
            all_faces.append((img_path.name, img, feat))

        if not_face_count > 0:
            print(f"[INFO] {not_face_count} image(s) moved to: {not_face_dir}\n")

    else:
        # ── Detection mode: detect faces in photos/videos ────────────────
        print("[INFO] Initializing face detector ...")
        detector = FaceDetector(
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=tuple(args.min_size),
        )
        print("[INFO] Detector ready.\n")

        media_files = list(iter_media(input_dir))
        n_img = sum(1 for p in media_files if p.suffix.lower() in IMAGE_EXTS)
        n_vid = sum(1 for p in media_files if p.suffix.lower() in VIDEO_EXTS)
        print(f"[INFO] Found {len(media_files)} media file(s) "
              f"({n_img} image(s), {n_vid} video(s)) in: {input_dir}\n")

        if not media_files:
            print("[WARN] No media found. Exiting.")
            sys.exit(0)

        print("[PHASE 1] Detecting faces and extracting features ...")

        for media_path in tqdm(media_files, desc="Scanning", unit="file"):
            ext = media_path.suffix.lower()

            if ext in IMAGE_EXTS:
                img = cv2.imread(str(media_path))
                if img is None:
                    continue
                detections = detector.detect(img)
                if detections.xyxy is not None:
                    for x1, y1, x2, y2 in detections.xyxy:
                        crop = crop_face(img, x1, y1, x2, y2)
                        if crop is not None:
                            feat = _extract_features(crop)
                            all_faces.append((media_path.name, crop, feat))

            else:  # video
                cap = cv2.VideoCapture(str(media_path))
                if not cap.isOpened():
                    continue
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % args.sample_rate == 0:
                        detections = detector.detect(frame)
                        if detections.xyxy is not None:
                            for x1, y1, x2, y2 in detections.xyxy:
                                crop = crop_face(frame, x1, y1, x2, y2)
                                if crop is not None:
                                    feat = _extract_features(crop)
                                    all_faces.append(
                                        (f"{media_path.stem}_f{frame_idx:06d}", crop, feat)
                                    )
                    frame_idx += 1
                cap.release()

    total_faces = len(all_faces)
    print(f"\n[INFO] Total faces loaded: {total_faces}")

    if total_faces == 0:
        print("[WARN] No faces found. Nothing to cluster.")
        if gdrive_temp and gdrive_temp.exists():
            shutil.rmtree(gdrive_temp, ignore_errors=True)
        sys.exit(0)

    # ── Phase 2: Cluster faces by person ─────────────────────────────────
    print(f"\n[PHASE 2] Clustering {total_faces} face(s) (threshold={args.cluster_threshold}) ...")
    features = [f[2] for f in all_faces]
    labels = cluster_faces(features, threshold=args.cluster_threshold)
    n_persons = max(labels) + 1
    print(f"[INFO] Identified {n_persons} unique person(s).\n")

    # ── Phase 3: Save faces grouped by person ────────────────────────────
    print("[PHASE 3] Saving faces grouped by person ...")

    flat_mode = args.flat
    gallery_only = args.gallery_only

    person_counters: dict[int, int] = {}
    csv_rows: list[dict] = []
    gallery_dir = output_dir / "gallery"
    ensure_dir(gallery_dir)

    # In flat mode, create a single faces/ subfolder
    if flat_mode and not gallery_only:
        flat_dir = output_dir / "faces"
        ensure_dir(flat_dir)

    for idx, (source, crop, _feat) in enumerate(all_faces):
        person_id = labels[idx]
        person_counters[person_id] = person_counters.get(person_id, 0)

        person_label = f"person_{person_id + 1}"
        count = person_counters[person_id]
        filename = f"{person_label}_{count:04d}.jpg"

        # ── Gallery: save 1 sample per person ───────────────────────
        if count == 0:
            gallery_name = f"{person_label}.jpg"
            cv2.imwrite(str(gallery_dir / gallery_name), crop)

        # ── Save faces (unless gallery-only mode) ───────────────────
        if not gallery_only:
            if flat_mode:
                # Flat: save into single faces/ folder
                cv2.imwrite(str(flat_dir / filename), crop)
            else:
                # Normal: save into person_X/ subfolder
                person_dir = output_dir / person_label
                ensure_dir(person_dir)
                cv2.imwrite(str(person_dir / filename), crop)

        csv_rows.append({
            "person": person_label,
            "source_file": source,
            "saved_as": filename,
        })
        person_counters[person_id] += 1

    # ── Summary CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "source_file", "saved_as"])
        writer.writeheader()
        writer.writerows(csv_rows)

    counts_path = output_dir / "person_counts.csv"
    with open(counts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "total_faces"])
        writer.writeheader()
        for pid in range(n_persons):
            writer.writerow({
                "person": f"person_{pid + 1}",
                "total_faces": person_counters.get(pid, 0),
            })

    # ── Print summary ────────────────────────────────────────────────────
    print("=" * 60)
    print("  FACE IDENTIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total faces found    : {total_faces}")
    print(f"  Unique persons       : {n_persons}")
    print(f"  Output folder        : {output_dir.resolve()}")
    print(f"  Summary CSV          : {csv_path.resolve()}")
    print(f"  Person counts CSV    : {counts_path.resolve()}")
    print(f"  Gallery folder       : {gallery_dir.resolve()}")
    if flat_mode and not gallery_only:
        print(f"  Flat faces folder    : {flat_dir.resolve()}")
    if pre_cropped_mode and not_face_dir.exists():
        nf_count = len(list(not_face_dir.iterdir()))
        if nf_count > 0:
            print(f"  Not-face folder      : {not_face_dir.resolve()} ({nf_count} images)")
    print("=" * 60)

    print("\nPer-person breakdown:")
    for pid in range(n_persons):
        label = f"person_{pid + 1}"
        cnt = person_counters.get(pid, 0)
        folder = output_dir / label
        print(f"  {label:<15s} -> {cnt} face(s)  [{folder}]")

    print("\n[DONE]")

    if gdrive_temp and gdrive_temp.exists():
        shutil.rmtree(gdrive_temp, ignore_errors=True)


if __name__ == "__main__":
    main()
