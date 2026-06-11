"""
Face Detection & Counter
========================
Scans a folder containing photos and videos, detects faces using
OpenCV Haar Cascades + supervision, saves cropped face images,
and reports how many faces were found per file and in total.

Supports Google Drive: pass a shared folder URL or ID via --google-drive.

Usage:
    python detect_faces.py --input ./my_photos --output ./output
    python detect_faces.py --google-drive https://drive.google.com/drive/folders/XXXXX
    python detect_faces.py --google-drive FOLDER_ID --output ./output
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
    """
    Return (id, kind) where kind is 'folder' or 'file'.
    Accepts a full URL or a bare ID.
    """
    m = _DRIVE_FOLDER_RE.search(url_or_id)
    if m:
        return m.group(1), "folder"
    m = _DRIVE_FILE_RE.search(url_or_id)
    if m:
        return m.group(1), "file"
    # Bare ID – assume folder if long, else file
    if len(url_or_id) >= 20:
        return url_or_id.strip(), "folder"
    return url_or_id.strip(), "file"


def download_from_google_drive(
    url_or_id: str,
    dest: Path,
) -> Path:
    """
    Download files from a shared Google Drive folder (or single file) into
    *dest*.  Returns the path to the local folder containing the downloads.
    """
    ensure_dir(dest)
    drive_id, kind = _extract_drive_id(url_or_id)

    print(f"[GDRIVE] Downloading from Google Drive ({kind}: {drive_id}) ...")

    if kind == "folder":
        gdown.download_folder(
            id=drive_id,
            output=str(dest),
            quiet=False,
            use_cookies=False,
        )
    else:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, output=str(dest), quiet=False, fuzzy=True)

    # If gdown created a single sub-folder, descend into it
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]

    return dest


# ── Haar Cascade Face Detector wrapper ───────────────────────────────────────

# Skin-tone range in HSV for face validation
_SKIN_LOWER = np.array([0, 20, 50], dtype=np.uint8)
_SKIN_UPPER = np.array([25, 180, 255], dtype=np.uint8)
_MIN_SKIN_RATIO = 0.15  # at least 15% of crop must be skin-tone


def _is_real_face(crop_bgr: np.ndarray, min_skin_ratio: float = _MIN_SKIN_RATIO) -> bool:
    """
    Validate whether a cropped region is likely a real face by checking:
      1. Aspect ratio is roughly square (0.5 – 1.5)
      2. Contains a meaningful amount of skin-tone pixels
    Returns True if the crop passes both checks.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return False

    # ── 1. Aspect ratio check ────────────────────────────────────────────
    ratio = w / h
    if ratio < 0.5 or ratio > 1.5:
        return False

    # ── 2. Skin-tone check (HSV) ─────────────────────────────────────────
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _SKIN_LOWER, _SKIN_UPPER)
    skin_pixels = int(cv2.countNonZero(mask))
    total_pixels = h * w
    if skin_pixels / total_pixels < min_skin_ratio:
        return False

    return True


class FaceDetector:
    """
    Face detector using OpenCV Haar Cascades (no external download required).
    Uses BOTH frontal-face and profile-face cascades, then validates each
    candidate with skin-tone + aspect-ratio filtering to reduce false positives.
    """

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
        """Run face detection on a BGR OpenCV frame, return supervision Detections."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        fh, fw = frame_bgr.shape[:2]

        # ── Detect with frontal + profile cascades ───────────────────────
        frontal_rects = self._frontal.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        profile_rects = self._profile.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # ── Merge & deduplicate overlapping boxes ────────────────────────
        all_rects: list[tuple[int, int, int, int]] = []
        if len(frontal_rects):
            all_rects.extend((int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in frontal_rects)
        if len(profile_rects):
            all_rects.extend((int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in profile_rects)

        if not all_rects:
            return sv.Detections.empty()

        # Non-maximum suppression to remove overlapping boxes
        boxes = np.array(all_rects, dtype=np.float32)
        xyxy = np.column_stack([boxes[:, 0], boxes[:, 1],
                                boxes[:, 0] + boxes[:, 2],
                                boxes[:, 1] + boxes[:, 3]])
        keep = cv2.dnn.NMSBoxes(
            bboxes=all_rects,
            scores=[1.0] * len(all_rects),
            score_threshold=0.0,
            nms_threshold=0.4,
        )
        if keep is None or len(keep) == 0:
            return sv.Detections.empty()

        # ── Validate each candidate (skin tone + aspect ratio) ───────────
        valid_xyxy: list[list[float]] = []
        for i in keep.flatten():
            x1, y1, w, h = all_rects[i]
            x2, y2 = x1 + w, y1 + h
            cx1 = max(0, x1 - int(w * 0.1))
            cy1 = max(0, y1 - int(h * 0.1))
            cx2 = min(fw, x2 + int(w * 0.1))
            cy2 = min(fh, y2 + int(h * 0.1))
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if _is_real_face(crop):
                valid_xyxy.append([float(x1), float(y1), float(x2), float(y2)])

        if not valid_xyxy:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(valid_xyxy, dtype=np.float32),
            confidence=np.ones(len(valid_xyxy), dtype=np.float32),
            class_id=np.zeros(len(valid_xyxy), dtype=np.int64),
        )


# ── Face deduplication ────────────────────────────────────────────────────────

class FaceDeduplicator:
    """
    Tracks perceptual hashes of saved face crops so that the same person's
    face is only saved once.  Uses a combination of:
      - average hash (aHash) – fast structural similarity
      - grayscale histogram correlation
    A face is considered a duplicate when BOTH metrics exceed their thresholds.
    """

    HASH_SIZE = 16  # 16x16 = 256-bit hash

    def __init__(self, hash_threshold: int = 20, hist_threshold: float = 0.85):
        """
        Args:
            hash_threshold: Max Hamming distance to consider same face (lower = stricter).
            hist_threshold: Min histogram correlation to consider same face [0-1].
        """
        self._hash_threshold = hash_threshold
        self._hist_threshold = hist_threshold
        self._hashes: list[np.ndarray] = []      # stored aHash arrays
        self._hists: list[np.ndarray] = []        # stored histograms

    @staticmethod
    def _ahash(img: np.ndarray, size: int = 16) -> np.ndarray:
        """Compute average hash (aHash) of an image."""
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        avg = gray.mean()
        return (gray > avg).astype(np.uint8).flatten()

    @staticmethod
    def _hist(img: np.ndarray) -> np.ndarray:
        """Compute normalised grayscale histogram."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        return h / (h.sum() + 1e-10)

    def is_duplicate(self, face_crop: np.ndarray) -> bool:
        """Return True if *face_crop* matches an already-seen face."""
        if not self._hashes:
            return False

        new_hash = self._ahash(face_crop, self.HASH_SIZE)
        new_hist = self._hist(face_crop)

        for stored_hash, stored_hist in zip(self._hashes, self._hists):
            hamming = int(np.sum(new_hash != stored_hash))
            corr = float(cv2.compareHist(
                new_hist.astype(np.float32),
                stored_hist.astype(np.float32),
                cv2.HISTCMP_CORREL,
            ))
            # Duplicate only when BOTH metrics agree
            if hamming <= self._hash_threshold and corr >= self._hist_threshold:
                return True

        return False

    def register(self, face_crop: np.ndarray) -> None:
        """Store the hash/histogram of a newly saved face."""
        self._hashes.append(self._ahash(face_crop, self.HASH_SIZE))
        self._hists.append(self._hist(face_crop))

    @property
    def count(self) -> int:
        return len(self._hashes)


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_media(folder: Path) -> Generator[Path, None, None]:
    """Yield all image and video paths inside *folder* (recursive)."""
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_and_save(
    frame: np.ndarray,
    xyxy: np.ndarray,
    save_dir: Path,
    prefix: str,
    dedup: FaceDeduplicator | None = None,
    padding: float = 0.15,
) -> list[Path]:
    """
    Crop each detected face from *frame* and save as individual image.
    If *dedup* is provided, skip faces that have already been saved.
    """
    ensure_dir(save_dir)
    h, w = frame.shape[:2]
    saved: list[Path] = []
    save_idx = 0

    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        fw, fh = x2 - x1, y2 - y1
        px, py = int(fw * padding), int(fh * padding)
        cx1 = max(0, int(x1) - px)
        cy1 = max(0, int(y1) - py)
        cx2 = min(w, int(x2) + px)
        cy2 = min(h, int(y2) + py)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # ── Deduplication check ──────────────────────────────────────────
        if dedup is not None and dedup.is_duplicate(crop):
            continue  # same face already saved – skip

        out_path = save_dir / f"{prefix}_face_{save_idx:03d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved.append(out_path)

        if dedup is not None:
            dedup.register(crop)

        save_idx += 1

    return saved


# ── Image processing ─────────────────────────────────────────────────────────

def process_image(
    path: Path,
    detector: FaceDetector,
    output_dir: Path,
    dedup: FaceDeduplicator | None = None,
) -> dict:
    """Detect faces in a single image. Returns result dict."""
    img = cv2.imread(str(path))
    if img is None:
        return {"file": str(path), "faces": 0, "error": "cannot read"}

    detections = detector.detect(img)
    n_faces = len(detections)

    # Save cropped faces (all in one flat folder)
    face_dir = output_dir / "faces"
    saved_faces: list[Path] = []
    if n_faces > 0 and detections.xyxy is not None:
        saved_faces = crop_and_save(img, detections.xyxy, face_dir, path.stem, dedup)

    # Save annotated image with supervision
    annotated = img.copy()
    if n_faces > 0:
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        labels = [f"face {i+1}" for i in range(n_faces)]
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

    annotated_dir = output_dir / "annotated"
    ensure_dir(annotated_dir)
    cv2.imwrite(str(annotated_dir / path.name), annotated)

    return {
        "file": path.name,
        "type": "image",
        "faces": n_faces,
        "saved_faces": len(saved_faces),
    }


# ── Video processing ─────────────────────────────────────────────────────────

def process_video(
    path: Path,
    detector: FaceDetector,
    output_dir: Path,
    dedup: FaceDeduplicator | None = None,
    sample_rate: int = 5,
) -> dict:
    """
    Detect faces in a video.  Reads every *sample_rate*-th frame to keep
    processing time reasonable.  Returns result dict.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": str(path), "faces": 0, "error": "cannot open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    face_dir = output_dir / "faces"
    ensure_dir(face_dir)

    total_faces = 0
    saved_faces = 0
    frame_idx = 0
    processed = 0
    pbar = tqdm(
        total=max(1, total_frames // sample_rate),
        desc=f"  {path.name}",
        unit="frame",
        leave=False,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            detections = detector.detect(frame)
            n = len(detections)
            total_faces += n

            if n > 0 and detections.xyxy is not None:
                prefix = f"{path.stem}_f{frame_idx:06d}"
                saved = crop_and_save(frame, detections.xyxy, face_dir, prefix, dedup)
                saved_faces += len(saved)

            processed += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()

    return {
        "file": path.name,
        "type": "video",
        "frames_processed": processed,
        "faces": total_faces,
        "saved_faces": saved_faces,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect faces in a folder of photos/videos, save crops & count."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Local folder containing photos and/or videos to scan.",
    )
    parser.add_argument(
        "--google-drive", "-gd",
        type=str,
        default=None,
        metavar="URL_OR_ID",
        help=(
            "Google Drive shared folder URL or file/folder ID. "
            "Files will be downloaded to a temp folder before processing."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Folder where results will be saved (default: ./output).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar cascade scale factor per image pyramid step (default: 1.1).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=6,
        help=(
            "Minimum neighbours each candidate rectangle should have to be kept. "
            "Higher = fewer false positives (default: 6)."
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=[50, 50],
        metavar=("W", "H"),
        help="Minimum face size in pixels (default: 50 50).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Process every N-th frame in videos (default: 5).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable face deduplication (save every detected face, even duplicates).",
    )
    parser.add_argument(
        "--dedup-sensitivity",
        type=int,
        default=20,
        metavar="N",
        help="Hash distance threshold for dedup (lower = stricter, default: 20).",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    ensure_dir(output_dir)

    # ── Resolve input source ─────────────────────────────────────────────
    gdrive_temp: Optional[Path] = None

    if args.google_drive:
        gdrive_temp = Path(tempfile.mkdtemp(prefix="gdrive_"))
        input_dir = download_from_google_drive(args.google_drive, gdrive_temp)
        print(f"[GDRIVE] Downloaded to: {gdrive_temp}\n")
    elif args.input:
        input_dir = args.input
    else:
        parser.error("You must provide either --input or --google-drive.")

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Init detector ────────────────────────────────────────────────────
    print("[INFO] Initializing OpenCV Haar Cascade Face Detector ...")
    detector = FaceDetector(
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
        min_size=tuple(args.min_size),
    )
    print("[INFO] Detector ready.\n")

    # ── Collect media files ──────────────────────────────────────────────
    media_files = list(iter_media(input_dir))
    n_images = sum(1 for p in media_files if p.suffix.lower() in IMAGE_EXTS)
    n_videos = sum(1 for p in media_files if p.suffix.lower() in VIDEO_EXTS)

    print(
        f"[INFO] Found {len(media_files)} media file(s) "
        f"({n_images} image(s), {n_videos} video(s)) in: {input_dir}\n"
    )

    if not media_files:
        print("[WARN] No images or videos found. Exiting.")
        if gdrive_temp and gdrive_temp.exists():
            shutil.rmtree(gdrive_temp, ignore_errors=True)
        sys.exit(0)

    # ── Init deduplicator ────────────────────────────────────────────────
    dedup: FaceDeduplicator | None = None
    if not args.no_dedup:
        dedup = FaceDeduplicator(hash_threshold=args.dedup_sensitivity)
        print("[INFO] Face deduplication enabled (same face will only be saved once).")
        print(f"       Hash threshold: {args.dedup_sensitivity} (lower = stricter)")
    else:
        print("[INFO] Face deduplication disabled – saving all faces.")

    # ── Process ──────────────────────────────────────────────────────────
    results: list[dict] = []
    grand_total_faces = 0
    grand_total_saved = 0

    for media_path in tqdm(media_files, desc="Processing files", unit="file"):
        ext = media_path.suffix.lower()

        if ext in IMAGE_EXTS:
            res = process_image(media_path, detector, output_dir, dedup)
        else:
            res = process_video(
                media_path, detector, output_dir, dedup, args.sample_rate
            )

        results.append(res)
        grand_total_faces += res["faces"]
        grand_total_saved += res.get("saved_faces", 0)

    # ── Summary CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "type", "faces", "saved_faces", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FACE DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Files processed   : {len(results)}")
    print(f"  Total faces found : {grand_total_faces}")
    print(f"  Face crops saved  : {grand_total_saved}")
    if dedup is not None:
        skipped = grand_total_faces - grand_total_saved
        print(f"  Duplicates skipped: {skipped}")
        print(f"  Unique faces saved  : {dedup.count}")
    print(f"  Output folder     : {output_dir.resolve()}")
    print(f"  Summary CSV       : {csv_path.resolve()}")
    print("=" * 60)

    print("\nPer-file breakdown:")
    for r in results:
        status = f"[{r['type']:5s}] {r['file']:<40s} -> {r['faces']} face(s)"
        if r.get("error"):
            status += f"  (ERROR: {r['error']})"
        print(f"  {status}")

    print("\n[DONE]")

    # ── Cleanup Google Drive temp folder ─────────────────────────────────
    if gdrive_temp and gdrive_temp.exists():
        print(f"[INFO] Cleaning up temp folder: {gdrive_temp}")
        shutil.rmtree(gdrive_temp, ignore_errors=True)


if __name__ == "__main__":
    main()
