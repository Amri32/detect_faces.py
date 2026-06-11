"""
Face Detection & Counter
========================
Scans a folder containing photos and videos, detects faces using
OpenCV Haar Cascades + supervision, saves cropped face images,
and reports how many faces were found per file and in total.

Usage:
    python detect_faces.py --input ./my_photos --output ./output
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

# ── Supported file extensions ────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── Haar Cascade Face Detector wrapper ───────────────────────────────────────

class FaceDetector:
    """Face detector using OpenCV Haar Cascades (no external download required)."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple[int, int] = (30, 30)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        """Run face detection on a BGR OpenCV frame, return supervision Detections."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(rects) == 0:
            return sv.Detections.empty()

        xyxy_list = []
        for x, y, w, h in rects:
            xyxy_list.append([float(x), float(y), float(x + w), float(y + h)])

        return sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.ones(len(xyxy_list), dtype=np.float32),
            class_id=np.zeros(len(xyxy_list), dtype=np.int64),
        )


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
    padding: float = 0.15,
) -> list[Path]:
    """Crop each detected face from *frame* and save as individual image."""
    ensure_dir(save_dir)
    h, w = frame.shape[:2]
    saved: list[Path] = []

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

        out_path = save_dir / f"{prefix}_face_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved.append(out_path)

    return saved


# ── Image processing ─────────────────────────────────────────────────────────

def process_image(
    path: Path,
    detector: FaceDetector,
    output_dir: Path,
) -> dict:
    """Detect faces in a single image. Returns result dict."""
    img = cv2.imread(str(path))
    if img is None:
        return {"file": str(path), "faces": 0, "error": "cannot read"}

    detections = detector.detect(img)
    n_faces = len(detections)

    # Save cropped faces
    face_dir = output_dir / "faces" / path.stem
    saved_faces: list[Path] = []
    if n_faces > 0 and detections.xyxy is not None:
        saved_faces = crop_and_save(img, detections.xyxy, face_dir, path.stem)

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

    face_dir = output_dir / "faces" / path.stem
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
                saved = crop_and_save(frame, detections.xyxy, face_dir, prefix)
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
        required=True,
        help="Folder containing photos and/or videos to scan.",
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
        default=5,
        help=(
            "Minimum neighbours each candidate rectangle should have to be kept. "
            "Higher = fewer false positives (default: 5)."
        ),
    )
    parser.add_argument(
        "--min-size",
        type=int,
        nargs=2,
        default=[30, 30],
        metavar=("W", "H"),
        help="Minimum face size in pixels (default: 30 30).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Process every N-th frame in videos (default: 5).",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(output_dir)

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
        sys.exit(0)

    # ── Process ──────────────────────────────────────────────────────────
    results: list[dict] = []
    grand_total_faces = 0
    grand_total_saved = 0

    for media_path in tqdm(media_files, desc="Processing files", unit="file"):
        ext = media_path.suffix.lower()

        if ext in IMAGE_EXTS:
            res = process_image(media_path, detector, output_dir)
        else:
            res = process_video(
                media_path, detector, output_dir, args.sample_rate
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


if __name__ == "__main__":
    main()
