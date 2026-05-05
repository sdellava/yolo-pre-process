from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS_ID = 0
DEFAULT_MODEL = "yolov8n.pt"


@dataclass
class PersonDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    source: str


@dataclass
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray


@dataclass
class DetectionArtifacts:
    original_annotated: np.ndarray
    tile_layout: np.ndarray
    tile_only_annotated: np.ndarray
    merged_annotated: np.ndarray
    original_people: int
    tile_people: int
    merged_people: int
    timing: "TimingStats"


@dataclass
class TimingStats:
    load_ms: float = 0.0
    original_yolo_ms: float = 0.0
    attention_ms: float = 0.0
    tile_yolo_ms: float = 0.0
    merge_ms: float = 0.0
    save_ms: float = 0.0

    @property
    def enhanced_ms(self) -> float:
        return self.attention_ms + self.tile_yolo_ms + self.merge_ms

    @property
    def total_ms(self) -> float:
        return self.load_ms + self.original_yolo_ms + self.enhanced_ms + self.save_ms

    @property
    def latency_increase_percent(self) -> float:
        if self.original_yolo_ms <= 0:
            return 0.0
        return (self.enhanced_ms / self.original_yolo_ms) * 100.0

    @property
    def total_vs_original_percent(self) -> float:
        if self.original_yolo_ms <= 0:
            return 0.0
        return (self.total_ms / self.original_yolo_ms) * 100.0


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect people in images with YOLO, then improve small-person recall "
            "with overlapping high-resolution tiles and merge the detections."
        )
    )
    parser.add_argument("--input", required=True, help="Input image file or directory containing images.")
    parser.add_argument("--output", default="output", help="Directory where output images will be written.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO model name or local model path. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold. Default: 0.25",
    )
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width. Default: 2")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size for full-frame images. Default: 640",
    )
    parser.add_argument(
        "--tile-imgsz",
        type=int,
        default=960,
        help="YOLO inference size for zoomed tile/ROI images. Default: 960",
    )
    parser.add_argument("--tile-rows", type=int, default=2, help="Number of vertical tile bands. Default: 2")
    parser.add_argument("--tile-cols", type=int, default=2, help="Number of horizontal tile bands. Default: 2")
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.35,
        help="Tile overlap ratio between 0 and 0.8. Default: 0.35",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.30,
        help="IoU threshold used to suppress duplicate detections. Default: 0.30",
    )
    parser.add_argument(
        "--tile-edge-margin",
        type=float,
        default=0.06,
        help="Discard tile detections too close to tile borders unless the tile touches the image edge. Default: 0.06",
    )
    parser.add_argument(
        "--attention-tiles",
        type=int,
        default=8,
        help="Number of attention-selected tiles added after lightweight image pre-analysis. Default: 8",
    )
    parser.add_argument(
        "--attention-grid-rows",
        type=int,
        default=5,
        help="Rows used to score candidate attention regions. Default: 5",
    )
    parser.add_argument(
        "--attention-grid-cols",
        type=int,
        default=8,
        help="Columns used to score candidate attention regions. Default: 8",
    )
    parser.add_argument(
        "--attention-tile-scale",
        type=float,
        default=0.28,
        help="Normalized size of attention-selected tiles. Default: 0.28",
    )
    parser.add_argument(
        "--min-zoom-width",
        type=int,
        default=640,
        help="Tile crops narrower than this are upscaled before YOLO. Default: 640",
    )
    parser.add_argument(
        "--min-zoom-height",
        type=int,
        default=480,
        help="Tile crops shorter than this are upscaled before YOLO. Default: 480",
    )
    parser.add_argument(
        "--max-zoom-scale",
        type=float,
        default=2.5,
        help="Maximum pre-YOLO tile upscale factor. Default: 2.5",
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        images = sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            return images
        raise ValueError(f"No supported images found in directory: {input_path}")

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def ensure_base_output_path(base_output: Path, image_path: Path, input_root: Path) -> Path:
    relative_path = image_path.name if input_root.is_file() else image_path.relative_to(input_root)
    destination = base_output / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def output_variant_path(base_output_path: Path, suffix: str) -> Path:
    return base_output_path.with_name(f"{base_output_path.stem}_{suffix}{base_output_path.suffix}")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is not None:
        return image

    with Image.open(image_path) as pil_image:
        rgb_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to write image: {path}")


def draw_detections(image: np.ndarray, detections: list[PersonDetection], line_width: int) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        cv2.rectangle(annotated, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 200, 0), line_width)
        label = f"person {detection.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (detection.x1, max(detection.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def draw_tiles_on_original(
    image: np.ndarray,
    tiles: list[Tile],
    attention_map: np.ndarray | None = None,
) -> np.ndarray:
    annotated = image.copy()
    if attention_map is not None:
        heatmap = cv2.applyColorMap((normalize_map(attention_map) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        annotated = cv2.addWeighted(annotated, 0.72, heatmap, 0.28, 0)

    palette = [(255, 180, 40), (40, 180, 255), (130, 220, 90), (255, 110, 180), (200, 200, 70), (70, 200, 200)]
    for tile in tiles:
        color = palette[tile.index % len(palette)]
        cv2.rectangle(annotated, (tile.x, tile.y), (tile.x + tile.width - 1, tile.y + tile.height - 1), color, 2)
        label = f"tile {tile.index + 1}"
        cv2.putText(
            annotated,
            label,
            (tile.x + 8, min(tile.y + 26, annotated.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def detect_people(
    image: np.ndarray,
    model: YOLO,
    confidence: float,
    source: str,
    imgsz: int,
) -> list[PersonDetection]:
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=imgsz,
        verbose=False,
    )
    result = results[0]
    detections: list[PersonDetection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        if int(box.cls.item()) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append(
            PersonDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(box.conf.item()),
                source=source,
            )
        )
    return detections


def zoom_image_for_yolo(
    image: np.ndarray,
    min_width: int,
    min_height: int,
    max_scale: float,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image, 1.0

    scale = max(min_width / width, min_height / height, 1.0)
    scale = min(scale, max(max_scale, 1.0))
    if scale <= 1.01:
        return image, 1.0

    zoomed = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return zoomed, scale


def detect_people_in_tile(
    tile: Tile,
    model: YOLO,
    confidence: float,
    tile_imgsz: int,
    min_zoom_width: int,
    min_zoom_height: int,
    max_zoom_scale: float,
) -> list[PersonDetection]:
    zoomed_image, scale = zoom_image_for_yolo(
        tile.image,
        min_width=min_zoom_width,
        min_height=min_zoom_height,
        max_scale=max_zoom_scale,
    )
    detections = detect_people(
        image=zoomed_image,
        model=model,
        confidence=confidence,
        source=f"tile-{tile.index + 1}",
        imgsz=tile_imgsz,
    )
    if scale == 1.0:
        return detections

    return [
        PersonDetection(
            x1=int(round(detection.x1 / scale)),
            y1=int(round(detection.y1 / scale)),
            x2=int(round(detection.x2 / scale)),
            y2=int(round(detection.y2 / scale)),
            score=detection.score,
            source=detection.source,
        )
        for detection in detections
    ]


def compute_tile_length(length: int, bands: int, overlap: float) -> int:
    if bands <= 1:
        return length
    return min(length, math.ceil(length * ((1.0 / bands) + overlap * 0.5)))


def compute_tile_starts(length: int, tile_length: int, bands: int) -> list[int]:
    if bands <= 1 or tile_length >= length:
        return [0]
    return [int(round(value)) for value in np.linspace(0, length - tile_length, bands)]


def build_tiles(image: np.ndarray, rows: int, cols: int, overlap: float) -> list[Tile]:
    height, width = image.shape[:2]
    rows = max(rows, 1)
    cols = max(cols, 1)
    overlap = float(np.clip(overlap, 0.0, 0.8))

    tile_height = compute_tile_length(height, rows, overlap)
    tile_width = compute_tile_length(width, cols, overlap)
    y_starts = compute_tile_starts(height, tile_height, rows)
    x_starts = compute_tile_starts(width, tile_width, cols)

    tiles: list[Tile] = []
    index = 0
    for y in y_starts:
        for x in x_starts:
            tile_image = image[y : y + tile_height, x : x + tile_width].copy()
            tiles.append(Tile(index=index, x=x, y=y, width=tile_image.shape[1], height=tile_image.shape[0], image=tile_image))
            index += 1
    return tiles


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def build_attention_map(image: np.ndarray, detections_to_mask: list[PersonDetection]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    saturation = hsv[:, :, 1].astype(np.float32)

    attention = (
        0.55 * normalize_map(gradient)
        + 0.30 * normalize_map(laplacian)
        + 0.15 * normalize_map(saturation)
    )
    attention = cv2.GaussianBlur(attention, (0, 0), 7)

    for detection in detections_to_mask:
        pad_x = int(round((detection.x2 - detection.x1) * 0.25))
        pad_y = int(round((detection.y2 - detection.y1) * 0.25))
        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(attention.shape[1], detection.x2 + pad_x)
        y2 = min(attention.shape[0], detection.y2 + pad_y)
        attention[y1:y2, x1:x2] *= 0.2

    return normalize_map(attention)


def build_attention_tiles(
    image: np.ndarray,
    start_index: int,
    attention_map: np.ndarray,
    rows: int,
    cols: int,
    max_tiles: int,
    tile_scale: float,
) -> list[Tile]:
    height, width = image.shape[:2]
    rows = max(rows, 1)
    cols = max(cols, 1)
    max_tiles = max(max_tiles, 0)
    tile_scale = float(np.clip(tile_scale, 0.15, 0.75))
    if max_tiles == 0:
        return []

    cell_height = height / rows
    cell_width = width / cols
    tile_width = min(width, max(96, int(round(width * tile_scale))))
    tile_height = min(height, max(96, int(round(height * tile_scale))))

    candidates: list[tuple[float, int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(col * cell_width))
            y1 = int(round(row * cell_height))
            x2 = int(round((col + 1) * cell_width))
            y2 = int(round((row + 1) * cell_height))
            cell = attention_map[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            score = float(cell.mean()) + 0.25 * float(cell.max())
            center_x = int(round((x1 + x2) / 2))
            center_y = int(round((y1 + y2) / 2))
            tile_x = int(np.clip(center_x - tile_width // 2, 0, max(width - tile_width, 0)))
            tile_y = int(np.clip(center_y - tile_height // 2, 0, max(height - tile_height, 0)))
            candidates.append((score, tile_x, tile_y, tile_width, tile_height))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    tiles: list[Tile] = []
    for score, x, y, candidate_width, candidate_height in candidates:
        candidate = PersonDetection(
            x1=x,
            y1=y,
            x2=x + candidate_width,
            y2=y + candidate_height,
            score=score,
            source="attention",
        )
        if any(intersection_over_union(candidate, existing) > 0.55 for existing in tiles_as_detections(tiles)):
            continue
        tile_image = image[y : y + candidate_height, x : x + candidate_width].copy()
        tiles.append(
            Tile(
                index=start_index + len(tiles),
                x=x,
                y=y,
                width=tile_image.shape[1],
                height=tile_image.shape[0],
                image=tile_image,
            )
        )
        if len(tiles) >= max_tiles:
            break
    return tiles


def tiles_as_detections(tiles: list[Tile]) -> list[PersonDetection]:
    return [
        PersonDetection(
            x1=tile.x,
            y1=tile.y,
            x2=tile.x + tile.width,
            y2=tile.y + tile.height,
            score=1.0,
            source="tile-region",
        )
        for tile in tiles
    ]


def clip_detection(detection: PersonDetection, width: int, height: int) -> PersonDetection:
    return PersonDetection(
        x1=int(np.clip(detection.x1, 0, width - 1)),
        y1=int(np.clip(detection.y1, 0, height - 1)),
        x2=int(np.clip(detection.x2, 0, width - 1)),
        y2=int(np.clip(detection.y2, 0, height - 1)),
        score=detection.score,
        source=detection.source,
    )


def keep_tile_detection(
    detection: PersonDetection,
    tile: Tile,
    image_width: int,
    image_height: int,
    edge_margin_ratio: float,
) -> bool:
    margin_x = max(8, int(round(tile.width * edge_margin_ratio)))
    margin_y = max(8, int(round(tile.height * edge_margin_ratio)))
    center_x = (detection.x1 + detection.x2) / 2.0
    center_y = (detection.y1 + detection.y2) / 2.0

    touches_left = tile.x == 0
    touches_top = tile.y == 0
    touches_right = tile.x + tile.width >= image_width
    touches_bottom = tile.y + tile.height >= image_height

    inside_x = (touches_left or center_x >= margin_x) and (touches_right or center_x <= tile.width - margin_x)
    inside_y = (touches_top or center_y >= margin_y) and (touches_bottom or center_y <= tile.height - margin_y)
    return inside_x and inside_y


def map_tile_detections_to_original(
    tile_detections: list[PersonDetection],
    tile: Tile,
    image_width: int,
    image_height: int,
    edge_margin_ratio: float,
) -> list[PersonDetection]:
    mapped: list[PersonDetection] = []
    for detection in tile_detections:
        if not keep_tile_detection(detection, tile, image_width, image_height, edge_margin_ratio):
            continue
        mapped.append(
            clip_detection(
                PersonDetection(
                    x1=detection.x1 + tile.x,
                    y1=detection.y1 + tile.y,
                    x2=detection.x2 + tile.x,
                    y2=detection.y2 + tile.y,
                    score=detection.score,
                    source=f"tile-{tile.index + 1}",
                ),
                image_width,
                image_height,
            )
        )
    return mapped


def intersection_over_union(first: PersonDetection, second: PersonDetection) -> float:
    inter_x1 = max(first.x1, second.x1)
    inter_y1 = max(first.y1, second.y1)
    inter_x2 = min(first.x2, second.x2)
    inter_y2 = min(first.y2, second.y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
    second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def intersection_over_min_area(first: PersonDetection, second: PersonDetection) -> float:
    inter_x1 = max(first.x1, second.x1)
    inter_y1 = max(first.y1, second.y1)
    inter_x2 = min(first.x2, second.x2)
    inter_y2 = min(first.y2, second.y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
    second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
    min_area = min(first_area, second_area)
    return 0.0 if min_area <= 0 else intersection / min_area


def merge_pair(first: PersonDetection, second: PersonDetection) -> PersonDetection:
    score_sum = first.score + second.score
    if score_sum == 0:
        score_sum = 2.0
    return PersonDetection(
        x1=int(round((first.x1 * first.score + second.x1 * second.score) / score_sum)),
        y1=int(round((first.y1 * first.score + second.y1 * second.score) / score_sum)),
        x2=int(round((first.x2 * first.score + second.x2 * second.score) / score_sum)),
        y2=int(round((first.y2 * first.score + second.y2 * second.score) / score_sum)),
        score=max(first.score, second.score),
        source="merged",
    )


def merge_detections(detections: list[PersonDetection], iou_threshold: float) -> list[PersonDetection]:
    merged: list[PersonDetection] = []
    pending = sorted(detections, key=lambda detection: (detection.score, detection.source == "full"), reverse=True)

    while pending:
        candidate = pending.pop(0)
        keep_candidate = True
        for index, existing in enumerate(merged):
            iou = intersection_over_union(candidate, existing)
            overlap_ratio = intersection_over_min_area(candidate, existing)
            if iou >= iou_threshold or overlap_ratio >= 0.75:
                merged[index] = merge_pair(existing, candidate)
                keep_candidate = False
                break
        if keep_candidate:
            merged.append(candidate)

    return merged


def build_detection_artifacts(
    image: np.ndarray,
    model: YOLO,
    confidence: float,
    imgsz: int,
    tile_imgsz: int,
    line_width: int,
    tile_rows: int,
    tile_cols: int,
    tile_overlap: float,
    iou_threshold: float,
    edge_margin_ratio: float,
    attention_tiles: int,
    attention_grid_rows: int,
    attention_grid_cols: int,
    attention_tile_scale: float,
    min_zoom_width: int,
    min_zoom_height: int,
    max_zoom_scale: float,
) -> DetectionArtifacts:
    timing = TimingStats()
    image_height, image_width = image.shape[:2]
    start = time.perf_counter()
    original_detections = detect_people(
        image=image,
        model=model,
        confidence=confidence,
        source="full",
        imgsz=imgsz,
    )
    original_annotated = draw_detections(image, original_detections, line_width)
    timing.original_yolo_ms = elapsed_ms(start)

    start = time.perf_counter()
    attention_map = build_attention_map(image=image, detections_to_mask=original_detections)
    tiles = build_tiles(image=image, rows=tile_rows, cols=tile_cols, overlap=tile_overlap)
    tiles.extend(
        build_attention_tiles(
            image=image,
            start_index=len(tiles),
            attention_map=attention_map,
            rows=attention_grid_rows,
            cols=attention_grid_cols,
            max_tiles=attention_tiles,
            tile_scale=attention_tile_scale,
        )
    )
    tile_layout = draw_tiles_on_original(image, tiles, attention_map=attention_map)
    timing.attention_ms = elapsed_ms(start)

    tile_only_detections: list[PersonDetection] = []
    start = time.perf_counter()
    for tile in tiles:
        tile_detections = detect_people_in_tile(
            tile=tile,
            model=model,
            confidence=confidence,
            tile_imgsz=tile_imgsz,
            min_zoom_width=min_zoom_width,
            min_zoom_height=min_zoom_height,
            max_zoom_scale=max_zoom_scale,
        )
        tile_only_detections.extend(
            map_tile_detections_to_original(
                tile_detections=tile_detections,
                tile=tile,
                image_width=image_width,
                image_height=image_height,
                edge_margin_ratio=edge_margin_ratio,
            )
        )
    timing.tile_yolo_ms = elapsed_ms(start)

    start = time.perf_counter()
    tile_only_detections = merge_detections(tile_only_detections, iou_threshold)
    tile_only_annotated = draw_detections(image, tile_only_detections, line_width)
    merged_detections = merge_detections(original_detections + tile_only_detections, iou_threshold)
    merged_annotated = draw_detections(image, merged_detections, line_width)
    timing.merge_ms = elapsed_ms(start)

    return DetectionArtifacts(
        original_annotated=original_annotated,
        tile_layout=tile_layout,
        tile_only_annotated=tile_only_annotated,
        merged_annotated=merged_annotated,
        original_people=len(original_detections),
        tile_people=len(tile_only_detections),
        merged_people=len(merged_detections),
        timing=timing,
    )


def process_images(
    images: Iterable[Path],
    input_root: Path,
    output_dir: Path,
    model_name: str,
    confidence: float,
    imgsz: int,
    tile_imgsz: int,
    line_width: int,
    tile_rows: int,
    tile_cols: int,
    tile_overlap: float,
    iou_threshold: float,
    edge_margin_ratio: float,
    attention_tiles: int,
    attention_grid_rows: int,
    attention_grid_cols: int,
    attention_tile_scale: float,
    min_zoom_width: int,
    min_zoom_height: int,
    max_zoom_scale: float,
) -> None:
    model = YOLO(model_name)
    total_original_people = 0
    total_tile_people = 0
    total_merged_people = 0
    total_original_yolo_ms = 0.0
    total_enhanced_ms = 0.0
    total_pipeline_ms = 0.0
    image_count = 0

    for image_path in images:
        start = time.perf_counter()
        image = load_image(image_path)
        load_ms = elapsed_ms(start)
        artifacts = build_detection_artifacts(
            image=image,
            model=model,
            confidence=confidence,
            imgsz=imgsz,
            tile_imgsz=tile_imgsz,
            line_width=line_width,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            tile_overlap=tile_overlap,
            iou_threshold=iou_threshold,
            edge_margin_ratio=edge_margin_ratio,
            attention_tiles=attention_tiles,
            attention_grid_rows=attention_grid_rows,
            attention_grid_cols=attention_grid_cols,
            attention_tile_scale=attention_tile_scale,
            min_zoom_width=min_zoom_width,
            min_zoom_height=min_zoom_height,
            max_zoom_scale=max_zoom_scale,
        )
        artifacts.timing.load_ms = load_ms

        base_output_path = ensure_base_output_path(output_dir, image_path, input_root)
        original_output_path = output_variant_path(base_output_path, "A")
        tile_layout_output_path = output_variant_path(base_output_path, "B")
        tile_only_output_path = output_variant_path(base_output_path, "C")
        merged_output_path = output_variant_path(base_output_path, "D")

        start = time.perf_counter()
        save_image(original_output_path, artifacts.original_annotated)
        save_image(tile_layout_output_path, artifacts.tile_layout)
        save_image(tile_only_output_path, artifacts.tile_only_annotated)
        save_image(merged_output_path, artifacts.merged_annotated)
        artifacts.timing.save_ms = elapsed_ms(start)

        total_original_people += artifacts.original_people
        total_tile_people += artifacts.tile_people
        total_merged_people += artifacts.merged_people
        total_original_yolo_ms += artifacts.timing.original_yolo_ms
        total_enhanced_ms += artifacts.timing.enhanced_ms
        total_pipeline_ms += artifacts.timing.total_ms
        image_count += 1

        print(
            f"[OK] {image_path.name}: "
            f"orig={artifacts.original_people}, "
            f"tile={artifacts.tile_people}, "
            f"merged={artifacts.merged_people}, "
            f"delta={artifacts.merged_people - artifacts.original_people:+d}"
        )
        print(
            "     timing ms: "
            f"load={artifacts.timing.load_ms:.1f}, "
            f"orig_yolo={artifacts.timing.original_yolo_ms:.1f}, "
            f"attention={artifacts.timing.attention_ms:.1f}, "
            f"tile_yolo={artifacts.timing.tile_yolo_ms:.1f}, "
            f"merge={artifacts.timing.merge_ms:.1f}, "
            f"save={artifacts.timing.save_ms:.1f}, "
            f"total={artifacts.timing.total_ms:.1f}"
        )
        print(
            "     latency: "
            f"+{artifacts.timing.enhanced_ms:.1f} ms "
            f"(+{artifacts.timing.latency_increase_percent:.1f}% vs original YOLO only), "
            f"total={artifacts.timing.total_vs_original_percent:.1f}% of original YOLO time"
        )
        print(f"     A original bbox -> {original_output_path}")
        print(f"     B tile layout   -> {tile_layout_output_path}")
        print(f"     C tile-only     -> {tile_only_output_path}")
        print(f"     D merged bbox   -> {merged_output_path}")

    print(
        "Processed "
        f"{image_count} image(s). "
        f"Original detections: {total_original_people}. "
        f"Tile detections: {total_tile_people}. "
        f"Merged detections: {total_merged_people}. "
        f"Delta: {total_merged_people - total_original_people:+d}."
    )
    print(
        "Timing totals: "
        f"original_yolo={total_original_yolo_ms:.1f} ms, "
        f"enhanced_extra={total_enhanced_ms:.1f} ms "
        f"(+{(total_enhanced_ms / total_original_yolo_ms * 100.0) if total_original_yolo_ms > 0 else 0.0:.1f}%), "
        f"pipeline_total={total_pipeline_ms:.1f} ms."
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    images = collect_images(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_images(
        images=images,
        input_root=input_path,
        output_dir=output_dir,
        model_name=args.model,
        confidence=args.confidence,
        imgsz=args.imgsz,
        tile_imgsz=args.tile_imgsz,
        line_width=args.line_width,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        tile_overlap=args.tile_overlap,
        iou_threshold=args.iou_threshold,
        edge_margin_ratio=args.tile_edge_margin,
        attention_tiles=args.attention_tiles,
        attention_grid_rows=args.attention_grid_rows,
        attention_grid_cols=args.attention_grid_cols,
        attention_tile_scale=args.attention_tile_scale,
        min_zoom_width=args.min_zoom_width,
        min_zoom_height=args.min_zoom_height,
        max_zoom_scale=args.max_zoom_scale,
    )


if __name__ == "__main__":
    main()
