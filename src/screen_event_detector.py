import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = e
else:
    CV2_IMPORT_ERROR = None

try:
    import mss
except Exception as e:
    mss = None
    MSS_IMPORT_ERROR = e
else:
    MSS_IMPORT_ERROR = None

try:
    import pytesseract
except Exception as e:
    pytesseract = None
    PYTESSERACT_IMPORT_ERROR = e
else:
    PYTESSERACT_IMPORT_ERROR = None


@dataclass
class ScreenEvent:
    event_type: str
    confidence: float
    created_at: float
    details: dict = field(default_factory=dict)


@dataclass
class DeathDetectionResult:
    detected: bool
    final_score: float
    template_score: float
    shape_score: float
    dark_ratio: float
    white_ratio: float
    cols_with_white: float
    rows_with_white: float
    log_confidence: float
    reason: str = ""


@dataclass
class OcrResult:
    text: str
    confidence: float
    reason: str = ""
    preprocess_mode: str = ""
    roi: tuple = ()
    scale: float = 0.0


@dataclass
class DeathDetectionThresholds:
    template_min_score: float = 0.55
    shape_min_template_score: float = 0.40
    template_min_dark_ratio: float = 0.06
    shape_min_dark_ratio: float = 0.12
    white_ratio_min: float = 0.015
    white_ratio_max: float = 0.18
    shape_white_ratio_min: float = 0.04
    shape_white_ratio_max: float = 0.14
    min_cols_with_white: float = 0.18
    min_rows_with_white: float = 0.08
    shape_min_cols_with_white: float = 0.20
    shape_min_rows_with_white: float = 0.10


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_int_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if max(box) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2, iy2


def crop_roi(image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _as_int_box(roi, width, height)
    return image[y1:y2, x1:x2]


def normalize_ocr_text(text: str) -> str:
    return "".join(str(text).split())


def _resolve_template_path(path: str) -> Path:
    template_path = Path(path)
    if not template_path.is_absolute():
        template_path = _project_root() / template_path
    return template_path


def load_template(path: str) -> Optional[np.ndarray]:
    if not path or cv2 is None:
        return None
    template_path = _resolve_template_path(path)
    if not template_path.exists():
        print(f"[SCREEN] template missing path={template_path}", flush=True)
        return None
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        print(f"[SCREEN] template unreadable path={template_path}", flush=True)
        return None
    return template


def load_templates(path: str) -> List[np.ndarray]:
    primary_path = _resolve_template_path(path)
    candidates = [primary_path]
    if primary_path.parent.exists():
        candidates.extend(sorted(primary_path.parent.glob(f"{primary_path.stem}*.png")))

    templates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        template = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE) if cv2 is not None else None
        if template is not None and template.size > 0:
            templates.append(template)
        elif candidate == primary_path:
            print(f"[SCREEN] template unreadable path={candidate}", flush=True)
    if not templates:
        print(f"[SCREEN] template missing path={primary_path}", flush=True)
    return templates


def match_template_score(gray_crop: np.ndarray, templates) -> float:
    if cv2 is None or not templates:
        return 0.0
    if isinstance(templates, np.ndarray):
        templates = [templates]

    crop_h, crop_w = gray_crop.shape[:2]
    scores = []
    for template in templates:
        tpl_h, tpl_w = template.shape[:2]
        if crop_h < tpl_h or crop_w < tpl_w:
            scale = min(crop_w / max(1, tpl_w), crop_h / max(1, tpl_h), 1.0)
            new_w = max(8, int(tpl_w * scale))
            new_h = max(8, int(tpl_h * scale))
            template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray_crop, template, cv2.TM_CCOEFF_NORMED)
        if result.size:
            scores.append(float(result.max()))
    return max(scores) if scores else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _frame_source_name(cfg) -> str:
    return str(getattr(cfg, "screen_frame_source", "mss") or "mss").strip().lower()


def _source_override(cfg, base_attr: str, obs_attr: str):
    if _frame_source_name(cfg) == "obs_virtual_camera":
        value = getattr(cfg, obs_attr, None)
        if value not in (None, ""):
            return value
    return getattr(cfg, base_attr)


def effective_death_roi(cfg):
    return _source_override(cfg, "death_event_roi", "death_event_roi_obs")


def effective_death_text_roi(cfg):
    return _source_override(cfg, "death_event_text_roi", "death_event_text_roi_obs")


def effective_template_path(cfg) -> str:
    return str(_source_override(cfg, "death_event_template_path", "death_event_template_path_obs") or "")


def _range_score(value: float, low: float, high: float, peak: float) -> float:
    if value < low or value > high:
        return 0.0
    if value <= peak:
        return _clamp01((value - low) / max(1e-6, peak - low))
    return _clamp01((high - value) / max(1e-6, high - peak))


def get_thresholds(cfg) -> DeathDetectionThresholds:
    return DeathDetectionThresholds(
        template_min_score=float(_source_override(
            cfg, "death_event_min_template_score", "death_event_min_template_score_obs"
        )),
        shape_min_template_score=float(_source_override(
            cfg, "death_event_shape_min_template_score", "death_event_shape_min_template_score_obs"
        )),
        template_min_dark_ratio=float(getattr(cfg, "death_event_template_min_dark_ratio", 0.06)),
        shape_min_dark_ratio=float(getattr(cfg, "death_event_shape_min_dark_ratio", 0.12)),
        white_ratio_min=float(getattr(cfg, "death_event_white_ratio_min", 0.015)),
        white_ratio_max=float(getattr(cfg, "death_event_white_ratio_max", 0.18)),
        shape_white_ratio_min=float(getattr(cfg, "death_event_shape_white_ratio_min", 0.04)),
        shape_white_ratio_max=float(getattr(cfg, "death_event_shape_white_ratio_max", 0.14)),
        min_cols_with_white=float(getattr(cfg, "death_event_min_cols_with_white", 0.18)),
        min_rows_with_white=float(getattr(cfg, "death_event_min_rows_with_white", 0.08)),
        shape_min_cols_with_white=float(getattr(cfg, "death_event_shape_min_cols_with_white", 0.20)),
        shape_min_rows_with_white=float(getattr(cfg, "death_event_shape_min_rows_with_white", 0.10)),
    )


def _resize_for_detection(image: np.ndarray, cfg) -> np.ndarray:
    target_width = int(getattr(cfg, "screen_capture_width", 0))
    target_height = int(getattr(cfg, "screen_capture_height", 0))
    if (
        cv2 is not None
        and target_width > 0
        and target_height > 0
        and (image.shape[1] != target_width or image.shape[0] != target_height)
    ):
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return image


def compute_shape_score(
    dark_ratio: float,
    white_ratio: float,
    cols_with_white: float,
    rows_with_white: float,
    thresholds: DeathDetectionThresholds,
) -> float:
    dark_score = _clamp01((dark_ratio - 0.04) / 0.30)
    white_score = _range_score(white_ratio, thresholds.white_ratio_min, thresholds.white_ratio_max, 0.09)
    col_score = _clamp01(cols_with_white / 0.55)
    row_score = _clamp01(rows_with_white / 0.55)
    return _clamp01(dark_score * 0.35 + white_score * 0.25 + col_score * 0.25 + row_score * 0.15)


def detect_death_event(image: np.ndarray, cfg, template=None) -> DeathDetectionResult:
    if cv2 is None:
        return DeathDetectionResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "cv2_unavailable")

    image = _resize_for_detection(image, cfg)

    death_roi = effective_death_roi(cfg)
    text_roi = effective_death_text_roi(cfg)
    thresholds = get_thresholds(cfg)

    death_crop = crop_roi(image, death_roi)
    text_crop = crop_roi(image, text_roi)
    death_gray = cv2.cvtColor(death_crop, cv2.COLOR_RGB2GRAY)
    text_gray = cv2.cvtColor(text_crop, cv2.COLOR_RGB2GRAY)

    white_mask = text_gray > 205
    dark_ratio = float((death_gray < 45).mean())
    white_ratio = float(white_mask.mean())
    cols_with_white = float(np.mean(np.any(white_mask, axis=0)))
    rows_with_white = float(np.mean(np.any(white_mask, axis=1)))
    template_score = match_template_score(text_gray, template)
    shape_score = compute_shape_score(dark_ratio, white_ratio, cols_with_white, rows_with_white, thresholds)
    final_score = max(template_score, template_score * 0.8 + shape_score * 0.2)
    log_confidence = final_score

    white_in_template_range = thresholds.white_ratio_min <= white_ratio <= thresholds.white_ratio_max
    white_in_shape_range = thresholds.shape_white_ratio_min <= white_ratio <= thresholds.shape_white_ratio_max
    template_text_like = (
        cols_with_white >= thresholds.min_cols_with_white
        and rows_with_white >= thresholds.min_rows_with_white
    )
    shape_text_like = (
        cols_with_white >= thresholds.shape_min_cols_with_white
        and rows_with_white >= thresholds.shape_min_rows_with_white
    )

    template_detected = (
        template_score >= thresholds.template_min_score
        and dark_ratio >= thresholds.template_min_dark_ratio
        and white_in_template_range
        and template_text_like
    )
    shape_detected = (
        template_score >= thresholds.shape_min_template_score
        and dark_ratio >= thresholds.shape_min_dark_ratio
        and white_in_shape_range
        and shape_text_like
    )
    detected = template_detected or shape_detected

    if detected:
        reason = "detected"
    elif white_ratio > thresholds.white_ratio_max:
        reason = "too_much_text"
    elif template_score < thresholds.shape_min_template_score:
        reason = "low_template_score"
    elif dark_ratio < thresholds.template_min_dark_ratio and white_ratio < thresholds.white_ratio_min:
        reason = "roi_out_of_position_candidate"
    elif shape_score < 0.45:
        reason = "low_shape_score"
    else:
        reason = "low_score"

    return DeathDetectionResult(
        detected=detected,
        final_score=final_score,
        template_score=template_score,
        shape_score=shape_score,
        dark_ratio=dark_ratio,
        white_ratio=white_ratio,
        cols_with_white=cols_with_white,
        rows_with_white=rows_with_white,
        log_confidence=log_confidence,
        reason=reason,
    )


def _build_capture_target(monitor: dict, cfg) -> dict:
    region = getattr(cfg, "screen_capture_region", None)
    if region is None:
        return monitor

    try:
        left, top, width, height = region
    except Exception:
        print(f"[SCREEN] invalid capture region={region}", flush=True)
        return monitor

    if width <= 0 or height <= 0:
        print(f"[SCREEN] invalid capture region={region}", flush=True)
        return monitor

    return {
        "left": int(monitor["left"] + left),
        "top": int(monitor["top"] + top),
        "width": int(width),
        "height": int(height),
    }


def capture_crop(sct, monitor: dict, cfg) -> np.ndarray:
    width = int(getattr(cfg, "screen_capture_width", 640))
    height = int(getattr(cfg, "screen_capture_height", 360))
    target = _build_capture_target(monitor, cfg)
    grab = np.asarray(sct.grab(target))[:, :, :3]
    image = grab[:, :, ::-1].copy()
    if cv2 is not None and width > 0 and height > 0:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


class FrameSource:
    def start(self) -> None:
        pass

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class MSSFrameSource(FrameSource):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sct = None
        self.monitor = None

    def start(self) -> None:
        if mss is None:
            raise RuntimeError(f"mss_unavailable:{MSS_IMPORT_ERROR}")
        monitor_index = int(getattr(self.cfg, "screen_capture_monitor_index", 1))
        self.sct = mss.mss()
        monitors = self.sct.monitors
        if monitor_index < 1 or monitor_index >= len(monitors):
            raise RuntimeError(f"monitor_not_found index={monitor_index}")
        self.monitor = monitors[monitor_index]
        target = _build_capture_target(self.monitor, self.cfg)
        print(f"[SCREEN] monitor index={monitor_index}", flush=True)
        print(
            f"[SCREEN] capture region left={target['left']} top={target['top']} "
            f"width={target['width']} height={target['height']}",
            flush=True,
        )

    def read(self) -> np.ndarray:
        if self.sct is None or self.monitor is None:
            raise RuntimeError("frame_source_not_started")
        return capture_crop(self.sct, self.monitor, self.cfg)

    def stop(self) -> None:
        if self.sct is not None:
            self.sct.close()
        self.sct = None
        self.monitor = None


class OBSVirtualCameraFrameSource(FrameSource):
    def __init__(self, cfg):
        self.cfg = cfg
        self.capture = None

    def start(self) -> None:
        if cv2 is None:
            raise RuntimeError(f"cv2_unavailable:{CV2_IMPORT_ERROR}")
        camera_index = int(getattr(self.cfg, "obs_virtual_camera_index", 0))
        width = int(getattr(self.cfg, "obs_virtual_camera_width", 1920))
        height = int(getattr(self.cfg, "obs_virtual_camera_height", 1080))
        fps = int(getattr(self.cfg, "obs_virtual_camera_fps", 30))
        print("[SCREEN] frame source=obs_virtual_camera", flush=True)
        print(f"[SCREEN] obs camera index={camera_index}", flush=True)
        print(f"[SCREEN] obs requested width={width} height={height} fps={fps}", flush=True)

        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"obs_virtual_camera_unavailable index={camera_index}")

        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, fps)

        warmup_frames = max(0, int(getattr(self.cfg, "obs_virtual_camera_warmup_frames", 5)))
        for _ in range(warmup_frames):
            self.capture.read()

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        print(
            f"[SCREEN] obs actual width={actual_width} height={actual_height} fps={actual_fps:.1f}",
            flush=True,
        )

    def read(self) -> np.ndarray:
        if self.capture is None:
            raise RuntimeError("frame_source_not_started")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("obs_virtual_camera_frame_unavailable")
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = int(getattr(self.cfg, "screen_capture_width", 640))
        height = int(getattr(self.cfg, "screen_capture_height", 360))
        if width > 0 and height > 0 and (image.shape[1] != width or image.shape[0] != height):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return image

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None


def create_frame_source(cfg) -> FrameSource:
    frame_source = str(getattr(cfg, "screen_frame_source", "mss") or "mss").strip().lower()
    if frame_source == "mss":
        return MSSFrameSource(cfg)
    if frame_source == "obs_virtual_camera":
        return OBSVirtualCameraFrameSource(cfg)
    raise RuntimeError(f"unknown_frame_source source={frame_source}")


class DeathDetector:
    def __init__(self, cfg, templates):
        self.cfg = cfg
        self.templates = templates

    def detect(self, image: np.ndarray) -> DeathDetectionResult:
        return detect_death_event(image, self.cfg, self.templates)

    def ocr(self, image: np.ndarray) -> OcrResult:
        return ocr_death_text(image, self.cfg)


def format_detection_metrics(result: DeathDetectionResult) -> str:
    return (
        f"final_score={result.final_score:.2f} template_score={result.template_score:.2f} "
        f"shape_score={result.shape_score:.2f} dark={result.dark_ratio:.3f} "
        f"white={result.white_ratio:.3f} cols_with_white={result.cols_with_white:.3f} "
        f"rows_with_white={result.rows_with_white:.3f}"
    )


def detection_debug_regions(image: np.ndarray, cfg) -> dict:
    original_height, original_width = image.shape[:2]
    resized = _resize_for_detection(image, cfg)
    height, width = resized.shape[:2]
    death_roi = effective_death_roi(cfg)
    text_roi = effective_death_text_roi(cfg)
    death_box = _as_int_box(death_roi, width, height)
    text_box = _as_int_box(text_roi, width, height)
    dx1, dy1, dx2, dy2 = death_box
    tx1, ty1, tx2, ty2 = text_box
    return {
        "frame": resized,
        "original_width": original_width,
        "original_height": original_height,
        "frame_width": width,
        "frame_height": height,
        "death_roi": death_roi,
        "death_box": death_box,
        "death_crop": resized[dy1:dy2, dx1:dx2],
        "text_roi": text_roi,
        "text_box": text_box,
        "text_crop": resized[ty1:ty2, tx1:tx2],
    }


def format_detection_geometry(image: np.ndarray, cfg) -> str:
    regions = detection_debug_regions(image, cfg)
    return (
        f"frame_width={regions['frame_width']} frame_height={regions['frame_height']} "
        f"death_roi={format_roi(regions['death_roi'])} death_box={regions['death_box']} "
        f"text_roi={format_roi(regions['text_roi'])} text_box={regions['text_box']}"
    )


def save_detection_debug_frames(
    source_name: str,
    image: np.ndarray,
    cfg,
    result: Optional[DeathDetectionResult] = None,
    output_dir: Optional[str] = None,
    save_roi_crops: bool = True,
    event_label: str = "",
    metrics_extra: Optional[dict] = None,
    save_full: bool = True,
    save_overlay: bool = True,
    save_roi: Optional[bool] = None,
    save_metrics: bool = True,
) -> Path:
    if cv2 is None:
        raise RuntimeError("cv2_unavailable")
    base_dir = Path(output_dir or getattr(cfg, "screen_event_debug_dir", "debug/screen_event"))
    if not base_dir.is_absolute():
        base_dir = _project_root() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    source = _safe_name(str(source_name or _frame_source_name(cfg) or "screen"))
    label = _safe_name(str(event_label or "").strip())
    prefix = f"{timestamp}_{source}"
    if label:
        prefix = f"{prefix}_{label}"
    regions = detection_debug_regions(image, cfg)
    frame = regions["frame"]
    should_save_roi = save_roi_crops if save_roi is None else save_roi

    full_path = base_dir / f"{prefix}_full.png"
    overlay_path = base_dir / f"{prefix}_overlay.png"
    if save_full:
        cv2.imwrite(str(full_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if save_overlay:
        overlay = frame.copy()
        dx1, dy1, dx2, dy2 = regions["death_box"]
        tx1, ty1, tx2, ty2 = regions["text_box"]
        cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if should_save_roi:
        cv2.imwrite(str(base_dir / f"{prefix}_death_roi.png"), cv2.cvtColor(regions["death_crop"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base_dir / f"{prefix}_text_roi.png"), cv2.cvtColor(regions["text_crop"], cv2.COLOR_RGB2BGR))

    metrics = {
        "timestamp": timestamp,
        "source": str(source_name or _frame_source_name(cfg) or "screen"),
        "frame_source": _frame_source_name(cfg),
        "original_width": regions["original_width"],
        "original_height": regions["original_height"],
        "resized_width": regions["frame_width"],
        "resized_height": regions["frame_height"],
        "frame_width": regions["frame_width"],
        "frame_height": regions["frame_height"],
        "effective_death_roi": list(regions["death_roi"]),
        "effective_text_roi": list(regions["text_roi"]),
        "death_roi": list(regions["death_roi"]),
        "death_box": list(regions["death_box"]),
        "text_roi": list(regions["text_roi"]),
        "text_box": list(regions["text_box"]),
        "screen_frame_source": _frame_source_name(cfg),
        "template_path": effective_template_path(cfg),
    }
    if result is not None:
        metrics.update({
            "detected": result.detected,
            "raw_detected": result.detected,
            "reason": result.reason,
            "detection_reason": result.reason,
            "final_score": result.final_score,
            "confidence": result.final_score,
            "template_score": result.template_score,
            "shape_score": result.shape_score,
            "dark_ratio": result.dark_ratio,
            "white_ratio": result.white_ratio,
            "cols_with_white": result.cols_with_white,
            "rows_with_white": result.rows_with_white,
            "log_confidence": result.log_confidence,
        })
    if metrics_extra:
        metrics.update(metrics_extra)
    if save_metrics:
        with (base_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    return base_dir


def prune_debug_files(directory: Path, max_files: int) -> None:
    if max_files <= 0 or not directory.exists():
        return
    suffixes = ("_full.png", "_overlay.png", "_death_roi.png", "_text_roi.png", "_metrics.json")
    files = [path for path in directory.iterdir() if path.is_file() and path.name.endswith(suffixes)]
    excess = len(files) - max_files
    if excess <= 0:
        return
    for path in sorted(files, key=lambda item: item.stat().st_mtime)[:excess]:
        try:
            path.unlink()
        except OSError as e:
            print(f"[SCREEN] debug prune skip path={path} error={e}", flush=True)


def _ocr_enabled(cfg) -> bool:
    return getattr(cfg, "screen_event_mode", "death_detect_only") == "death_ocr"


def configure_tesseract(cfg) -> None:
    if pytesseract is None:
        return
    command = str(getattr(cfg, "death_event_ocr_tesseract_cmd", "") or "").strip()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    if getattr(cfg, "death_event_ocr_debug_log", False):
        active_command = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        source = "config" if command else "PATH"
        print(f"[SCREEN] OCR tesseract_cmd={active_command} source={source}", flush=True)


def _ocr_preprocess_mode(cfg) -> str:
    return str(getattr(cfg, "death_event_ocr_preprocess_mode", "default") or "default")


def preprocess_ocr_crop(image: np.ndarray, cfg) -> np.ndarray:
    scale = max(1.0, float(getattr(cfg, "death_event_ocr_scale", 3.0)))
    mode = _ocr_preprocess_mode(cfg)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if mode == "default":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    elif mode == "threshold":
        _, processed = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    elif mode == "adaptive":
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    elif mode == "invert_threshold":
        _, processed = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    elif mode == "sharpen_threshold":
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharpened = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)
        _, processed = cv2.threshold(sharpened, 185, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    return cv2.copyMakeBorder(processed, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)


def parse_roi(value: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must have 4 comma-separated values: {value}")
    x1, y1, x2, y2 = (float(part) for part in parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI must be x1,y1,x2,y2 with x2>x1 and y2>y1: {value}")
    return x1, y1, x2, y2


def format_roi(roi) -> str:
    return ",".join(f"{float(value):.4f}".rstrip("0").rstrip(".") for value in roi)


def save_ocr_debug_images(
    source_name: str,
    crop: np.ndarray,
    processed: np.ndarray,
    cfg,
    output_dir: Optional[str] = None,
) -> None:
    if cv2 is None:
        return
    base_dir = Path(output_dir or getattr(cfg, "death_event_ocr_debug_dir", "debug/ocr"))
    if not base_dir.is_absolute():
        base_dir = _project_root() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    roi = getattr(cfg, "death_event_ocr_roi", ())
    scale = max(1.0, float(getattr(cfg, "death_event_ocr_scale", 3.0)))
    mode = _ocr_preprocess_mode(cfg)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    roi_text = "_".join(str(v).replace(".", "p") for v in roi)
    stem = _safe_name(Path(source_name).stem if source_name else "screen")
    prefix = f"{stem}_roi-{roi_text}_scale-{str(scale).replace('.', 'p')}_mode-{mode}_{timestamp}"
    cv2.imwrite(str(base_dir / f"{prefix}_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(base_dir / f"{prefix}_processed.png"), processed)


def ocr_death_text(image: np.ndarray, cfg) -> OcrResult:
    mode = _ocr_preprocess_mode(cfg)
    scale = max(1.0, float(getattr(cfg, "death_event_ocr_scale", 3.0)))
    ocr_roi = getattr(cfg, "death_event_ocr_roi", getattr(cfg, "death_event_roi", (0.32, 0.21, 0.67, 0.54)))
    if not _ocr_enabled(cfg):
        return OcrResult("", 0.0, "disabled", mode, ocr_roi, scale)
    if cv2 is None:
        return OcrResult("", 0.0, "cv2_unavailable", mode, ocr_roi, scale)
    if pytesseract is None:
        return OcrResult("", 0.0, f"pytesseract_unavailable:{PYTESSERACT_IMPORT_ERROR}", mode, ocr_roi, scale)

    configure_tesseract(cfg)
    crop = crop_roi(image, ocr_roi)
    processed = preprocess_ocr_crop(crop, cfg)
    if getattr(cfg, "death_event_ocr_save_debug_images", False):
        source_name = str(getattr(cfg, "death_event_ocr_debug_source_name", "screen"))
        save_ocr_debug_images(source_name, crop, processed, cfg)
    lang = str(getattr(cfg, "death_event_ocr_lang", "jpn+eng") or "jpn+eng")
    tesseract_config = str(getattr(cfg, "death_event_ocr_config", "--psm 6") or "--psm 6")
    min_confidence = float(getattr(cfg, "death_event_ocr_min_confidence", 0.0))

    try:
        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        return OcrResult("", 0.0, f"ocr_error:{e}", mode, ocr_roi, scale)

    parts = []
    confidences = []
    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        normalized = normalize_ocr_text(text)
        if not normalized:
            continue
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = -1.0
        if confidence_value >= 0:
            confidences.append(confidence_value)
        parts.append(normalized)

    text = normalize_ocr_text("".join(parts))
    confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    if not text:
        return OcrResult("", confidence, "empty", mode, ocr_roi, scale)
    if confidence < min_confidence:
        return OcrResult(text, confidence, "low_confidence", mode, ocr_roi, scale)
    return OcrResult(text, confidence, "ok", mode, ocr_roi, scale)


class ScreenEventDetector:
    def __init__(self, cfg, output_queue: queue.Queue):
        self.cfg = cfg
        self.output_queue = output_queue
        self.running = False
        self.thread = None
        self.templates = []
        self.detector = None
        self.frame_source = None
        self.last_emit_time = 0.0
        self.last_low_score_log_time = 0.0

    def start(self):
        if not getattr(self.cfg, "screen_event_enabled", False):
            print("[SCREEN] disabled", flush=True)
            return
        print("[SCREEN] enabled", flush=True)
        if cv2 is None:
            print(f"[SCREEN] disabled reason=cv2_unavailable error={CV2_IMPORT_ERROR}", flush=True)
            return
        frame_source_name = str(getattr(self.cfg, "screen_frame_source", "mss") or "mss").strip().lower()
        if frame_source_name == "mss" and mss is None:
            print(f"[SCREEN] disabled reason=mss_unavailable error={MSS_IMPORT_ERROR}", flush=True)
            return
        self.templates = load_templates(effective_template_path(self.cfg))
        if not self.templates:
            print("[SCREEN] disabled reason=template_unavailable", flush=True)
            return
        print(f"[SCREEN] templates loaded count={len(self.templates)}", flush=True)
        self.detector = DeathDetector(self.cfg, self.templates)
        self.frame_source = create_frame_source(self.cfg)
        if _ocr_enabled(self.cfg):
            configure_tesseract(self.cfg)
            if pytesseract is None:
                print(f"[SCREEN] OCR disabled reason=pytesseract_unavailable error={PYTESSERACT_IMPORT_ERROR}", flush=True)
            else:
                print("[SCREEN] OCR enabled backend=tesseract", flush=True)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[SCREEN] capture started", flush=True)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.frame_source is not None:
            self.frame_source.stop()

    def _should_log_skip(self, now: float) -> bool:
        if getattr(self.cfg, "screen_event_debug_log", False):
            return True
        interval = max(1.0, float(getattr(self.cfg, "screen_event_log_every_sec", 10.0)))
        if now - self.last_low_score_log_time >= interval:
            self.last_low_score_log_time = now
            return True
        return False

    def _run(self):
        interval = max(0.2, float(getattr(self.cfg, "screen_capture_interval_sec", 0.25)))
        cooldown = max(0.0, float(getattr(self.cfg, "death_event_cooldown_sec", 20.0)))
        debug = bool(getattr(self.cfg, "screen_event_debug_log", False))
        save_debug_frames = bool(getattr(self.cfg, "screen_event_save_debug_frames", False))
        save_roi_crops = bool(getattr(self.cfg, "screen_event_save_roi_crops", True))
        debug_dir = str(getattr(self.cfg, "screen_event_debug_dir", "debug/screen_event") or "debug/screen_event")
        save_detected_frames = bool(getattr(self.cfg, "screen_event_save_detected_frames", False))
        detected_frame_dir = str(
            getattr(self.cfg, "screen_event_detected_frame_dir", "debug/screen_events") or "debug/screen_events"
        )
        save_detected_full = bool(getattr(self.cfg, "screen_event_save_detected_full_frame", True))
        save_detected_overlay = bool(getattr(self.cfg, "screen_event_save_detected_overlay", True))
        save_detected_roi = bool(getattr(self.cfg, "screen_event_save_detected_roi", True))
        save_detected_metrics = bool(getattr(self.cfg, "screen_event_save_detected_metrics", True))
        max_debug_files = int(getattr(self.cfg, "screen_event_max_debug_files", 200))
        try:
            if self.frame_source is None:
                self.frame_source = create_frame_source(self.cfg)
            if self.detector is None:
                self.detector = DeathDetector(self.cfg, self.templates)
            self.frame_source.start()
            while self.running:
                start = time.monotonic()
                try:
                    image = self.frame_source.read()
                    result = self.detector.detect(image)
                    now = time.monotonic()
                    elapsed = now - start

                    if debug:
                        print(f"[SCREEN] capture elapsed={elapsed:.3f}s interval={interval:.3f}s", flush=True)
                        print(f"[SCREEN] {format_detection_geometry(image, self.cfg)}", flush=True)
                    if elapsed > interval:
                        print(
                            f"[SCREEN] warning capture elapsed > interval elapsed={elapsed:.3f}s interval={interval:.3f}s",
                            flush=True,
                        )

                    if result.detected:
                        cooldown_remaining = max(0.0, cooldown - (now - self.last_emit_time))
                        cooldown_active = cooldown_remaining > 0.0
                        emitted = not cooldown_active
                        if save_detected_frames:
                            saved_dir = save_detection_debug_frames(
                                _frame_source_name(self.cfg),
                                image,
                                self.cfg,
                                result,
                                detected_frame_dir,
                                save_detected_roi,
                                event_label="raw_detected",
                                metrics_extra={
                                    "emitted": emitted,
                                    "screen_event_cooldown_active": cooldown_active,
                                    "screen_event_cooldown_remaining": cooldown_remaining,
                                    "reaction_source": None,
                                },
                                save_full=save_detected_full,
                                save_overlay=save_detected_overlay,
                                save_roi=save_detected_roi,
                                save_metrics=save_detected_metrics,
                            )
                            prune_debug_files(saved_dir, max_debug_files)
                        if save_debug_frames:
                            save_detection_debug_frames(
                                _frame_source_name(self.cfg), image, self.cfg, result, debug_dir, save_roi_crops
                            )
                        if cooldown_active:
                            print("[SCREEN] skip reason=cooldown", flush=True)
                        else:
                            ocr_result = self.detector.ocr(image)
                            print(f"[SCREEN] detected type=death {format_detection_metrics(result)}", flush=True)
                            if _ocr_enabled(self.cfg):
                                if ocr_result.reason == "ok":
                                    print(
                                        f"[SCREEN] OCR text={ocr_result.text} confidence={ocr_result.confidence:.1f}",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[SCREEN] OCR skip reason={ocr_result.reason} "
                                        f"text={ocr_result.text} confidence={ocr_result.confidence:.1f}",
                                        flush=True,
                                    )
                            self.output_queue.put(ScreenEvent(
                                event_type="death",
                                confidence=result.final_score,
                                created_at=now,
                                details={
                                    "final_score": result.final_score,
                                    "template_score": result.template_score,
                                    "shape_score": result.shape_score,
                                    "dark_ratio": result.dark_ratio,
                                    "white_ratio": result.white_ratio,
                                    "cols_with_white": result.cols_with_white,
                                    "rows_with_white": result.rows_with_white,
                                    "ocr_text": ocr_result.text,
                                    "ocr_confidence": ocr_result.confidence,
                                    "ocr_reason": ocr_result.reason,
                                    "ocr_preprocess_mode": ocr_result.preprocess_mode,
                                    "ocr_roi": ocr_result.roi,
                                    "ocr_scale": ocr_result.scale,
                                },
                            ))
                            self.last_emit_time = now
                    else:
                        should_log_skip = self._should_log_skip(now)
                        if should_log_skip:
                            print(f"[SCREEN] skip reason={result.reason} {format_detection_metrics(result)}", flush=True)
                        if save_debug_frames and should_log_skip:
                            save_detection_debug_frames(
                                _frame_source_name(self.cfg), image, self.cfg, result, debug_dir, save_roi_crops
                            )
                except Exception as e:
                    print(f"[SCREEN] error: {e}", flush=True)
                    elapsed = time.monotonic() - start
                time.sleep(max(0.0, interval - elapsed))
        except Exception as e:
            print(f"[SCREEN] disabled reason={e}", flush=True)
        finally:
            if self.frame_source is not None:
                self.frame_source.stop()
            self.running = False


if __name__ == "__main__":
    import argparse
    import importlib.util

    def _load_cli_config():
        config_path = _project_root() / "config" / "config.py"
        if not config_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("screen_event_cli_config", config_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _cfg_value(module, name, default):
        if module is None:
            return default
        return getattr(module, name, default)

    parser = argparse.ArgumentParser(description="Test screen death detection against an image file or frame source.")
    parser.add_argument("image", nargs="?")
    parser.add_argument("--template", default=None)
    parser.add_argument("--frame-source", default=None, choices=("mss", "obs_virtual_camera"))
    parser.add_argument("--capture-once", action="store_true", help="Capture one frame from the configured frame source.")
    parser.add_argument("--save-frame", default=None, help="Save captured frame to this path.")
    parser.add_argument("--detect", action="store_true", help="Run death detection for captured frame.")
    parser.add_argument("--save-debug-frames", action="store_true", help="Save full frame, ROI crops, overlay, and metrics.")
    parser.add_argument("--debug-dir", default=None, help="Directory for screen detection debug frames.")
    parser.add_argument("--save-roi-crops", action="store_true", help="Save death/text ROI crops with debug frames.")
    parser.add_argument("--print-detection-metrics", action="store_true", help="Print detection geometry and metrics.")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--ocr", action="store_true", help="Run OCR when death is detected.")
    parser.add_argument("--ocr-lang", default=None)
    parser.add_argument("--ocr-roi", default=None, help="Override OCR ROI as x1,y1,x2,y2.")
    parser.add_argument("--ocr-preprocess-mode", default=None)
    parser.add_argument("--tesseract-cmd", default="")
    parser.add_argument("--debug-ocr", action="store_true", help="Print OCR backend settings before running OCR.")
    parser.add_argument("--save-ocr-debug", action="store_true", help="Save OCR crop and processed images.")
    parser.add_argument("--ocr-debug-dir", default=None)
    args = parser.parse_args()
    cli_config = _load_cli_config()

    class _Cfg:
        screen_frame_source = args.frame_source or _cfg_value(cli_config, "SCREEN_FRAME_SOURCE", "mss")
        death_event_roi = _cfg_value(cli_config, "DEATH_EVENT_ROI", (0.32, 0.21, 0.67, 0.54))
        death_event_text_roi = _cfg_value(cli_config, "DEATH_EVENT_TEXT_ROI", (0.42, 0.33, 0.59, 0.46))
        death_event_roi_obs = _cfg_value(cli_config, "DEATH_EVENT_ROI_OBS", None)
        death_event_text_roi_obs = _cfg_value(cli_config, "DEATH_EVENT_TEXT_ROI_OBS", None)
        death_event_template_path = args.template or _cfg_value(
            cli_config, "DEATH_EVENT_TEMPLATE_PATH", "assets/templates/splatoon_death_yarareta.png"
        )
        death_event_template_path_obs = "" if args.template else _cfg_value(cli_config, "DEATH_EVENT_TEMPLATE_PATH_OBS", "")
        screen_capture_width = _cfg_value(cli_config, "SCREEN_CAPTURE_WIDTH", 640)
        screen_capture_height = _cfg_value(cli_config, "SCREEN_CAPTURE_HEIGHT", 360)
        screen_capture_monitor_index = _cfg_value(cli_config, "SCREEN_CAPTURE_MONITOR_INDEX", 1)
        screen_capture_region = _cfg_value(cli_config, "SCREEN_CAPTURE_REGION", None)
        obs_virtual_camera_index = args.camera_index if args.camera_index is not None else _cfg_value(cli_config, "OBS_VIRTUAL_CAMERA_INDEX", 0)
        obs_virtual_camera_width = args.camera_width if args.camera_width is not None else _cfg_value(cli_config, "OBS_VIRTUAL_CAMERA_WIDTH", 1920)
        obs_virtual_camera_height = args.camera_height if args.camera_height is not None else _cfg_value(cli_config, "OBS_VIRTUAL_CAMERA_HEIGHT", 1080)
        obs_virtual_camera_fps = args.camera_fps if args.camera_fps is not None else _cfg_value(cli_config, "OBS_VIRTUAL_CAMERA_FPS", 30)
        obs_virtual_camera_warmup_frames = _cfg_value(cli_config, "OBS_VIRTUAL_CAMERA_WARMUP_FRAMES", 5)
        death_event_min_template_score = _cfg_value(cli_config, "DEATH_EVENT_MIN_TEMPLATE_SCORE", 0.55)
        death_event_shape_min_template_score = _cfg_value(cli_config, "DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE", 0.40)
        death_event_min_template_score_obs = _cfg_value(cli_config, "DEATH_EVENT_MIN_TEMPLATE_SCORE_OBS", None)
        death_event_shape_min_template_score_obs = _cfg_value(cli_config, "DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE_OBS", None)
        death_event_white_ratio_min = _cfg_value(cli_config, "DEATH_EVENT_WHITE_RATIO_MIN", 0.015)
        death_event_white_ratio_max = _cfg_value(cli_config, "DEATH_EVENT_WHITE_RATIO_MAX", 0.18)
        death_event_shape_white_ratio_min = _cfg_value(cli_config, "DEATH_EVENT_SHAPE_WHITE_RATIO_MIN", 0.04)
        death_event_shape_white_ratio_max = _cfg_value(cli_config, "DEATH_EVENT_SHAPE_WHITE_RATIO_MAX", 0.14)
        screen_event_mode = "death_ocr" if args.ocr else "death_detect_only"
        death_event_ocr_roi = parse_roi(args.ocr_roi) if args.ocr_roi else _cfg_value(cli_config, "DEATH_EVENT_OCR_ROI", (0.34, 0.25, 0.66, 0.48))
        death_event_ocr_lang = args.ocr_lang or _cfg_value(cli_config, "DEATH_EVENT_OCR_LANG", "jpn+eng")
        death_event_ocr_config = _cfg_value(cli_config, "DEATH_EVENT_OCR_CONFIG", "--psm 6")
        death_event_ocr_min_confidence = _cfg_value(cli_config, "DEATH_EVENT_OCR_MIN_CONFIDENCE", 0.0)
        death_event_ocr_scale = _cfg_value(cli_config, "DEATH_EVENT_OCR_SCALE", 3.0)
        death_event_ocr_preprocess_mode = args.ocr_preprocess_mode or _cfg_value(cli_config, "DEATH_EVENT_OCR_PREPROCESS_MODE", "default")
        death_event_ocr_tesseract_cmd = args.tesseract_cmd or _cfg_value(cli_config, "DEATH_EVENT_OCR_TESSERACT_CMD", "")
        death_event_ocr_debug_log = args.debug_ocr or _cfg_value(cli_config, "DEATH_EVENT_OCR_DEBUG_LOG", False)
        death_event_ocr_save_debug_images = args.save_ocr_debug
        death_event_ocr_debug_dir = args.ocr_debug_dir or _cfg_value(cli_config, "DEATH_EVENT_OCR_DEBUG_DIR", "debug/ocr")
        death_event_ocr_debug_source_name = args.image or str(screen_frame_source)
        screen_event_debug_dir = args.debug_dir or _cfg_value(cli_config, "SCREEN_EVENT_DEBUG_DIR", "debug/screen_event")

    if cv2 is None:
        raise SystemExit(f"cv2 is unavailable: {CV2_IMPORT_ERROR}")

    if args.capture_once:
        frame_source = create_frame_source(_Cfg)
        try:
            frame_source.start()
            rgb = frame_source.read()
        finally:
            frame_source.stop()

        if args.save_frame:
            save_path = Path(args.save_frame)
            if not save_path.is_absolute():
                save_path = _project_root() / save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"[SCREEN] frame saved path={save_path}", flush=True)

        if not args.detect and not args.ocr:
            raise SystemExit(0)
    else:
        if not args.image:
            parser.error("image is required unless --capture-once is used.")
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"image unreadable: {args.image}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.print_detection_metrics:
        print(f"[SCREEN] {format_detection_geometry(rgb, _Cfg)}", flush=True)

    tpl = load_templates(effective_template_path(_Cfg))
    result = detect_death_event(rgb, _Cfg, tpl)
    print(result)
    if args.print_detection_metrics:
        print(f"[SCREEN] detected={result.detected} reason={result.reason} {format_detection_metrics(result)}", flush=True)
    if args.save_debug_frames:
        saved_dir = save_detection_debug_frames(
            args.image or str(_Cfg.screen_frame_source),
            rgb,
            _Cfg,
            result,
            _Cfg.screen_event_debug_dir,
            args.save_roi_crops or args.save_debug_frames,
        )
        print(f"[SCREEN] debug frames saved dir={saved_dir}", flush=True)
    if args.ocr and result.detected:
        print(ocr_death_text(rgb, _Cfg))
