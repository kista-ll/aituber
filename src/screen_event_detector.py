import queue
import threading
import time
from dataclasses import dataclass, field
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
class DeathDetectionThresholds:
    template_min_score: float = 0.55
    shape_min_template_score: float = 0.40
    template_min_dark_ratio: float = 0.06
    shape_min_dark_ratio: float = 0.12
    white_ratio_min: float = 0.015
    white_ratio_max: float = 0.18
    shape_white_ratio_min: float = 0.04
    shape_white_ratio_max: float = 0.16
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


def _range_score(value: float, low: float, high: float, peak: float) -> float:
    if value < low or value > high:
        return 0.0
    if value <= peak:
        return _clamp01((value - low) / max(1e-6, peak - low))
    return _clamp01((high - value) / max(1e-6, high - peak))


def get_thresholds(cfg) -> DeathDetectionThresholds:
    return DeathDetectionThresholds(
        template_min_score=float(getattr(cfg, "death_event_min_template_score", 0.55)),
        shape_min_template_score=float(getattr(cfg, "death_event_shape_min_template_score", 0.40)),
        template_min_dark_ratio=float(getattr(cfg, "death_event_template_min_dark_ratio", 0.06)),
        shape_min_dark_ratio=float(getattr(cfg, "death_event_shape_min_dark_ratio", 0.12)),
        white_ratio_min=float(getattr(cfg, "death_event_white_ratio_min", 0.015)),
        white_ratio_max=float(getattr(cfg, "death_event_white_ratio_max", 0.18)),
        shape_white_ratio_min=float(getattr(cfg, "death_event_shape_white_ratio_min", 0.04)),
        shape_white_ratio_max=float(getattr(cfg, "death_event_shape_white_ratio_max", 0.16)),
        min_cols_with_white=float(getattr(cfg, "death_event_min_cols_with_white", 0.18)),
        min_rows_with_white=float(getattr(cfg, "death_event_min_rows_with_white", 0.08)),
        shape_min_cols_with_white=float(getattr(cfg, "death_event_shape_min_cols_with_white", 0.20)),
        shape_min_rows_with_white=float(getattr(cfg, "death_event_shape_min_rows_with_white", 0.10)),
    )


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

    target_width = int(getattr(cfg, "screen_capture_width", 0))
    target_height = int(getattr(cfg, "screen_capture_height", 0))
    if (
        target_width > 0
        and target_height > 0
        and (image.shape[1] != target_width or image.shape[0] != target_height)
    ):
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    death_roi = getattr(cfg, "death_event_roi", (0.32, 0.21, 0.67, 0.54))
    text_roi = getattr(cfg, "death_event_text_roi", (0.42, 0.33, 0.59, 0.46))
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


def format_detection_metrics(result: DeathDetectionResult) -> str:
    return (
        f"final_score={result.final_score:.2f} template_score={result.template_score:.2f} "
        f"shape_score={result.shape_score:.2f} dark={result.dark_ratio:.3f} "
        f"white={result.white_ratio:.3f} cols_with_white={result.cols_with_white:.3f} "
        f"rows_with_white={result.rows_with_white:.3f}"
    )


class ScreenEventDetector:
    def __init__(self, cfg, output_queue: queue.Queue):
        self.cfg = cfg
        self.output_queue = output_queue
        self.running = False
        self.thread = None
        self.templates = []
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
        if mss is None:
            print(f"[SCREEN] disabled reason=mss_unavailable error={MSS_IMPORT_ERROR}", flush=True)
            return
        self.templates = load_templates(getattr(self.cfg, "death_event_template_path", ""))
        if not self.templates:
            print("[SCREEN] disabled reason=template_unavailable", flush=True)
            return
        print(f"[SCREEN] templates loaded count={len(self.templates)}", flush=True)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[SCREEN] capture started", flush=True)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

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
        monitor_index = int(getattr(self.cfg, "screen_capture_monitor_index", 1))
        debug = bool(getattr(self.cfg, "screen_event_debug_log", False))
        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if monitor_index < 1 or monitor_index >= len(monitors):
                    print(f"[SCREEN] disabled reason=monitor_not_found index={monitor_index}", flush=True)
                    return
                monitor = monitors[monitor_index]
                target = _build_capture_target(monitor, self.cfg)
                print(f"[SCREEN] monitor index={monitor_index}", flush=True)
                print(
                    f"[SCREEN] capture region left={target['left']} top={target['top']} "
                    f"width={target['width']} height={target['height']}",
                    flush=True,
                )
                while self.running:
                    start = time.monotonic()
                    try:
                        image = capture_crop(sct, monitor, self.cfg)
                        result = detect_death_event(image, self.cfg, self.templates)
                        now = time.monotonic()
                        elapsed = now - start

                        if debug:
                            print(f"[SCREEN] capture elapsed={elapsed:.3f}s interval={interval:.3f}s", flush=True)
                        if elapsed > interval:
                            print(
                                f"[SCREEN] warning capture elapsed > interval elapsed={elapsed:.3f}s interval={interval:.3f}s",
                                flush=True,
                            )

                        if result.detected:
                            if now - self.last_emit_time < cooldown:
                                print("[SCREEN] skip reason=cooldown", flush=True)
                            else:
                                print(f"[SCREEN] detected type=death {format_detection_metrics(result)}", flush=True)
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
                                    },
                                ))
                                self.last_emit_time = now
                        elif self._should_log_skip(now):
                            print(f"[SCREEN] skip reason={result.reason} {format_detection_metrics(result)}", flush=True)
                    except Exception as e:
                        print(f"[SCREEN] error: {e}", flush=True)
                        elapsed = time.monotonic() - start
                    time.sleep(max(0.0, interval - elapsed))
        finally:
            self.running = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test screen death detection against an image file.")
    parser.add_argument("image")
    parser.add_argument("--template", default="assets/templates/splatoon_death_yarareta.png")
    args = parser.parse_args()

    class _Cfg:
        death_event_roi = (0.32, 0.21, 0.67, 0.54)
        death_event_text_roi = (0.42, 0.33, 0.59, 0.46)
        death_event_template_path = args.template
        screen_capture_width = 640
        screen_capture_height = 360

    if cv2 is None:
        raise SystemExit(f"cv2 is unavailable: {CV2_IMPORT_ERROR}")
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"image unreadable: {args.image}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tpl = load_templates(args.template)
    result = detect_death_event(rgb, _Cfg, tpl)
    print(result)
