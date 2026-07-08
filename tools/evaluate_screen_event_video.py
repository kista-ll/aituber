import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from screen_event_detector import (  # noqa: E402
    cv2,
    detect_death_event,
    load_templates,
    load_weapon_templates,
    match_death_weapon_template,
    ocr_death_text,
    parse_roi,
    save_weapon_match_debug_images,
)
from screen_event_reactions import select_death_reaction  # noqa: E402


class EvalConfig:
    death_event_roi = (0.32, 0.21, 0.67, 0.54)
    death_event_text_roi = (0.42, 0.33, 0.59, 0.46)
    death_event_template_path = "assets/templates/splatoon_death_yarareta.png"
    screen_capture_width = 640
    screen_capture_height = 360
    death_event_min_template_score = 0.55
    death_event_shape_min_template_score = 0.40
    death_event_white_ratio_min = 0.015
    death_event_white_ratio_max = 0.18
    death_event_shape_white_ratio_min = 0.04
    death_event_shape_white_ratio_max = 0.14
    screen_event_mode = "death_detect_only"
    death_event_ocr_roi = (0.34, 0.25, 0.66, 0.48)
    death_event_ocr_lang = "jpn+eng"
    death_event_ocr_config = "--psm 6"
    death_event_ocr_min_confidence = 0.0
    death_event_ocr_scale = 3.0
    death_event_ocr_preprocess_mode = "default"
    death_event_ocr_tesseract_cmd = ""
    death_event_use_context_reactions = True
    death_event_use_weapon_category_reactions = False
    death_event_ocr_category_min_confidence = 60.0
    death_event_weapon_keywords = None
    death_event_reactions_by_category = None
    death_event_weapon_match_enabled = False
    death_event_weapon_template_dir = "assets/templates/weapons"
    death_event_weapon_template_metadata_path = "assets/templates/weapons/weapons.json"
    death_event_weapon_name_roi = None
    death_event_weapon_name_roi_obs = None
    death_event_weapon_preprocess_mode = "sharpen_threshold"
    death_event_weapon_min_score = 0.80
    death_event_weapon_min_score_obs = None
    death_event_weapon_min_margin = 0.08


FIELDNAMES = [
    "video_path",
    "interval_sec",
    "timestamp_sec",
    "frame_index",
    "detected",
    "emitted",
    "final_score",
    "template_score",
    "shape_score",
    "dark_ratio",
    "white_ratio",
    "cols_with_white",
    "rows_with_white",
    "reason",
    "ocr_preprocess_mode",
    "ocr_roi",
    "ocr_scale",
    "ocr_text",
    "ocr_confidence",
    "ocr_reason",
    "normalized_ocr_text",
    "weapon_category",
    "emotion_category",
    "selected_reaction",
    "reaction_source",
    "matched_keywords",
    "category_reason",
    "weapon_match_enabled",
    "weapon_id",
    "weapon_display_name",
    "weapon_match_category",
    "weapon_template_category",
    "weapon_best_id",
    "weapon_best_display_name",
    "weapon_best_category",
    "weapon_best_score",
    "weapon_second_best_id",
    "weapon_second_best_score",
    "weapon_match_margin",
    "weapon_match_accepted",
    "weapon_match_reason",
    "weapon_template_path",
]


def parse_intervals(value: str):
    intervals = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        interval = float(part)
        if interval <= 0:
            raise argparse.ArgumentTypeError("interval values must be positive.")
        intervals.append(interval)
    if not intervals:
        raise argparse.ArgumentTypeError("at least one interval is required.")
    return intervals


def parse_times(value: str):
    times = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        timestamp = float(part)
        if timestamp < 0:
            raise argparse.ArgumentTypeError("expected timestamps must be zero or positive.")
        times.append(timestamp)
    return times


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def video_info(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError("video FPS is unavailable.")
    duration = frame_count / fps if frame_count > 0 else 0.0
    return fps, frame_count, duration


def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def reaction_from_ocr(ocr_result, cfg):
    details = {}
    if ocr_result is not None:
        details = {
            "ocr_text": getattr(ocr_result, "text", ""),
            "ocr_confidence": getattr(ocr_result, "confidence", 0.0),
            "ocr_reason": getattr(ocr_result, "reason", ""),
        }
    return select_death_reaction(details, cfg)


def weapon_result_columns(weapon_result):
    return {
        "weapon_match_enabled": str(bool(weapon_result)).lower(),
        "weapon_id": getattr(weapon_result, "weapon_id", "") or "",
        "weapon_display_name": getattr(weapon_result, "display_name", "") or "",
        "weapon_match_category": getattr(weapon_result, "category", "") or "",
        "weapon_template_category": getattr(weapon_result, "category", "") or "",
        "weapon_best_id": getattr(weapon_result, "best_weapon_id", "") or "",
        "weapon_best_display_name": getattr(weapon_result, "best_display_name", "") or "",
        "weapon_best_category": getattr(weapon_result, "best_category", "") or "",
        "weapon_best_score": f"{getattr(weapon_result, 'best_score', 0.0):.6f}" if weapon_result is not None else "",
        "weapon_second_best_id": getattr(weapon_result, "second_best_weapon_id", "") or "",
        "weapon_second_best_score": (
            f"{weapon_result.second_best_score:.6f}"
            if weapon_result is not None and weapon_result.second_best_score is not None else ""
        ),
        "weapon_match_margin": (
            f"{weapon_result.margin:.6f}" if weapon_result is not None and weapon_result.margin is not None else ""
        ),
        "weapon_match_accepted": str(getattr(weapon_result, "accepted", False)).lower() if weapon_result is not None else "",
        "weapon_match_reason": getattr(weapon_result, "reason", "") if weapon_result is not None else "",
        "weapon_template_path": getattr(weapon_result, "template_path", "") if weapon_result is not None else "",
    }


def result_to_row(video_path: Path, interval: float, timestamp: float, frame_index: int, result, emitted: bool, cfg, ocr_result=None, weapon_result=None):
    reaction = reaction_from_ocr(ocr_result, cfg) if result.detected else {}
    row = {
        "video_path": str(video_path),
        "interval_sec": f"{interval:.3f}",
        "timestamp_sec": f"{timestamp:.3f}",
        "frame_index": str(frame_index),
        "detected": str(result.detected).lower(),
        "emitted": str(emitted).lower(),
        "final_score": f"{result.final_score:.6f}",
        "template_score": f"{result.template_score:.6f}",
        "shape_score": f"{result.shape_score:.6f}",
        "dark_ratio": f"{result.dark_ratio:.6f}",
        "white_ratio": f"{result.white_ratio:.6f}",
        "cols_with_white": f"{result.cols_with_white:.6f}",
        "rows_with_white": f"{result.rows_with_white:.6f}",
        "reason": result.reason,
        "ocr_preprocess_mode": getattr(ocr_result, "preprocess_mode", ""),
        "ocr_roi": repr(getattr(ocr_result, "roi", "")) if ocr_result is not None else "",
        "ocr_scale": f"{getattr(ocr_result, 'scale', 0.0):.6f}" if ocr_result is not None else "",
        "ocr_text": getattr(ocr_result, "text", ""),
        "ocr_confidence": f"{getattr(ocr_result, 'confidence', 0.0):.6f}" if ocr_result is not None else "",
        "ocr_reason": getattr(ocr_result, "reason", ""),
        "normalized_ocr_text": reaction.get("normalized_ocr_text", ""),
        "weapon_category": reaction.get("weapon_category", ""),
        "emotion_category": reaction.get("emotion_category", ""),
        "selected_reaction": reaction.get("phrase", ""),
        "reaction_source": reaction.get("reaction_source", ""),
        "matched_keywords": ",".join(reaction.get("matched_keywords", ())),
        "category_reason": reaction.get("category_reason", ""),
    }
    row.update(weapon_result_columns(weapon_result))
    return row


def save_frame(frame, output_dir: Path, prefix: str, interval: float, timestamp: float, frame_index: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_interval = str(interval).replace(".", "p")
    filename = f"{prefix}_interval-{safe_interval}_t-{timestamp:08.3f}_f-{frame_index}.png"
    cv2.imwrite(str(output_dir / filename), frame)


def update_summary(summary, row):
    summary["sampled"] += 1
    if row["detected"] == "true":
        summary["raw_detected"] += 1
        summary["detections"].append(row)
    if row["emitted"] == "true":
        summary["emitted"] += 1


def parse_modes(value: str):
    modes = []
    for part in value.split(","):
        mode = part.strip()
        if mode:
            modes.append(mode)
    return modes or ["default"]


def parse_rois(value: str):
    rois = []
    for part in value.split(";"):
        roi = part.strip()
        if roi:
            rois.append(parse_roi(roi))
    return rois


def evaluate_video(
    video_path: Path,
    intervals,
    cfg,
    templates,
    cooldown_sec: float,
    save_dir: Path | None,
    save_high_score_dir: Path | None,
    high_score_threshold: float,
    max_seconds: float | None = None,
    run_ocr: bool = False,
    ocr_modes=None,
    ocr_rois=None,
    weapon_template_match: bool = False,
    weapon_templates=None,
    save_weapon_debug: bool = False,
    weapon_debug_dir: Path | None = None,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"video unreadable: {video_path}")

    fps, frame_count, duration = video_info(capture)
    if max_seconds is not None and max_seconds > 0:
        duration = min(duration, max_seconds)
        frame_count = min(frame_count, int(round(max_seconds * fps)))

    interval_state = {}
    for interval in intervals:
        interval_state[interval] = {
            "step_frames": max(1, int(round(fps * interval))),
            "next_frame": 0,
            "last_emit_timestamp": -1e9,
            "summary": {
                "sampled": 0,
                "raw_detected": 0,
                "emitted": 0,
                "detections": [],
            },
        }

    rows = []
    result_cache = {}

    frame_index = 0
    while frame_index < frame_count:
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        states_to_sample = [
            (interval, state)
            for interval, state in interval_state.items()
            if frame_index >= state["next_frame"]
        ]
        if states_to_sample:
            timestamp = frame_index / fps
            result = result_cache.get(frame_index)
            if result is None:
                rgb_frame = frame_to_rgb(frame)
                result = detect_death_event(rgb_frame, cfg, templates)
                result_cache[frame_index] = result
            else:
                rgb_frame = None

            ocr_results = [None]
            weapon_result = None
            if weapon_template_match and result.detected:
                if rgb_frame is None:
                    rgb_frame = frame_to_rgb(frame)
                weapon_result = match_death_weapon_template(rgb_frame, cfg, weapon_templates or [])
                if save_weapon_debug:
                    cfg.death_event_weapon_debug_source_name = f"{video_path.stem}_t-{timestamp:.3f}_f-{frame_index}"
                    save_weapon_match_debug_images(
                        f"{video_path.stem}_t-{timestamp:.3f}_f-{frame_index}",
                        rgb_frame,
                        cfg,
                        weapon_templates or [],
                        weapon_result,
                        str(weapon_debug_dir) if weapon_debug_dir is not None else "debug/weapon_match",
                    )
            if run_ocr and result.detected:
                if rgb_frame is None:
                    rgb_frame = frame_to_rgb(frame)
                ocr_results = []
                cfg.death_event_ocr_debug_source_name = f"{video_path.stem}_t-{timestamp:.3f}_f-{frame_index}"
                for roi in (ocr_rois or [cfg.death_event_ocr_roi]):
                    cfg.death_event_ocr_roi = roi
                    for mode in (ocr_modes or ["default"]):
                        cfg.death_event_ocr_preprocess_mode = mode
                        ocr_results.append(ocr_death_text(rgb_frame, cfg))

            for interval, state in states_to_sample:
                emitted = False
                if result.detected:
                    if timestamp - state["last_emit_timestamp"] >= cooldown_sec:
                        emitted = True
                        state["last_emit_timestamp"] = timestamp
                    if save_dir is not None:
                        save_frame(frame, save_dir, "detected", interval, timestamp, frame_index)
                elif save_high_score_dir is not None and result.final_score >= high_score_threshold:
                    save_frame(frame, save_high_score_dir, result.reason or "high_score", interval, timestamp, frame_index)

                for ocr_result in ocr_results:
                    row = result_to_row(video_path, interval, timestamp, frame_index, result, emitted, cfg, ocr_result, weapon_result)
                    rows.append(row)
                    if ocr_result is ocr_results[0]:
                        update_summary(state["summary"], row)
                state["next_frame"] += state["step_frames"]

        frame_index += 1

    capture.release()

    rows.sort(key=lambda row: (float(row["interval_sec"]), float(row["timestamp_sec"])))
    for interval in intervals:
        summary = interval_state[interval]["summary"]
        print(
            f"interval={interval:.3f}s sampled={summary['sampled']} duration={duration:.2f}s "
            f"raw_detected={summary['raw_detected']} emitted={summary['emitted']}",
            flush=True,
        )
        for row in summary["detections"]:
            print(
                f"  detected t={row['timestamp_sec']}s frame={row['frame_index']} "
                f"emitted={row['emitted']} final={row['final_score']} "
                f"template={row['template_score']} reason={row['reason']}",
                flush=True,
            )
    return rows


def print_expected_summary(rows, intervals, expected_times, tolerance_sec):
    if not expected_times:
        return

    print(f"expected_times={','.join(f'{t:.3f}' for t in expected_times)} tolerance={tolerance_sec:.3f}s", flush=True)
    for interval in intervals:
        interval_key = f"{interval:.3f}"
        detected_rows = [
            row for row in rows
            if row["interval_sec"] == interval_key and row["detected"] == "true"
        ]
        emitted_rows = [
            row for row in rows
            if row["interval_sec"] == interval_key and row["emitted"] == "true"
        ]

        hits = 0
        print(f"interval={interval_key}s expected_check:", flush=True)
        for expected in expected_times:
            nearest = None
            nearest_delta = None
            for row in detected_rows:
                timestamp = float(row["timestamp_sec"])
                delta = abs(timestamp - expected)
                if nearest_delta is None or delta < nearest_delta:
                    nearest = row
                    nearest_delta = delta

            nearest_emitted = None
            nearest_emitted_delta = None
            for row in emitted_rows:
                timestamp = float(row["timestamp_sec"])
                delta = abs(timestamp - expected)
                if nearest_emitted_delta is None or delta < nearest_emitted_delta:
                    nearest_emitted = row
                    nearest_emitted_delta = delta

            if nearest is not None and nearest_delta is not None and nearest_delta <= tolerance_sec:
                hits += 1
                emitted_text = "emitted=none"
                if nearest_emitted is not None and nearest_emitted_delta is not None and nearest_emitted_delta <= tolerance_sec:
                    emitted_text = f"emitted={nearest_emitted['timestamp_sec']}s"
                print(
                    f"  hit expected={expected:.3f}s detected={nearest['timestamp_sec']}s "
                    f"delta={nearest_delta:.3f}s {emitted_text} "
                    f"final={nearest['final_score']} template={nearest['template_score']}",
                    flush=True,
                )
            elif nearest is not None:
                print(
                    f"  miss expected={expected:.3f}s nearest={nearest['timestamp_sec']}s "
                    f"delta={nearest_delta:.3f}s final={nearest['final_score']} "
                    f"template={nearest['template_score']}",
                    flush=True,
                )
            else:
                print(f"  miss expected={expected:.3f}s nearest=none", flush=True)

        extra_emitted = []
        for row in emitted_rows:
            timestamp = float(row["timestamp_sec"])
            if all(abs(timestamp - expected) > tolerance_sec for expected in expected_times):
                extra_emitted.append(row)
        print(f"  summary hits={hits}/{len(expected_times)} extra_emitted={len(extra_emitted)}", flush=True)
        for row in extra_emitted:
            print(
                f"    extra emitted t={row['timestamp_sec']}s final={row['final_score']} "
                f"template={row['template_score']} reason={row['reason']}",
                flush=True,
            )


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate screen event detection against a video file.")
    parser.add_argument("video", help="Video file path.")
    parser.add_argument("--interval-sec", type=float, default=None, help="Single sampling interval in seconds.")
    parser.add_argument(
        "--compare",
        type=parse_intervals,
        default=None,
        help="Comma-separated intervals to compare, for example: 1.5,0.5,0.25",
    )
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path.")
    parser.add_argument("--template", default=EvalConfig.death_event_template_path, help="Template path.")
    parser.add_argument("--ocr", action="store_true", help="Run OCR for detected frames.")
    parser.add_argument("--ocr-lang", default=EvalConfig.death_event_ocr_lang, help="Tesseract OCR language.")
    parser.add_argument("--ocr-roi", default=None, help="Override OCR ROI as x1,y1,x2,y2.")
    parser.add_argument("--ocr-preprocess-mode", default=EvalConfig.death_event_ocr_preprocess_mode, help="Single OCR preprocessing mode.")
    parser.add_argument("--compare-ocr-modes", default=None, help="Comma-separated OCR preprocessing modes.")
    parser.add_argument(
        "--compare-ocr-rois",
        default=None,
        help="Semicolon-separated OCR ROIs, for example: 0.34,0.25,0.66,0.48;0.42,0.34,0.59,0.47",
    )
    parser.add_argument("--save-ocr-debug", action="store_true", help="Save OCR crop and processed images for detected frames.")
    parser.add_argument("--ocr-debug-dir", default="debug/ocr", help="Directory for OCR debug images.")
    parser.add_argument("--use-weapon-category-reactions", action="store_true", help="Use experimental OCR weapon category reactions.")
    parser.add_argument("--disable-context-reactions", action="store_true", help="Disable context fixed reactions.")
    parser.add_argument("--tesseract-cmd", default="", help="Optional path to tesseract executable.")
    parser.add_argument("--weapon-template-match", action="store_true", help="Run experimental weapon-name template matching.")
    parser.add_argument("--weapon-template-dir", default=EvalConfig.death_event_weapon_template_dir)
    parser.add_argument("--weapon-template-metadata", default=EvalConfig.death_event_weapon_template_metadata_path)
    parser.add_argument("--weapon-roi", default=None, help="Override weapon-name ROI as x1,y1,x2,y2.")
    parser.add_argument("--weapon-preprocess-mode", default=EvalConfig.death_event_weapon_preprocess_mode)
    parser.add_argument("--save-weapon-debug", action="store_true", help="Save weapon template match debug images.")
    parser.add_argument("--weapon-debug-dir", default="debug/weapon_match")
    parser.add_argument("--cooldown-sec", type=float, default=20.0, help="Cooldown used to simulate emitted events.")
    parser.add_argument("--save-detected-dir", default=None, help="Directory to save detected frames.")
    parser.add_argument(
        "--save-high-score-dir",
        default=None,
        help="Directory to save high-score non-detected frames for false-positive review.",
    )
    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.30,
        help="Minimum final_score for saving non-detected review frames.",
    )
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional limit for quick checks.")
    parser.add_argument(
        "--expected-times",
        type=parse_times,
        default=None,
        help="Comma-separated rough expected death timestamps, for example: 133,183,333,373",
    )
    parser.add_argument(
        "--expected-tolerance-sec",
        type=float,
        default=3.0,
        help="Tolerance window for matching expected timestamps.",
    )
    args = parser.parse_args()

    if cv2 is None:
        raise SystemExit("opencv-python is not available.")

    video_path = resolve_project_path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    cfg = EvalConfig()
    cfg.death_event_template_path = args.template
    cfg.screen_event_mode = "death_ocr" if args.ocr else "death_detect_only"
    cfg.death_event_ocr_lang = args.ocr_lang
    if args.ocr_roi:
        cfg.death_event_ocr_roi = parse_roi(args.ocr_roi)
    cfg.death_event_ocr_save_debug_images = args.save_ocr_debug
    cfg.death_event_ocr_debug_dir = args.ocr_debug_dir
    cfg.death_event_use_weapon_category_reactions = args.use_weapon_category_reactions
    cfg.death_event_use_context_reactions = not args.disable_context_reactions
    cfg.death_event_ocr_tesseract_cmd = args.tesseract_cmd
    cfg.death_event_weapon_match_enabled = args.weapon_template_match
    cfg.death_event_weapon_template_dir = args.weapon_template_dir
    cfg.death_event_weapon_template_metadata_path = args.weapon_template_metadata
    cfg.death_event_weapon_preprocess_mode = args.weapon_preprocess_mode
    if args.weapon_roi:
        cfg.death_event_weapon_name_roi = parse_roi(args.weapon_roi)
    ocr_modes = parse_modes(args.compare_ocr_modes) if args.compare_ocr_modes else [args.ocr_preprocess_mode]
    ocr_rois = parse_rois(args.compare_ocr_rois) if args.compare_ocr_rois else [cfg.death_event_ocr_roi]
    templates = load_templates(args.template)
    if not templates:
        raise SystemExit("no templates loaded.")
    weapon_templates = load_weapon_templates(cfg) if args.weapon_template_match else []

    if args.compare is not None:
        intervals = args.compare
    elif args.interval_sec is not None:
        intervals = [args.interval_sec]
    else:
        intervals = [1.5, 0.5, 0.25]

    save_detected_dir = resolve_project_path(args.save_detected_dir) if args.save_detected_dir else None
    save_high_score_dir = resolve_project_path(args.save_high_score_dir) if args.save_high_score_dir else None

    all_rows = evaluate_video(
        video_path=video_path,
        intervals=intervals,
        cfg=cfg,
        templates=templates,
        cooldown_sec=max(0.0, args.cooldown_sec),
        save_dir=save_detected_dir,
        save_high_score_dir=save_high_score_dir,
        high_score_threshold=args.high_score_threshold,
        max_seconds=args.max_seconds,
        run_ocr=args.ocr,
        ocr_modes=ocr_modes,
        ocr_rois=ocr_rois,
        weapon_template_match=args.weapon_template_match,
        weapon_templates=weapon_templates,
        save_weapon_debug=args.save_weapon_debug,
        weapon_debug_dir=resolve_project_path(args.weapon_debug_dir),
    )
    print_expected_summary(
        all_rows,
        intervals,
        args.expected_times or [],
        max(0.0, args.expected_tolerance_sec),
    )

    if args.csv_path:
        write_csv(resolve_project_path(args.csv_path), all_rows)


if __name__ == "__main__":
    main()
