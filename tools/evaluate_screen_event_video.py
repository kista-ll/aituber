import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from screen_event_detector import cv2, detect_death_event, load_templates  # noqa: E402


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


def result_to_row(video_path: Path, interval: float, timestamp: float, frame_index: int, result, emitted: bool):
    return {
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
    }


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
                result = detect_death_event(frame_to_rgb(frame), cfg, templates)
                result_cache[frame_index] = result

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

                row = result_to_row(video_path, interval, timestamp, frame_index, result, emitted)
                rows.append(row)
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
    templates = load_templates(args.template)
    if not templates:
        raise SystemExit("no templates loaded.")

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
