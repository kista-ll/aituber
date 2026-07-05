import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from screen_event_detector import cv2, detect_death_event, load_templates  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


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


def expected_from_path(path: Path, samples_root: Path):
    rel = path.relative_to(samples_root)
    parts = rel.parts
    if not parts:
        return ""
    if parts[0] == "positive":
        return "true"
    if parts[0] == "negative":
        return "false"
    return ""


def group_from_path(path: Path, samples_root: Path) -> str:
    rel = path.relative_to(samples_root)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in {"positive", "negative"}:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return ""


def iter_images(samples_root: Path):
    for path in sorted(samples_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def evaluate_image(path: Path, cfg, templates):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"image unreadable: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return detect_death_event(rgb, cfg, templates)


def as_row(path: Path, samples_root: Path, result):
    return {
        "filename": str(path.relative_to(samples_root)),
        "group": group_from_path(path, samples_root),
        "expected": expected_from_path(path, samples_root),
        "detected": str(result.detected).lower(),
        "final_score": f"{result.final_score:.6f}",
        "template_score": f"{result.template_score:.6f}",
        "shape_score": f"{result.shape_score:.6f}",
        "dark_ratio": f"{result.dark_ratio:.6f}",
        "white_ratio": f"{result.white_ratio:.6f}",
        "cols_with_white": f"{result.cols_with_white:.6f}",
        "rows_with_white": f"{result.rows_with_white:.6f}",
        "reason": result.reason,
    }


def print_table(rows):
    if not rows:
        print("No sample images found.")
        return
    headers = ["filename", "expected", "detected", "final_score", "template_score", "dark_ratio", "white_ratio", "cols_with_white", "rows_with_white", "reason"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def main():
    parser = argparse.ArgumentParser(description="Evaluate screen event detection samples.")
    parser.add_argument("--samples", default="assets/screen_samples", help="Sample root directory.")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path.")
    parser.add_argument("--template", default=EvalConfig.death_event_template_path, help="Template path.")
    args = parser.parse_args()

    if cv2 is None:
        raise SystemExit("opencv-python is not available.")

    samples_root = Path(args.samples)
    if not samples_root.is_absolute():
        samples_root = PROJECT_ROOT / samples_root
    if not samples_root.exists():
        raise SystemExit(f"sample root not found: {samples_root}")

    cfg = EvalConfig()
    cfg.death_event_template_path = args.template
    templates = load_templates(args.template)
    if not templates:
        raise SystemExit("no templates loaded.")

    rows = []
    for image_path in iter_images(samples_root):
        result = evaluate_image(image_path, cfg, templates)
        rows.append(as_row(image_path, samples_root, result))

    print_table(rows)

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "filename", "group", "expected", "detected", "final_score", "template_score",
                "shape_score", "dark_ratio", "white_ratio", "cols_with_white", "rows_with_white", "reason",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
