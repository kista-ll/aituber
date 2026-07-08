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


def as_row(path: Path, samples_root: Path, result, cfg, ocr_result=None, weapon_result=None):
    reaction = reaction_from_ocr(ocr_result, cfg) if result.detected else {}
    row = {
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate screen event detection samples.")
    parser.add_argument("--samples", default="assets/screen_samples", help="Sample root directory.")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path.")
    parser.add_argument("--template", default=EvalConfig.death_event_template_path, help="Template path.")
    parser.add_argument("--ocr", action="store_true", help="Run OCR for detected images.")
    parser.add_argument("--ocr-lang", default=EvalConfig.death_event_ocr_lang, help="Tesseract OCR language.")
    parser.add_argument("--ocr-roi", default=None, help="Override OCR ROI as x1,y1,x2,y2.")
    parser.add_argument("--ocr-preprocess-mode", default=EvalConfig.death_event_ocr_preprocess_mode, help="Single OCR preprocessing mode.")
    parser.add_argument("--compare-ocr-modes", default=None, help="Comma-separated OCR preprocessing modes.")
    parser.add_argument(
        "--compare-ocr-rois",
        default=None,
        help="Semicolon-separated OCR ROIs, for example: 0.34,0.25,0.66,0.48;0.42,0.34,0.59,0.47",
    )
    parser.add_argument("--save-ocr-debug", action="store_true", help="Save OCR crop and processed images.")
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
    templates = load_templates(args.template)
    if not templates:
        raise SystemExit("no templates loaded.")
    weapon_templates = load_weapon_templates(cfg) if args.weapon_template_match else []

    modes = parse_modes(args.compare_ocr_modes) if args.compare_ocr_modes else [args.ocr_preprocess_mode]
    rois = parse_rois(args.compare_ocr_rois) if args.compare_ocr_rois else [cfg.death_event_ocr_roi]
    rows = []
    for image_path in iter_images(samples_root):
        result = evaluate_image(image_path, cfg, templates)
        weapon_result = None
        rgb = None
        if args.weapon_template_match and result.detected:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            weapon_result = match_death_weapon_template(rgb, cfg, weapon_templates)
            if args.save_weapon_debug:
                save_weapon_match_debug_images(str(image_path), rgb, cfg, weapon_templates, weapon_result, args.weapon_debug_dir)
        if args.ocr and result.detected:
            if rgb is None:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cfg.death_event_ocr_debug_source_name = str(image_path)
            for roi in rois:
                cfg.death_event_ocr_roi = roi
                for mode in modes:
                    cfg.death_event_ocr_preprocess_mode = mode
                    ocr_result = ocr_death_text(rgb, cfg)
                    rows.append(as_row(image_path, samples_root, result, cfg, ocr_result, weapon_result))
        else:
            rows.append(as_row(image_path, samples_root, result, cfg, None, weapon_result))

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
                "ocr_preprocess_mode", "ocr_roi", "ocr_scale", "ocr_text", "ocr_confidence", "ocr_reason",
                "normalized_ocr_text", "weapon_category", "emotion_category", "selected_reaction",
                "reaction_source", "matched_keywords", "category_reason",
                "weapon_match_enabled", "weapon_id", "weapon_display_name", "weapon_match_category", "weapon_template_category",
                "weapon_best_id", "weapon_best_display_name", "weapon_best_category",
                "weapon_best_score", "weapon_second_best_id", "weapon_second_best_score",
                "weapon_match_margin", "weapon_match_accepted", "weapon_match_reason", "weapon_template_path",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
