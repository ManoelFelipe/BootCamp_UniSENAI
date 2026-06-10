"""Count screws using a calibrated template and non-max suppression."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Use the project environment from env/env_vc_01.yml."
        ) from exc
    return cv2, np


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Use x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def load_template(cv2, reference_image_path: Path, args):
    reference = cv2.imread(str(reference_image_path))
    if reference is None:
        raise ValueError(f"Could not read reference image: {reference_image_path}")

    if args.template_image:
        template = cv2.imread(str(Path(args.template_image)))
        if template is None:
            raise ValueError(f"Could not read template image: {args.template_image}")
        return template

    if args.template_box:
        x, y, w, h = args.template_box
    else:
        roi = cv2.selectROI("Select one isolated screw", reference, showCrosshair=True)
        cv2.destroyWindow("Select one isolated screw")
        x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        raise ValueError("Template ROI must have positive width and height.")

    return reference[y : y + h, x : x + w].copy()


def nms(np, boxes, scores, overlap_threshold: float):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_np.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (width * height) / area[order[1:]]
        order = order[np.where(overlap <= overlap_threshold)[0] + 1]

    return keep


def match_image(cv2, np, image, template, args):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    boxes = []
    scores = []

    scales = [float(item) for item in args.scales.split(",")]
    for scale in scales:
        resized = cv2.resize(
            template_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        h, w = resized.shape[:2]
        if h >= image_gray.shape[0] or w >= image_gray.shape[1]:
            continue

        response = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(response >= args.threshold)
        for x, y in zip(xs, ys):
            boxes.append((int(x), int(y), int(w), int(h)))
            scores.append(float(response[y, x]))

    keep = nms(np, boxes, scores, args.nms_threshold)
    return [(boxes[index], scores[index]) for index in keep]


def annotate(cv2, image, matches):
    output = image.copy()
    for index, (box, score) in enumerate(matches, start=1):
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 180, 0), 2)
        cv2.putText(
            output,
            f"{index}:{score:.2f}",
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"count={len(matches)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def run(args) -> int:
    cv2, np = require_cv2()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    reference_path = Path(args.reference_image) if args.reference_image else image_paths[0]
    template = load_template(cv2, reference_path, args)
    cv2.imwrite(str(output_dir / "template_usado.jpg"), template)

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        matches = match_image(cv2, np, image, template, args)
        annotated = annotate(cv2, image, matches)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_template_matching.jpg"), annotated)
        avg_score = sum(score for _, score in matches) / len(matches) if matches else 0.0
        rows.append(
            {
                "image": image_path.name,
                "count": len(matches),
                "avg_score": round(avg_score, 4),
                "needs_review": avg_score < args.review_score or len(matches) == 0,
            }
        )
        print(f"{image_path.name}: {len(matches)} matches avg_score={avg_score:.3f}")

    csv_path = output_dir / "contagens_template_matching.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "count", "avg_score", "needs_review"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None):
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent.parent / "data" / "images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(default_input))
    parser.add_argument("--output-dir", default=str(script_dir / "resultados" / "template_matching"))
    parser.add_argument("--reference-image", default="")
    parser.add_argument("--template-image", default="")
    parser.add_argument("--template-box", type=parse_box, default=None, help="ROI as x,y,w,h")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--review-score", type=float, default=0.70)
    parser.add_argument("--scales", default="0.75,0.9,1.0,1.1,1.25")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
