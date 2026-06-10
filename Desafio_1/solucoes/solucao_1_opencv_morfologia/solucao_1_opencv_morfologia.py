"""Solução 1 do Desafio 1: contagem de parafusos com OpenCV.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Este script implementa uma solução clássica de visão computacional para
    contar parafusos em imagens. A proposta evita treinamento de modelo, pois
    o Desafio 1 possui poucas imagens disponíveis. O pipeline combina realce
    de contraste, limiarização, morfologia matemática, contornos e filtros
    geométricos.

Uso principal:
    python Desafio_1/solucoes/solucao_1_opencv_morfologia.py \
        --input-dir Desafio_1/data/images

Observação:
    A contagem automática deve ser interpretada como estimativa assistida.
    Em imagens difíceis, partes do mesmo parafuso podem ser separadas em mais
    de um contorno, e parafusos encostados podem virar um único contorno.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    """Representa um contorno aprovado como possível parafuso.

    Os campos guardam a caixa envolvente e métricas geométricas usadas para
    filtrar ruído: área, circularidade e proporção entre largura/altura.
    """

    x: int
    y: int
    w: int
    h: int
    area: float
    circularity: float
    aspect_ratio: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def touches_image_corner(candidate: Candidate, width: int, height: int, margin: float) -> bool:
    """Verifica se o candidato está colado em um canto da imagem.

    Esse filtro evita falsos positivos como bordas, sombras e objetos cortados
    no canto inferior/direito. Um candidato próximo de apenas uma borda pode ser
    válido; o descarte ocorre quando toca duas bordas adjacentes.
    """

    if margin < 0:
        return False

    left = candidate.x <= margin
    top = candidate.y <= margin
    right = candidate.x + candidate.w >= width - margin
    bottom = candidate.y + candidate.h >= height - margin
    return (left and top) or (right and top) or (left and bottom) or (right and bottom)


def require_cv2():
    """Importa OpenCV e NumPy somente quando o processamento é executado."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Use the project environment from env/env_vc_01.yml."
        ) from exc
    return cv2, np


def iter_images(input_dir: Path) -> list[Path]:
    """Lista imagens de entrada aceitas pelo script."""

    return sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_manual_threshold_mask(cv2, np, image, args):
    """Gera máscara por limiar manual.

    Este modo é útil quando o fundo é claro e os parafusos aparecem escuros.
    Ele segue um fluxo didático: escala de cinza, CLAHE, suavização, threshold
    invertido e limpeza morfológica.
    """

    # 1) A escala de cinza simplifica o problema para intensidade luminosa.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE melhora contraste local sem estourar tanto áreas claras.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3) Blur reduz pequenas variações que gerariam contornos quebrados.
    if args.manual_blur == "median":
        filtered = cv2.medianBlur(enhanced, 5)
    else:
        filtered = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=1.0)

    # 4) Threshold invertido: pixels escuros viram objeto e fundo claro vira 0.
    _, manual = cv2.threshold(
        filtered,
        float(args.manual_threshold),
        255,
        cv2.THRESH_BINARY_INV,
    )

    # 5) Open remove ruídos pequenos; close tenta unir pequenas falhas internas.
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(manual, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return cleaned


def build_masks(cv2, np, image, args):
    """Cria máscaras candidatas para segmentação dos parafusos.

    No modo automático, várias estratégias são geradas e a melhor é escolhida
    depois por pontuação. No modo manual, apenas a máscara de limiar manual é
    usada, dando controle direto ao usuário.
    """

    if args.mask_mode == "manual":
        return [(f"limiar_manual_t{int(args.manual_threshold)}", build_manual_threshold_mask(cv2, np, image, args))]

    # Fluxo automático: realce + suavização antes de testar limiares diferentes.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, otsu_dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, otsu_light = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        4,
    )

    edges = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = []
    # Cada máscara captura um tipo de contraste diferente:
    # - otsu_dark: objetos escuros contra fundo claro;
    # - otsu_light: objetos claros/reflexos;
    # - adaptive: variações locais de iluminação;
    # - edges: bordas quando a textura interna atrapalha o threshold.
    for name, mask in {
        "otsu_dark": otsu_dark,
        "otsu_light": otsu_light,
        "adaptive": adaptive,
        "edges": edges,
    }.items():
        if args.mask_mode == "auto" or args.mask_mode == name:
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            masks.append((name, cleaned))
    return masks


def contour_candidates(cv2, np, mask, args) -> list[Candidate]:
    """Extrai e filtra contornos que podem representar parafusos."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Candidate] = []
    image_height, image_width = mask.shape[:2]

    for contour in contours:
        # Área remove ruídos pequenos e regiões grandes demais para serem parafusos.
        area = float(cv2.contourArea(contour))
        if area < args.min_area or area > args.max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        # Aspect ratio limita formatos extremos, mas permite parafusos alongados.
        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > args.max_aspect_ratio:
            continue

        # Circularidade muito baixa costuma indicar borda quebrada ou ruído fino.
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < args.min_circularity:
            continue

        candidate = Candidate(
            x=x,
            y=y,
            w=w,
            h=h,
            area=area,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
        )
        if touches_image_corner(candidate, image_width, image_height, args.corner_margin):
            continue

        candidates.append(candidate)

    return sorted(candidates, key=lambda item: (item.y, item.x))


def pick_best_mask(cv2, np, image, args):
    """Escolhe a melhor máscara pelo conjunto de candidatos detectados.

    A pontuação tenta equilibrar quantidade de objetos, área típica, variação de
    áreas e excesso de componentes. Isso evita escolher uma máscara que parece
    rica em contornos, mas na prática está fragmentando o mesmo parafuso.
    """

    best_name = ""
    best_mask = None
    best_candidates: list[Candidate] = []
    best_score = float("-inf")

    for name, mask in build_masks(cv2, np, image, args):
        candidates = contour_candidates(cv2, np, mask, args)
        if not candidates:
            score = float("-inf")
        else:
            areas = np.array([candidate.area for candidate in candidates], dtype=float)
            median_area = float(np.median(areas))
            mean_area = float(np.mean(areas))
            area_cv = float(np.std(areas) / max(mean_area, 1.0))
            area_penalty = abs(median_area - args.expected_area) / max(args.expected_area, 1)
            small_component_penalty = max(0.0, (args.expected_area * 0.35 - median_area) / args.expected_area)
            noise_penalty = max(0, len(candidates) - args.max_reasonable_candidates) * 0.35

            # Score maior é melhor. Penalidades altas indicam candidatos muito
            # pequenos, muito irregulares ou numerosos demais para a cena.
            score = (
                len(candidates) * 0.15
                - min(area_penalty, 6.0)
                - min(area_cv, 3.0) * 0.35
                - small_component_penalty * 3.0
                - noise_penalty
            )

        if score > best_score:
            best_name = name
            best_mask = mask
            best_candidates = candidates
            best_score = score

    return best_name, best_mask, best_candidates


def annotate(cv2, image, candidates: list[Candidate], mask_name: str):
    """Desenha caixas e rótulos na imagem final para auditoria visual."""

    annotated = image.copy()
    for index, candidate in enumerate(candidates, start=1):
        p1 = (candidate.x, candidate.y)
        p2 = (candidate.x + candidate.w, candidate.y + candidate.h)
        cv2.rectangle(annotated, p1, p2, (0, 180, 0), 2)
        cv2.putText(
            annotated,
            str(index),
            (candidate.x, max(candidate.y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )

    label = f"count={len(candidates)} method={mask_name}"
    cv2.putText(
        annotated,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def process_image(cv2, np, image_path: Path, output_dir: Path, args) -> dict[str, object]:
    """Processa uma imagem e salva os artefatos de saída."""

    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask_name, mask, candidates = pick_best_mask(cv2, np, image, args)
    annotated = annotate(cv2, image, candidates, mask_name)

    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_annotated.jpg"), annotated)
    if mask is not None:
        cv2.imwrite(str(output_dir / f"{stem}_mask_{mask_name}.png"), mask)

    return {
        "image": image_path.name,
        "count": len(candidates),
        "method": mask_name,
        "mean_area": round(
            sum(candidate.area for candidate in candidates) / len(candidates), 2
        )
        if candidates
        else 0,
        "min_area": min((candidate.area for candidate in candidates), default=0),
        "max_area": max((candidate.area for candidate in candidates), default=0),
    }


def run(args) -> int:
    """Executa o processamento em lote e grava o CSV de contagens."""

    cv2, np = require_cv2()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    rows = []
    for image_path in image_paths:
        row = process_image(cv2, np, image_path, output_dir, args)
        rows.append(row)
        print(f"{row['image']}: {row['count']} screws ({row['method']})")

    csv_path = output_dir / "contagens_opencv_morfologia.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "count", "method", "mean_area", "min_area", "max_area"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None):
    """Define parâmetros de linha de comando do pipeline OpenCV."""

    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent.parent / "data" / "images"
    default_output = script_dir / "resultados" / "opencv_morfologia"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(default_input))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--max-area", type=float, default=20000.0)
    parser.add_argument("--expected-area", type=float, default=1600.0)
    parser.add_argument("--max-reasonable-candidates", type=int, default=20)
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--min-circularity", type=float, default=0.02)
    parser.add_argument("--corner-margin", type=float, default=4.0)
    parser.add_argument(
        "--mask-mode",
        choices=["auto", "manual", "edges", "otsu_dark", "otsu_light", "adaptive"],
        default="edges",
    )
    parser.add_argument("--manual-threshold", type=float, default=90.0)
    parser.add_argument("--manual-blur", choices=["gauss", "median"], default="median")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
