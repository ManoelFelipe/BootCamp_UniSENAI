"""Solução 3 do Desafio 2: segmentação assistida com OpenCV e métricas.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Este script implementa uma linha de base clássica para segmentar fissuras
    com OpenCV. Ele gera máscaras por realce morfológico e bordas, compara as
    predições com os labels de segmentação e salva métricas como Precision,
    Recall, IoU e Dice em CSV.

Uso principal:
    python Desafio_2/solucoes/solucao_3_opencv_assistido/solucao_3_opencv_assistido.py
    python Desafio_2/solucoes/solucao_3_opencv_assistido/solucao_3_opencv_assistido.py --save-images

Artefato gerado:
    Desafio_2/solucoes/solucao_3_opencv_assistido/resultados/opencv_assistido/metricas_opencv_assistido.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_cv2():
    """Carrega OpenCV e NumPy quando o processamento de imagem é executado."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Use the project environment from env/env_vc_01.yml."
        ) from exc
    return cv2, np


def label_to_mask(cv2, np, label_path: Path, width: int, height: int):
    """Converte um arquivo YOLO de segmentação em máscara binária de referência.

    Essa máscara é o "gabarito" usado para calcular as métricas da abordagem
    OpenCV. Quando não há label, retorna uma máscara vazia para manter o fluxo
    robusto.
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if len(values) < 7:
            continue

        # A primeira coluna é a classe; as demais são pares x/y normalizados.
        coords = [float(value) for value in values[1:]]
        points = []
        for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
            # Converte de coordenada normalizada para pixel real da imagem.
            x = int(round(x_norm * (width - 1)))
            y = int(round(y_norm * (height - 1)))
            points.append([x, y])
        if len(points) >= 3:
            # Preenche o polígono para obter a região completa da fissura.
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def crack_mask(cv2, np, image, args):
    """Gera a máscara prevista de fissuras usando filtros clássicos.

    A combinação de black-hat e Canny tenta capturar fissuras escuras e finas:
    o black-hat destaca linhas escuras sobre fundo claro, enquanto o Canny
    reforça bordas.
    """

    # Trabalhar em tons de cinza simplifica os filtros morfológicos.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE melhora contraste local sem estourar a imagem inteira.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Black-hat destaca estruturas escuras menores que o elemento estruturante.
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (args.kernel, args.kernel))
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, kernel_line)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # Otsu escolhe automaticamente um limiar para separar destaque e fundo.
    _, threshold = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Canny complementa o limiar pegando bordas finas que o black-hat pode perder.
    edges = cv2.Canny(clahe, args.canny_low, args.canny_high)
    combined = cv2.bitwise_or(threshold, edges)

    # Fechamento conecta pequenos trechos quebrados antes da filtragem por área.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask = remove_small_components(cv2, np, mask, args.min_component_area)
    return mask


def remove_small_components(cv2, np, mask, min_area: int):
    """Remove componentes conectados menores que a área mínima informada."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255
    return output


def component_summary(cv2, np, mask, min_area: int = 500):
    """Resume componentes relevantes de uma máscara para inspeção diagnóstica."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    areas = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = 255
            areas.append(area)

    total_area = int(sum(areas))
    area_percent = 100.0 * total_area / max(mask.size, 1)
    return {
        "count": len(areas),
        "area_percent": round(area_percent, 2),
        "largest_area": max(areas, default=0),
        "filtered_mask": filtered,
    }


def metrics(np, pred_mask, true_mask):
    """Calcula métricas pixel a pixel entre máscara prevista e máscara real."""

    # Converte para booleano: qualquer pixel acima de zero conta como positivo.
    pred = pred_mask > 0
    true = true_mask > 0

    # Contagens básicas de classificação binária em nível de pixel.
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    union = int(np.logical_or(pred, true).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / union if union else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
    }


def overlay(cv2, np, image, mask, color=(0, 0, 255), alpha=0.45):
    """Sobrepõe a máscara na imagem original para avaliação visual."""

    output = image.copy()
    colored = np.zeros_like(image)
    colored[mask > 0] = color
    return cv2.addWeighted(output, 1.0, colored, alpha, 0)


def process(args) -> int:
    """Processa as imagens, calcula métricas e salva os resultados em CSV."""

    cv2, np = require_cv2()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    rows = []

    # Filtra somente imagens válidas e permite limitar a execução para testes rápidos.
    image_paths = [
        path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.limit:
        image_paths = image_paths[: args.limit]

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]

        # Predição OpenCV e máscara real são comparadas no mesmo tamanho.
        pred = crack_mask(cv2, np, image, args)
        true = label_to_mask(cv2, np, labels_dir / f"{image_path.stem}.txt", width, height)
        row = {"image": image_path.name}
        row.update(metrics(np, pred, true))
        rows.append(row)

        if args.save_images:
            # As imagens salvas ajudam a entender falsos positivos e falsos negativos.
            cv2.imwrite(str(output_dir / f"{image_path.stem}_mask_pred.png"), pred)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_overlay.jpg"), overlay(cv2, np, image, pred))

        print(
            f"{image_path.name}: IoU={row['iou']:.4f} Dice={row['dice']:.4f} "
            f"Precision={row['precision']:.4f} Recall={row['recall']:.4f}"
        )

    # O CSV consolida os resultados por imagem para uso no relatório.
    csv_path = output_dir / "metricas_opencv_assistido.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "precision", "recall", "iou", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    # Média simples das métricas para uma visão geral do comportamento do baseline.
    if rows:
        avg = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in ["precision", "recall", "iou", "dice"]
        }
        print("Averages:", {key: round(value, 4) for key, value in avg.items()})
    print(f"Saved metrics to {csv_path}")
    return 0


def parse_args(argv: list[str] | None = None):
    """Define os parâmetros da abordagem OpenCV assistida."""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(script_dir.parent.parent / "data"))
    parser.add_argument("--output-dir", default=str(script_dir / "resultados" / "opencv_assistido"))
    parser.add_argument("--kernel", type=int, default=17)
    parser.add_argument("--canny-low", type=int, default=40)
    parser.add_argument("--canny-high", type=int, default=140)
    parser.add_argument("--min-component-area", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(process(parse_args()))
