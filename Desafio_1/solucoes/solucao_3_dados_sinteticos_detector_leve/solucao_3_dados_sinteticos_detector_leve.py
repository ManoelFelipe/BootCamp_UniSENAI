"""Solução 3 do Desafio 1: dados sintéticos e detector leve de parafusos.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Este script cria um dataset YOLO com pseudo-rótulos para detecção de
    parafusos. Ele usa um detector clássico em OpenCV para sugerir caixas,
    gera variações simples das imagens e, opcionalmente, inicia o treino de um
    modelo YOLO leve quando o Ultralytics está instalado.

Uso principal:
    python Desafio_1/solucoes/solucao_3_dados_sinteticos_detector_leve/solucao_3_dados_sinteticos_detector_leve.py
    python Desafio_1/solucoes/solucao_3_dados_sinteticos_detector_leve/solucao_3_dados_sinteticos_detector_leve.py --train-yolo

Artefato gerado:
    Desafio_1/solucoes/solucao_3_dados_sinteticos_detector_leve/dataset_yolo_sintetico/data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    """Representa uma caixa delimitadora no formato de pixel do OpenCV."""

    x: int
    y: int
    w: int
    h: int


def require_cv2():
    """Carrega OpenCV e NumPy com uma mensagem clara em caso de ambiente incorreto."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Use the project environment from env/env_vc_01.yml."
        ) from exc
    return cv2, np


def iter_images(input_dir: Path) -> list[Path]:
    """Lista somente arquivos com extensões de imagem aceitas pelo script."""

    return sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def weak_boxes(cv2, np, image, args) -> list[Box]:
    """Gera pseudo-caixas usando processamento clássico de imagem.

    A ideia é obter rótulos fracos: eles aceleram uma primeira versão do
    dataset, mas ainda devem ser revisados visualmente antes de virarem base
    final de treino.
    """

    # Converte para tons de cinza e aumenta contraste local para destacar peças.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Suavização reduz ruído antes do limiar automático de Otsu.
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Operações morfológicas fecham falhas pequenas e removem pontos isolados.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[Box] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.min_area or area > args.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        # Descartar proporções extremas ajuda a reduzir falsas detecções.
        if max(w / h, h / w) > args.max_aspect_ratio:
            continue
        boxes.append(Box(x, y, w, h))

    # Ordenação estável facilita inspeção e comparação de resultados.
    return sorted(boxes, key=lambda box: (box.y, box.x))


def yolo_line(box: Box, image_width: int, image_height: int) -> str:
    """Converte uma caixa em linha YOLO: classe, centro x/y, largura e altura."""

    # YOLO espera coordenadas normalizadas pelo tamanho da imagem.
    x_center = (box.x + box.w / 2) / image_width
    y_center = (box.y + box.h / 2) / image_height
    width = box.w / image_width
    height = box.h / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def write_pair(cv2, image, boxes: list[Box], image_path: Path, label_path: Path):
    """Salva uma imagem e seu arquivo `.txt` de labels no formato YOLO."""

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)
    height, width = image.shape[:2]
    label_path.write_text(
        "\n".join(yolo_line(box, width, height) for box in boxes),
        encoding="utf-8",
    )


def transform_boxes_for_flip(boxes: list[Box], width: int, height: int, flip_code: int):
    """Ajusta as caixas quando a imagem é espelhada.

    `flip_code=1` espelha horizontalmente, `flip_code=0` verticalmente e outros
    valores representam espelhamento nos dois eixos.
    """

    transformed = []
    for box in boxes:
        if flip_code == 1:
            transformed.append(Box(width - box.x - box.w, box.y, box.w, box.h))
        elif flip_code == 0:
            transformed.append(Box(box.x, height - box.y - box.h, box.w, box.h))
        else:
            transformed.append(Box(width - box.x - box.w, height - box.y - box.h, box.w, box.h))
    return transformed


def brightness_variant(cv2, np, image, alpha: float, beta: int):
    """Cria uma variação simples de brilho/contraste da imagem."""

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def create_dataset(args) -> Path:
    """Cria o dataset YOLO sintético com pseudo-rótulos e aumentações simples."""

    cv2, np = require_cv2()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # `--clean` garante que a nova geração não misture arquivos antigos.
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    generated = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        boxes = weak_boxes(cv2, np, image, args)
        if not boxes:
            print(f"Warning: no pseudo-labels for {image_path.name}", file=sys.stderr)

        # Cada imagem original gera um pequeno conjunto de variações coerentes.
        variants = [(image, boxes, "orig")]
        height, width = image.shape[:2]
        for flip_code, name in [(1, "flip_h"), (0, "flip_v")]:
            variants.append(
                (
                    cv2.flip(image, flip_code),
                    transform_boxes_for_flip(boxes, width, height, flip_code),
                    name,
                )
            )
        for alpha, beta, name in [(0.85, -10, "dark"), (1.15, 12, "bright")]:
            variants.append((brightness_variant(cv2, np, image, alpha, beta), boxes, name))

        for variant_image, variant_boxes, name in variants:
            generated.append((image_path.stem, name, variant_image, variant_boxes))

    # A divisão é feita depois das aumentações para manter distribuição simples.
    random.shuffle(generated)
    val_count = max(1, int(len(generated) * args.val_fraction))
    val_items = set(range(val_count))

    for index, (stem, name, image, boxes) in enumerate(generated):
        split = "val" if index in val_items else "train"
        file_stem = f"{stem}_{name}"

        # Estrutura esperada pelo Ultralytics: images/split e labels/split.
        write_pair(
            cv2,
            image,
            boxes,
            output_dir / "images" / split / f"{file_stem}.jpg",
            output_dir / "labels" / split / f"{file_stem}.txt",
        )

    # `data.yaml` é o ponto de entrada usado pelo YOLO no treinamento.
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: screw",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created weak YOLO dataset at {output_dir}")
    print(f"Review labels before using this as final training data: {data_yaml}")
    return data_yaml


def train_yolo(data_yaml: Path, args):
    """Treina YOLO usando o dataset de pseudo-rótulos criado por este script."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Use the project environment from env/env_vc_01.yml."
        ) from exc

    # O modelo padrão é pequeno para manter o experimento leve.
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path(args.output_dir) / "runs"),
        name="screw_detector_weak_labels",
    )


def parse_args(argv: list[str] | None = None):
    """Define os parâmetros de geração de dados e treino opcional."""

    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent.parent / "data" / "images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(default_input))
    parser.add_argument("--output-dir", default=str(script_dir / "dataset_yolo_sintetico"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--max-area", type=float, default=20000.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=8.0)
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args(argv)


def run(args) -> int:
    """Gera o dataset e, se solicitado, inicia o treino YOLO."""

    data_yaml = create_dataset(args)
    if args.train_yolo:
        train_yolo(data_yaml, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
