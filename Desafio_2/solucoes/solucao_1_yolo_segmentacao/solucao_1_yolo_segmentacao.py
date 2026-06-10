"""Solução 1 do Desafio 2: preparação, treino e uso de YOLO segmentação.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Este script organiza o dataset de fissuras no formato esperado pelo
    Ultralytics YOLO, opcionalmente treina um modelo de segmentação, executa
    predições e exporta pesos treinados. Ele trabalha com imagens e labels no
    formato YOLO de segmentação, sem duplicar o dataset original.

Uso principal:
    python Desafio_2/solucoes/solucao_1_yolo_segmentacao.py --prepare
    python Desafio_2/solucoes/solucao_1_yolo_segmentacao.py --train

Artefato base:
    Desafio_2/modelos/yolo26n-seg.pt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Localiza pares imagem/label válidos no dataset original.

    Um par só entra no treino quando existe uma imagem em `images/` e um arquivo
    `.txt` com o mesmo nome em `labels/`. Isso evita passar imagens sem rótulo
    para o treinamento supervisionado.
    """

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    pairs = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def write_split_file(pairs: list[tuple[Path, Path]], output_dir: Path, split: str):
    """Escreve train.txt, val.txt ou test.txt com caminhos absolutos das imagens."""

    split_path = output_dir / f"{split}.txt"
    split_path.write_text(
        "\n".join(image_path.resolve().as_posix() for image_path, _ in pairs) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(args) -> Path:
    """Cria divisão reprodutível de treino, validação e teste.

    O Ultralytics aceita arquivos `.txt` apontando para as imagens. Assim, o
    script cria apenas listas de caminhos e um `data.yaml`, preservando os
    arquivos originais em `Desafio_2/data`.
    """

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # `--clean` força recriação do split e remove resultados anteriores no diretório.
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"No image/label pairs found in {data_dir}")

    # A semente garante que a divisão seja repetível no relatório.
    random.seed(args.seed)
    random.shuffle(pairs)

    # Primeiro separa teste, depois validação, e o restante vira treino.
    test_count = int(len(pairs) * args.test_fraction)
    val_count = int(len(pairs) * args.val_fraction)
    test_pairs = pairs[:test_count]
    val_pairs = pairs[test_count : test_count + val_count]
    train_pairs = pairs[test_count + val_count :]

    write_split_file(train_pairs, output_dir, "train")
    write_split_file(val_pairs, output_dir, "val")
    if test_pairs:
        write_split_file(test_pairs, output_dir, "test")

    # data.yaml é o arquivo de configuração consumido pelo YOLO.
    data_yaml = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: train.txt",
        "val: val.txt",
    ]
    if test_pairs:
        lines.append("test: test.txt")
    lines.extend(["names:", "  0: fissura", ""])
    data_yaml.write_text("\n".join(lines), encoding="utf-8")

    print(f"Pairs: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"Created YOLO dataset at {output_dir}")
    return data_yaml


def train(args, data_yaml: Path):
    """Treina o modelo YOLO de segmentação com os parâmetros informados."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Use the project environment from env/env_vc_01.yml."
        ) from exc

    # `args.model` pode apontar para o peso base local ou para um nome reconhecido
    # pelo Ultralytics. O padrão do projeto fica em Desafio_2/modelos.
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path(args.output_dir) / "runs"),
        name="fissuras_yolo_seg",
    )


def predict(args):
    """Executa inferência em imagens usando pesos treinados."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Use the project environment from env/env_vc_01.yml."
        ) from exc

    if not args.weights:
        raise SystemExit("--weights is required for prediction")

    # As predições são salvas para inspeção visual posterior.
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        save_txt=True,
        project=str(Path(args.output_dir) / "predicoes"),
        name="yolo_seg",
    )


def export(args):
    """Exporta pesos treinados para outro formato, como ONNX."""

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Use the project environment from env/env_vc_01.yml."
        ) from exc

    if not args.weights:
        raise SystemExit("--weights is required for export")
    model = YOLO(args.weights)
    model.export(format=args.export_format, imgsz=args.imgsz, int8=args.int8)


def parse_args(argv: list[str] | None = None):
    """Define a interface de linha de comando do script YOLO."""

    script_dir = Path(__file__).resolve().parent
    default_data = script_dir.parent.parent / "data"
    default_output = script_dir / "dataset_yolo_split"
    default_model = script_dir.parent.parent / "modelos" / "yolo26n-seg.pt"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(default_data))
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--model", default=str(default_model))
    parser.add_argument("--weights", default="")
    parser.add_argument("--source", default=str(default_data / "images"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--export-format", default="onnx")
    parser.add_argument("--int8", action="store_true")
    return parser.parse_args(argv)


def run(args) -> int:
    """Coordena as ações solicitadas: preparar, treinar, predizer e exportar."""

    data_yaml = Path(args.output_dir) / "data.yaml"

    # Sem flags, o comportamento padrão é preparar o dataset.
    no_action = not any([args.prepare, args.train, args.predict, args.export])
    should_prepare = args.prepare or args.train or no_action
    if should_prepare and (args.prepare or not data_yaml.exists()):
        data_yaml = prepare_dataset(args)
    if args.train:
        train(args, data_yaml)
    if args.predict:
        predict(args)
    if args.export:
        export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
