"""Solução 2 do Desafio 2: preparação de máscaras e treino de U-Net leve.

Autor: Manoel Furtado
Data: 31/05/2026

Descrição:
    Este script transforma os rótulos de segmentação do dataset de fissuras
    em máscaras binárias, organiza os dados em treino/validação e, se
    solicitado, treina uma U-Net pequena em PyTorch. A proposta é manter uma
    alternativa leve e didática para segmentação sem depender de uma arquitetura
    grande.

Uso principal:
    python Desafio_2/solucoes/solucao_2_unet_leve/solucao_2_unet_leve.py --prepare
    python Desafio_2/solucoes/solucao_2_unet_leve/solucao_2_unet_leve.py --train

Artefato gerado:
    Desafio_2/solucoes/solucao_2_unet_leve/dataset_unet_masks/tiny_unet_fissuras.pt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_cv2():
    """Carrega OpenCV e NumPy apenas quando o script realmente precisa deles.

    Isso deixa a importação inicial do arquivo mais leve e permite apresentar
    uma mensagem de erro clara quando o ambiente correto não estiver ativo.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Use the project environment from env/env_vc_01.yml."
        ) from exc
    return cv2, np


def label_to_mask(cv2, np, label_path: Path, width: int, height: int):
    """Converte um label YOLO de segmentação em uma máscara binária.

    O label de segmentação armazena pontos normalizados entre 0 e 1:
    `classe x1 y1 x2 y2 ...`. A U-Net, por outro lado, aprende comparando a
    saída com uma imagem de máscara. Por isso cada polígono é convertido para
    coordenadas de pixel e preenchido com o valor 255.
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        # Ignora a classe na primeira posição e usa apenas os pares x/y do polígono.
        values = [float(value) for value in parts[1:]]
        points = []
        for x_norm, y_norm in zip(values[0::2], values[1::2]):
            # Multiplica pelo tamanho real para sair do espaço normalizado do YOLO.
            x = int(round(x_norm * (width - 1)))
            y = int(round(y_norm * (height - 1)))
            points.append([x, y])
        if len(points) >= 3:
            # Um polígono precisa de pelo menos três pontos para formar uma área.
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def paired_items(data_dir: Path):
    """Percorre o dataset original retornando somente pares imagem/label válidos."""

    for image_path in sorted((data_dir / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = data_dir / "labels" / f"{image_path.stem}.txt"
        if label_path.exists():
            yield image_path, label_path


def prepare_masks(args) -> Path:
    """Prepara a estrutura de diretórios usada pelo treino da U-Net.

    Saída esperada:
        dataset_unet_masks/
            train/images, train/masks
            val/images,   val/masks
    """

    cv2, np = require_cv2()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # `--clean` remove uma preparação anterior para evitar mistura de arquivos.
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(paired_items(data_dir))
    if not items:
        raise SystemExit(f"No image/label pairs found in {data_dir}")

    # Embaralhar com seed fixa deixa a divisão treino/validação reprodutível.
    random.seed(args.seed)
    random.shuffle(items)
    val_count = int(len(items) * args.val_fraction)

    for index, (image_path, label_path) in enumerate(items):
        # As primeiras imagens embaralhadas vão para validação; o restante vai para treino.
        split = "val" if index < val_count else "train"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        mask = label_to_mask(cv2, np, label_path, width, height)

        # Mantém a imagem original e salva a máscara com o mesmo nome-base em PNG.
        image_dest = output_dir / split / "images" / image_path.name
        mask_dest = output_dir / split / "masks" / f"{image_path.stem}.png"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_dest)
        cv2.imwrite(str(mask_dest), mask)

    print(f"Prepared U-Net masks at {output_dir}")
    return output_dir


def train_tiny_unet(args, dataset_dir: Path):
    """Treina uma U-Net compacta para segmentar fissuras.

    A rede recebe uma imagem RGB e produz um mapa de 1 canal. Cada pixel do mapa
    indica a chance de pertencer à classe fissura. O treino combina BCE, que
    aprende pixel a pixel, com Dice Loss, que ajuda em objetos finos e
    desbalanceados como fissuras.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Training needs the project environment from env/env_vc_01.yml."
        ) from exc

    class CrackDataset(Dataset):
        """Dataset PyTorch que carrega pares imagem/máscara preparados no disco."""

        def __init__(self, root: Path, split: str):
            self.images = sorted((root / split / "images").glob("*"))
            self.mask_dir = root / split / "masks"

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image_path = self.images[index]
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Redimensiona entrada e alvo para o mesmo tamanho fixo do treino.
            image = cv2.resize(image, (args.size, args.size))
            mask = cv2.resize(mask, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

            # OpenCV lê BGR; a rede recebe RGB normalizado em [0, 1].
            image = image[:, :, ::-1].astype(np.float32) / 255.0

            # A máscara vira um tensor 1xHxW com valores 0.0 ou 1.0.
            mask = (mask.astype(np.float32) / 255.0)[None, :, :]
            image = np.transpose(image, (2, 0, 1))
            return torch.tensor(image), torch.tensor(mask)

    class Block(nn.Module):
        """Bloco básico da U-Net: duas convoluções com ReLU preservando resolução."""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class TinyUNet(nn.Module):
        """U-Net pequena com dois níveis de encoder, ponte e decoder."""

        def __init__(self):
            super().__init__()

            # Encoder: reduz a resolução e aumenta canais para capturar contexto.
            self.down1 = Block(3, 16)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = Block(16, 32)
            self.pool2 = nn.MaxPool2d(2)
            self.bridge = Block(32, 64)

            # Decoder: recupera resolução e combina detalhes via skip connections.
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = Block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = Block(32, 16)

            # Saída de 1 canal: logit por pixel para a classe fissura.
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            b = self.bridge(self.pool2(d2))

            # Concatena mapas do encoder para recuperar bordas e detalhes finos.
            u2 = self.up2(b)
            u2 = torch.cat([u2, d2], dim=1)
            u2 = self.dec2(u2)
            u1 = self.up1(u2)
            u1 = torch.cat([u1, d1], dim=1)
            return self.out(self.dec1(u1))

    def dice_loss(logits, targets, eps=1e-6):
        """Calcula Dice Loss usando probabilidades derivadas dos logits."""

        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return 1 - ((2 * inter + eps) / (union + eps)).mean()

    # Usa GPU quando disponível; caso contrário, mantém o treino em CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyUNet().to(device)
    train_loader = DataLoader(
        CrackDataset(dataset_dir, "train"), batch_size=args.batch, shuffle=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Ciclo padrão de treino: zera gradientes, calcula perda e atualiza pesos.
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f}")

    # Salva apenas os pesos, mantendo o script como definição reprodutível da arquitetura.
    output_path = dataset_dir / "tiny_unet_fissuras.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


def parse_args(argv: list[str] | None = None):
    """Define os parâmetros de linha de comando da preparação e do treino."""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(script_dir.parent.parent / "data"))
    parser.add_argument("--output-dir", default=str(script_dir / "dataset_unet_masks"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args(argv)


def run(args) -> int:
    """Executa o fluxo solicitado pelo usuário.

    Se o dataset preparado ainda não existir, o script prepara as máscaras antes
    do treino. Assim `--train` funciona em uma execução limpa.
    """

    dataset_dir = Path(args.output_dir)
    if args.prepare or not dataset_dir.exists():
        dataset_dir = prepare_masks(args)
    if args.train:
        train_tiny_unet(args, dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
