"""Smoke and functional checks for the vc_01 conda environment.

Create the environment once from env/:
    mamba env create -f env_vc_01.yml --channel-priority flexible

Run inside the activated environment:
    mamba activate vc_01
    python test_env_vc_01.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Check:
    name: str
    func: Callable[[], None]


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
SKIPPED: list[tuple[str, str]] = []
CHECKS: list[Check] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def skip(reason: str) -> None:
    raise RuntimeError(f"SKIP: {reason}")


def check(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        CHECKS.append(Check(name, func))
        return func

    return decorator


@check("python runtime")
def _() -> None:
    require(sys.version_info[:2] == (3, 11), f"expected Python 3.11, got {platform.python_version()}")
    require(Path(sys.executable).exists(), f"missing executable: {sys.executable}")


@check("core data imports")
def _() -> None:
    modules = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "polars",
        "pyarrow",
        "duckdb",
        "sqlalchemy",
        "duckdb_engine",
        "statsmodels",
        "openpyxl",
        "dash",
        "dash_bootstrap_components",
        "tqdm",
        "datasets",
    ]
    for module_name in modules:
        importlib.import_module(module_name)


@check("vision and video imports")
def _() -> None:
    modules = [
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "clip",
        "skimage",
        "cv2",
        "ffmpeg",
        "av",
    ]
    for module_name in modules:
        importlib.import_module(module_name)


@check("numpy version for pytorch stack")
def _() -> None:
    import numpy as np

    major = int(np.__version__.split(".", maxsplit=1)[0])
    require(major < 2, f"expected NumPy <2 for vc_01, got {np.__version__}")


@check("torch install and tensor ops")
def _() -> None:
    import torch

    require(torch.__version__.startswith("2.2.2"), f"expected torch 2.2.2, got {torch.__version__}")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ x
    require(y.shape == (2, 2), f"unexpected tensor shape: {y.shape}")
    require(float(y[0, 0]) == 7.0, f"unexpected tensor value: {y}")


@check("cuda visibility")
def _() -> None:
    import torch

    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        skip("CUDA runtime/GPU is not available; CPU torch checks passed separately")
    device = torch.device("cuda")
    x = torch.ones((128, 128), device=device)
    y = x @ x
    torch.cuda.synchronize()
    require(float(y[0, 0].detach().cpu()) == 128.0, "CUDA matrix multiplication returned bad value")


@check("torchvision image transform")
def _() -> None:
    import torch
    from torchvision import transforms

    image = torch.zeros((3, 32, 32), dtype=torch.float32)
    transform = transforms.Compose([transforms.Resize((16, 16), antialias=True)])
    output = transform(image)
    require(tuple(output.shape) == (3, 16, 16), f"unexpected transformed shape: {output.shape}")


@check("torchaudio basic waveform")
def _() -> None:
    import torch
    import torchaudio.functional as F

    waveform = torch.sin(torch.linspace(0, 1, 16000)).unsqueeze(0)
    resampled = F.resample(waveform, orig_freq=16000, new_freq=8000)
    require(resampled.shape[-1] > 0, f"bad resampled shape: {resampled.shape}")


@check("opencv and scikit-image processing")
def _() -> None:
    import cv2
    import numpy as np
    from skimage.color import rgb2gray

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    gray = rgb2gray(image)
    require(gray.shape == (32, 32), f"unexpected gray shape: {gray.shape}")
    edges = cv2.Canny(image, 50, 150)
    require(edges.shape == (32, 32), f"unexpected edge shape: {edges.shape}")


@check("opencv video write/read")
def _() -> None:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "check.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
        require(writer.isOpened(), "cv2.VideoWriter did not open")
        for index in range(5):
            frame = np.full((32, 32, 3), index * 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        capture = cv2.VideoCapture(path)
        require(capture.isOpened(), "cv2.VideoCapture did not open generated video")
        ok, frame = capture.read()
        capture.release()
    require(ok, "could not read generated video frame")
    require(frame.shape == (32, 32, 3), f"unexpected video frame shape: {frame.shape}")


@check("pyav encode/decode")
def _() -> None:
    import av
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "check.mp4"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=5)
            stream.width = 32
            stream.height = 32
            stream.pix_fmt = "yuv420p"
            for index in range(3):
                array = np.full((32, 32, 3), index * 60, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        with av.open(str(path), mode="r") as container:
            frames = list(container.decode(video=0))
    require(len(frames) >= 1, "pyav decoded no frames")


@check("ffmpeg executable and python wrapper")
def _() -> None:
    import ffmpeg

    executable = shutil.which("ffmpeg")
    require(executable is not None, "ffmpeg executable is not on PATH")
    completed = subprocess.run(
        [executable, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(completed.returncode == 0, f"ffmpeg -version failed: {completed.stderr}")
    probe = ffmpeg.input("color=c=black:s=16x16:d=0.1", f="lavfi").output("pipe:", f="null")
    command = ffmpeg.compile(probe)
    require(command, "ffmpeg-python did not compile a command")


@check("clip model construction")
def _() -> None:
    import clip

    models = clip.available_models()
    require("RN50" in models, f"RN50 missing from available CLIP models: {models}")
    tokens = clip.tokenize(["a small test"])
    require(tuple(tokens.shape) == (1, 77), f"unexpected CLIP token shape: {tokens.shape}")


@check("ultralytics package metadata")
def _() -> None:
    from ultralytics import YOLO

    require(YOLO is not None, "ultralytics.YOLO missing")


@check("pytorchvideo transform import")
def _() -> None:
    if importlib.util.find_spec("pytorchvideo") is None:
        skip("pytorchvideo is optional and is not installed in vc_01")

    try:
        from pytorchvideo.transforms import ApplyTransformToKey
    except ModuleNotFoundError as exc:
        if exc.name == "torchvision.transforms.functional_tensor":
            skip("pytorchvideo is installed, but incompatible with this torchvision version")
        raise

    transform = ApplyTransformToKey(key="video", transform=lambda value: value + 1)
    output = transform({"video": 2})
    require(output["video"] == 3, f"unexpected pytorchvideo transform result: {output}")


@check("datasets local dataset")
def _() -> None:
    from datasets import Dataset

    dataset = Dataset.from_dict({"image_id": [1, 2], "label": [0, 1]})
    require(len(dataset) == 2, f"unexpected dataset length: {len(dataset)}")
    require(dataset[1]["label"] == 1, f"unexpected dataset row: {dataset[1]}")


@check("dataframe stack roundtrip")
def _() -> None:
    import duckdb
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    table = pa.Table.from_pandas(frame)
    polars_frame = pl.from_arrow(table)
    require(polars_frame["y"].sum() == 12, "polars/arrow sum failed")
    result = duckdb.sql("select avg(y) from frame").fetchone()[0]
    require(result == 4, f"duckdb average failed: {result}")


@check("matplotlib non-interactive render")
def _() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "plot.png"
        plt.figure()
        plt.imshow([[0, 1], [1, 0]], cmap="gray")
        plt.savefig(output)
        plt.close()
        require(output.stat().st_size > 0, "plot file was empty")


@check("optional network model checks")
def _() -> None:
    if not os.environ.get("RUN_NETWORK_CHECKS"):
        skip("set RUN_NETWORK_CHECKS=1 to allow model/data downloads")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    require(model is not None, "YOLO model download/load failed")


def run_checks() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print()

    for item in CHECKS:
        print(f"[RUN] {item.name}")
        try:
            item.func()
        except RuntimeError as exc:
            if str(exc).startswith("SKIP:"):
                reason = str(exc).replace("SKIP:", "", 1).strip()
                SKIPPED.append((item.name, reason))
                print(f"[SKIP] {item.name}: {reason}")
            else:
                FAILED.append((item.name, traceback.format_exc()))
                print(f"[FAIL] {item.name}")
        except Exception:
            FAILED.append((item.name, traceback.format_exc()))
            print(f"[FAIL] {item.name}")
        else:
            PASSED.append(item.name)
            print(f"[PASS] {item.name}")
        print()

    print("=" * 72)
    print(f"Passed: {len(PASSED)} | Skipped: {len(SKIPPED)} | Failed: {len(FAILED)}")

    if SKIPPED:
        print("\nSkipped checks:")
        for name, reason in SKIPPED:
            print(f"- {name}: {reason}")

    if FAILED:
        print("\nFailures:")
        for name, details in FAILED:
            print(f"\n--- {name} ---")
            print(details.rstrip())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())
