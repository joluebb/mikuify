"""Vocal/instrumental separation using Demucs' htdemucs model.

The Demucs CLI is invoked as a subprocess so its own argument handling and
model loading remain isolated from the rest of the application.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class SeparationError(RuntimeError):
    """Raised when Demucs cannot separate the supplied audio."""


def separate_audio(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = "htdemucs",
    device: str = "cuda",
) -> tuple[Path, Path]:
    """Separate ``input_path`` into vocals and instrumental WAV files.

    Returns:
        A ``(vocals_path, instrumental_path)`` tuple.

    Raises:
        FileNotFoundError: The input file does not exist.
        SeparationError: ffmpeg, Demucs, CUDA, or the separation run failed.
    """
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Audio-Datei nicht gefunden: {source}")
    if source.suffix.lower() not in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
        raise SeparationError(f"Nicht unterstütztes Audioformat: {source.suffix or '(ohne Endung)'}")
    if shutil.which("ffmpeg") is None:
        raise SeparationError(
            "ffmpeg wurde nicht gefunden. Installiere ffmpeg und füge es zum PATH hinzu."
        )
    if device not in {"cuda", "cpu"}:
        raise SeparationError("device muss entweder 'cuda' oder 'cpu' sein.")

    try:
        import torch
    except ImportError as exc:
        raise SeparationError(
            "PyTorch fehlt. Installiere zuerst die Abhängigkeiten aus requirements.txt."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise SeparationError(
            "CUDA ist nicht verfügbar. Prüfe den NVIDIA-Treiber und installiere den "
            "CUDA-Build von PyTorch; alternativ kannst du den CLI-Aufruf mit --device cpu starten."
        )

    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "--name",
        model,
        "--device",
        device,
        "--out",
        str(destination),
        str(source),
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SeparationError(
            "Demucs ist nicht installiert. Installiere zuerst die Abhängigkeiten aus requirements.txt."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SeparationError(
            f"Demucs ist mit Exit-Code {exc.returncode} fehlgeschlagen. "
            "Prüfe die Ausgabe oben sowie freien Speicherplatz und GPU-Speicher."
        ) from exc

    track_dir = destination / model / source.stem
    vocals_path = track_dir / "vocals.wav"
    instrumental_path = track_dir / "no_vocals.wav"
    missing = [path for path in (vocals_path, instrumental_path) if not path.is_file()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise SeparationError(f"Demucs beendete sich, aber Ausgabedateien fehlen: {missing_list}")

    return vocals_path, instrumental_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trennt Gesang und Instrumental mit Demucs (htdemucs)."
    )
    parser.add_argument("input", type=Path, help="Eingabe-Audiodatei, z. B. eine MP3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work"),
        help="Ausgabeordner (Standard: ./work)",
    )
    parser.add_argument("--model", default="htdemucs", help="Demucs-Modell (Standard: htdemucs)")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Rechengerät (Standard: cuda)",
    )
    args = parser.parse_args()

    try:
        vocals_path, instrumental_path = separate_audio(
            args.input,
            args.output_dir,
            model=args.model,
            device=args.device,
        )
    except (FileNotFoundError, SeparationError) as exc:
        parser.exit(1, f"Fehler: {exc}\n")

    print(f"Vocals:      {vocals_path}")
    print(f"Instrumental: {instrumental_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
