"""Mix converted vocals with instrumental (and optional harmonies) to MP3."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


class MixError(RuntimeError):
    """Raised when ffmpeg cannot produce the requested mix."""


def mix_tracks(
    vocals_path: str | Path,
    instrumental_path: str | Path,
    output_path: str | Path,
    *,
    harmony_path: str | Path | None = None,
    vocal_volume: float = 1.0,
    instrumental_volume: float = 1.0,
    harmony_volume: float = 1.0,
    bitrate: str = "192k",
) -> Path:
    """Mix 2-3 tracks to MP3, with independent linear volume controls."""
    vocals = Path(vocals_path).expanduser().resolve()
    instrumental = Path(instrumental_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    harmony = Path(harmony_path).expanduser().resolve() if harmony_path else None

    for label, path in (("Vocals", vocals), ("Instrumental", instrumental)):
        if not path.is_file():
            raise FileNotFoundError(f"{label}-Datei nicht gefunden: {path}")
    if harmony is not None and not harmony.is_file():
        raise FileNotFoundError(f"Harmonie-Datei nicht gefunden: {harmony}")
    if output.suffix.lower() != ".mp3":
        raise MixError("Die Ausgabedatei muss die Endung .mp3 haben.")
    if not 0.0 <= float(vocal_volume) <= 2.0:
        raise MixError("Vocal-Lautstärke muss zwischen 0.0 und 2.0 liegen.")
    if not 0.0 <= float(instrumental_volume) <= 2.0:
        raise MixError("Instrumental-Lautstärke muss zwischen 0.0 und 2.0 liegen.")
    if harmony is not None and not 0.0 <= float(harmony_volume) <= 2.0:
        raise MixError("Harmonie-Lautstärke muss zwischen 0.0 und 2.0 liegen.")
    if shutil.which("ffmpeg") is None:
        raise MixError("ffmpeg wurde nicht gefunden. Installiere ffmpeg und füge es zum PATH hinzu.")

    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"[0:a]volume={float(vocal_volume):.4f}[vocals];"
        f"[1:a]volume={float(instrumental_volume):.4f}[instrumental];"
    )
    if harmony is not None:
        filter_graph += (
            f"[2:a]volume={float(harmony_volume):.4f}[harmony];"
            "[vocals][instrumental][harmony]amix=inputs=3:"
        )
    else:
        filter_graph += "[vocals][instrumental]amix=inputs=2:"
    filter_graph += "duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.98[mix]"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(vocals),
        "-i",
        str(instrumental),
    ]
    if harmony is not None:
        command += ["-i", str(harmony)]
    command += [
        "-filter_complex",
        filter_graph,
        "-map",
        "[mix]",
        "-map_metadata",
        "-1",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip()
        raise MixError(
            f"ffmpeg konnte die MP3 nicht erstellen. {details or f'Exit-Code {exc.returncode}'}"
        ) from exc

    if not output.is_file() or output.stat().st_size == 0:
        raise MixError("ffmpeg meldete keinen Fehler, aber die MP3-Ausgabe fehlt oder ist leer.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Mischt Vocals, Instrumental (und optional Harmonien) zu einer MP3.")
    parser.add_argument("vocals", type=Path, help="Konvertierte Vocal-WAV")
    parser.add_argument("instrumental", type=Path, help="Instrumental-WAV von Demucs")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Ziel-MP3")
    parser.add_argument("--harmony", type=Path, default=None, help="Optionale Harmonie-/Backing-WAV der Originalstimme")
    parser.add_argument("--vocal-volume", type=float, default=1.0, help="0.0 bis 2.0")
    parser.add_argument("--instrumental-volume", type=float, default=1.0, help="0.0 bis 2.0")
    parser.add_argument("--harmony-volume", type=float, default=1.0, help="0.0 bis 2.0")
    args = parser.parse_args()

    try:
        output = mix_tracks(
            args.vocals,
            args.instrumental,
            args.output,
            harmony_path=args.harmony,
            vocal_volume=args.vocal_volume,
            instrumental_volume=args.instrumental_volume,
            harmony_volume=args.harmony_volume,
        )
    except (FileNotFoundError, MixError) as exc:
        parser.exit(1, f"Fehler: {exc}\n")
    print(f"MP3 erstellt: {output} ({output.stat().st_size} Bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
