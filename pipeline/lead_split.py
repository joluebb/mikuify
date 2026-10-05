"""Split Demucs vocals into lead vocals and harmony/backing vocals.

The split runs in the dedicated tools venv (.venv-tools) because
audio-separator requires numpy>=2/librosa>=0.10, which conflicts with the
RVC stack's pinned numpy 1.26.4/librosa 0.9.1. The same file serves two
roles:

- main venv: import split_lead_and_harmonies() to launch the worker below;
- tools venv: executed as a script with --worker for the actual separation,
  using the official UVR karaoke model via audio-separator's model index.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PYTHON = PROJECT_ROOT / ".venv-tools" / "Scripts" / "python.exe"
UVR_MODEL_DIR = PROJECT_ROOT / "uvr_models"
# Mel-Band-Roformer would be stronger, but its librosa path needs numba/
# llvmlite.dll, which a Windows application-control policy blocks on this
# machine. The VR-arch karaoke model avoids numba and runs on the GPU.
KARAOKE_MODEL = "6_HP-Karaoke-UVR.pth"


class LeadSplitError(RuntimeError):
    """Raised when the lead/harmony split cannot be produced."""


def _worker_main(args: argparse.Namespace) -> int:
    """Run inside .venv-tools; performs the actual separation."""
    # Must precede the audio_separator import so librosa's lazy imports get
    # the compatible numba/resampy surface on this machine.
    from tools_compat import install_tools_compat

    install_tools_compat()

    from audio_separator.separator import Separator

    separator = Separator(
        model_file_dir=str(Path(args.model_dir)),
        output_dir=str(Path(args.output_dir)),
        output_format="WAV",
    )
    separator.load_model(args.model)
    stems = separator.separate(str(Path(args.input)))

    lead = next((s for s in stems if re.search(r"\(Vocals\)", s)), None)
    harmony = next((s for s in stems if re.search(r"\(Instrumental\)", s)), None)
    if not lead or not harmony:
        print(json.dumps({"error": f"Unerwartete Stems: {stems}"}))
        return 2
    print(json.dumps({"lead": lead, "harmony": harmony}))
    return 0


def split_lead_and_harmonies(
    vocals_path: str | Path,
    output_dir: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    """Split separated vocals into (lead, harmony) WAV files.

    Raises:
        LeadSplitError: The tools venv, the karaoke model, or the split run failed.
    """
    source = Path(vocals_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Vocal-Datei nicht gefunden: {source}")
    if not TOOLS_PYTHON.is_file():
        raise LeadSplitError(
            "Das Tools-venv (.venv-tools) fehlt. Es enthält die Lead-/Harmonie-"
            "Trennung; Setup siehe README (Abschnitt Einrichtung)."
        )

    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(TOOLS_PYTHON),
        str(Path(__file__).resolve()),
        "--worker",
        "--input",
        str(source),
        "--output-dir",
        str(destination),
        "--model-dir",
        str(UVR_MODEL_DIR),
        "--model",
        KARAOKE_MODEL,
    ]
    if progress:
        progress(
            "Trenne Lead-Stimme und Harmonien "
            "(6HP-Karaoke-Modell, erster Lauf lädt es) …"
        )
    try:
        completed = subprocess.run(
            command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise LeadSplitError(f"Die Lead-/Harmonie-Trennung konnte nicht gestartet werden: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise LeadSplitError(
            "Die Lead-/Harmonie-Trennung ist fehlgeschlagen. "
            f"Details: {detail[-1] if detail else 'unbekannt'}"
        )

    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise LeadSplitError(
            "Die Lead-/Harmonie-Trennung lieferte kein Ergebnis. "
            "Prüfe die Konsolenausgabe und den GPU-Speicher."
        ) from exc

    lead_path = destination / payload["lead"]
    harmony_path = destination / payload["harmony"]
    missing = [p for p in (lead_path, harmony_path) if not p.is_file()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise LeadSplitError(f"Die Lead-/Harmonie-Trennung erzeugte fehlende Dateien: {names}")
    return lead_path, harmony_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model", default=KARAOKE_MODEL)
    args = parser.parse_args()

    if args.worker:
        return _worker_main(args)
    if not args.input or not args.output_dir:
        parser.error("--input und --output-dir sind außerhalb des Worker-Modus erforderlich.")

    try:
        lead, harmony = split_lead_and_harmonies(args.input, args.output_dir)
    except (FileNotFoundError, LeadSplitError) as exc:
        parser.exit(1, f"Fehler: {exc}\n")
    print(f"Lead:     {lead}")
    print(f"Harmonie: {harmony}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
