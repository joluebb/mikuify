"""RVC-v2 inference for locally supplied voice and support models."""

from __future__ import annotations

import dataclasses
import gc
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Callable


class ConversionError(RuntimeError):
    """Raised when RVC cannot convert the separated vocals."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
HUBERT_DIR = MODELS_DIR / "hubert"
RMVPE_PATH = MODELS_DIR / "rmvpe.pt"
DEFAULT_MODEL_NAME = "miku_default"


@dataclasses.dataclass(frozen=True)
class RVCModel:
    """One locally installed voice checkpoint with its retrieval index."""

    name: str
    pth_path: Path
    index_path: Path | None


def _find_index_for(pth_path: Path) -> Path | None:
    """Best-effort pairing of a voice checkpoint with a FAISS index file."""
    stem = pth_path.stem.lower()
    local_indexes = sorted(pth_path.parent.glob("*.index"))
    for candidate in local_indexes:
        if stem in candidate.stem.lower():
            return candidate
    if len(local_indexes) == 1:
        return local_indexes[0]
    for candidate in sorted(MODELS_DIR.rglob("*.index")):
        if stem in candidate.stem.lower():
            return candidate
    return None


def discover_models(models_dir: Path = MODELS_DIR) -> list[RVCModel]:
    """Find RVC checkpoints in models/ (flat or one sub-directory level deep).

    A checkpoint placed directly in models/ is named after its file stem; a
    checkpoint inside a sub-directory is named after that folder. Shared
    helper directories such as models/hubert contain no .pth files and are
    therefore never listed as voice models.
    """
    found: dict[str, RVCModel] = {}
    if not models_dir.is_dir():
        return []
    checkpoints = sorted(models_dir.glob("*.pth")) + sorted(models_dir.glob("*/*.pth"))
    for pth_path in checkpoints:
        name = pth_path.stem if pth_path.parent == models_dir else pth_path.parent.name
        if name in found or name == "hubert":
            continue
        found[name] = RVCModel(
            name=name, pth_path=pth_path, index_path=_find_index_for(pth_path)
        )
    return [found[name] for name in sorted(found)]


def resolve_model(name: str | None = None) -> RVCModel:
    """Return the requested model, or a sensible default when name is None."""
    models = discover_models()
    if not models:
        raise ConversionError(
            "Kein Stimmenmodell gefunden. Lege in models/ eine RVC-.pth-Datei "
            "(idealerweise mit passender .index-Datei) ab."
        )
    if name is not None:
        for model in models:
            if model.name == name:
                return model
        available = ", ".join(model.name for model in models)
        raise ConversionError(
            f"Stimmenmodell '{name}' wurde nicht gefunden. Verfügbar: {available}"
        )
    for model in models:
        if model.name == DEFAULT_MODEL_NAME:
            return model
    return models[0]


def validate_support_models() -> None:
    """Fail early if shared RMVPE/HuBERT helper files are missing."""
    if not RMVPE_PATH.is_file():
        raise ConversionError(f"RMVPE-Hilfsmodell fehlt: {RMVPE_PATH}")

    required_hubert = (HUBERT_DIR / "config.json", HUBERT_DIR / "model.safetensors")
    missing = [path for path in required_hubert if not path.is_file()]
    if missing:
        files = "\n".join(f"  - {path}" for path in missing)
        raise ConversionError(
            f"Das lokale HuBERT-Modell ist unvollständig. Erwartet werden:\n{files}"
        )


def validate_model(model: RVCModel) -> None:
    """Fail early if the selected voice model's files are missing."""
    if not model.pth_path.is_file():
        raise ConversionError(f"Modell-Checkpoint fehlt: {model.pth_path}")
    if model.index_path is not None and not model.index_path.is_file():
        raise ConversionError(f"Index-Datei fehlt: {model.index_path}")


def _restore_env(previous_cache: str | None, previous_source: str | None) -> None:
    """Restore the FAISS cache environment after a conversion run."""
    if previous_cache is None:
        os.environ.pop("MIKUIFY_FAISS_VECTORS", None)
    else:
        os.environ["MIKUIFY_FAISS_VECTORS"] = previous_cache
    if previous_source is None:
        os.environ.pop("MIKUIFY_FAISS_SOURCE", None)
    else:
        os.environ["MIKUIFY_FAISS_SOURCE"] = previous_source


def convert_vocals_batch(
    items: Sequence[tuple[str | Path, str | Path]],
    pitch_shift: int = 0,
    *,
    model_name: str | None = None,
    index_rate: float = 0.30,
    protection: float = 0.50,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Convert one or more separated vocal WAVs with the same voice model.

    All inputs are converted with a single model load; the retrieval index
    (if configured) is prepared once for the whole batch.

    Args:
        items: (input_path, output_path) pairs, e.g. [(lead_wav, out_wav),
            (harmony_wav, out_wav)] for a lead+harmony conversion.
        pitch_shift: Semitone transposition applied by RVC/RMVPE.
        model_name: Voice model to use; defaults to miku_default.
        index_rate: Retrieval index mix (0 to 1).
        protection: RVC protection for unvoiced consonants (0 to 0.5).
        progress: Optional status callback for a UI/CLI caller.
    """
    pairs: list[tuple[Path, Path]] = []
    for raw_source, raw_target in items:
        source = Path(raw_source).expanduser().resolve()
        target = Path(raw_target).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Getrennte Vocal-Datei nicht gefunden: {source}")
        if source == target:
            raise ConversionError("Eingabe- und Ausgabedatei müssen verschieden sein.")
        pairs.append((source, target))
    if not pairs:
        raise ConversionError("Keine Konvertierungsziele angegeben.")

    targets = [target for _, target in pairs]
    if len(set(targets)) != len(targets):
        raise ConversionError("Die Ausgabedateien müssen sich unterscheiden.")
    if not -24 <= int(pitch_shift) <= 24:
        raise ConversionError("Pitch-Shift muss zwischen -24 und +24 Halbtönen liegen.")
    if not 0.0 <= float(index_rate) <= 1.0:
        raise ConversionError("Index-Stärke muss zwischen 0.0 und 1.0 liegen.")
    if not 0.0 <= float(protection) <= 0.5:
        raise ConversionError("Konsonantenschutz muss zwischen 0.0 und 0.5 liegen.")

    model = resolve_model(model_name)
    validate_support_models()
    validate_model(model)

    try:
        import torch
    except ImportError as exc:
        raise ConversionError("PyTorch fehlt im Projekt-venv.") from exc
    if not torch.cuda.is_available():
        raise ConversionError(
            "CUDA ist nicht verfügbar. Prüfe den NVIDIA-Treiber und den CUDA-Build von PyTorch."
        )

    # FAISS and this Windows PyTorch build load incompatible OpenMP DLLs in one
    # process. Export the IVF-Flat vectors in a short-lived FAISS-only process;
    # inference uses them through a CPU PyTorch exact-L2 proxy.
    temp_base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Temp" / "opencode"
    temp_base.mkdir(parents=True, exist_ok=True)
    index_temp: tempfile.TemporaryDirectory | None = None
    index_vectors_path: Path | None = None
    index_path = model.index_path
    use_index = index_path is not None and float(index_rate) > 0
    if use_index and index_path is not None:
        index_temp = tempfile.TemporaryDirectory(prefix="mikuify-faiss-", dir=str(temp_base))
        index_vectors_path = Path(index_temp.name) / "index_vectors.npy"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.index_cache",
                    str(index_path),
                    str(index_vectors_path),
                ],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            index_temp.cleanup()
            index_temp = None
            detail = getattr(exc, "stderr", None) or str(exc)
            raise ConversionError(
                f"Der FAISS-Index konnte nicht vorbereitet werden: {detail}"
            ) from exc
    elif progress and index_path is None:
        progress("Kein Index zu diesem Modell gefunden; Index-Beimischung entfällt.")

    previous_cache = os.environ.get("MIKUIFY_FAISS_VECTORS")
    previous_source = os.environ.get("MIKUIFY_FAISS_SOURCE")
    if index_vectors_path is not None and index_path is not None:
        os.environ["MIKUIFY_FAISS_VECTORS"] = str(index_vectors_path)
        os.environ["MIKUIFY_FAISS_SOURCE"] = str(index_path)
    else:
        os.environ.pop("MIKUIFY_FAISS_VECTORS", None)
        os.environ.pop("MIKUIFY_FAISS_SOURCE", None)

    try:
        # This RVC package otherwise fetches helper models on demand. Keep
        # inference offline and pass the verified local paths explicitly.
        from pipeline.rvc_compat import install_rvc_import_compat

        install_rvc_import_compat()
        from infer_rvc_python import BaseLoader
    except Exception as exc:
        _restore_env(previous_cache, previous_source)
        if index_temp is not None:
            index_temp.cleanup()
        raise ConversionError(
            "Die RVC-Inferenzbibliothek konnte nicht geladen werden. "
            "Prüfe die Installation aus requirements.txt."
        ) from exc

    tag = "miku_local"
    loader = None
    generated_paths = [source.with_name(f"{source.stem}_edited.wav") for source, _ in pairs]
    for generated in generated_paths:
        if generated.exists():
            generated.unlink()

    try:
        if progress:
            progress(f"Lade lokales HuBERT-, RMVPE- und '{model.name}'-Modell …")
        loader = BaseLoader(
            only_cpu=False,
            hubert_path=str(HUBERT_DIR),
            rmvpe_path=str(RMVPE_PATH),
        )
        loader.apply_conf(
            tag=tag,
            file_model=str(model.pth_path),
            pitch_algo="rmvpe+",
            pitch_lvl=int(pitch_shift),
            file_index=str(index_path) if index_path is not None else "",
            index_influence=float(index_rate),
            respiration_median_filtering=3,
            # Vocal level is controlled by the mixer; keep RVC envelope
            # matching neutral and avoid an extra librosa-dependent pass.
            envelope_ratio=1.0,
            consonant_breath_protection=float(protection),
            resample_sr=0,
        )
        if progress:
            subject = "Lead- und Harmoniestimme" if len(pairs) > 1 else "Vocals"
            progress(f"Konvertiere {subject} mit RMVPE …")
        results = loader(
            [str(source) for source, _ in pairs],
            tag,
            overwrite=False,
            parallel_workers=1,
            type_output="wav",
            show_progress=False,
        )
        del results  # Output matching is done via deterministic file names.

        converted: list[Path] = []
        for (source, target), generated in zip(pairs, generated_paths):
            if not generated.is_file():
                raise ConversionError(
                    f"RVC hat für '{source.name}' keine Ausgabedatei erzeugt. "
                    "Prüfe die Konsolenausgabe, Modell-/Index-Kompatibilität "
                    "und den GPU-Speicher."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(generated), str(target))
            converted.append(target)
        return converted
    except ConversionError:
        raise
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower() or "cuda" in message.lower():
            raise ConversionError(
                "RVC konnte die Konvertierung auf der GPU nicht abschließen. "
                "Schließe andere GPU-Anwendungen und versuche es erneut. "
                f"Details: {message}"
            ) from exc
        raise ConversionError(f"RVC-Inferenz fehlgeschlagen: {message}") from exc
    except Exception as exc:
        raise ConversionError(f"RVC-Inferenz fehlgeschlagen: {exc}") from exc
    finally:
        if loader is not None:
            try:
                loader.unload_models()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for generated in generated_paths:
            if generated.exists():
                generated.unlink()
        _restore_env(previous_cache, previous_source)
        if index_temp is not None:
            index_temp.cleanup()


def convert_vocals(
    input_path: str | Path,
    output_path: str | Path,
    pitch_shift: int = 0,
    *,
    model_name: str | None = None,
    index_rate: float = 0.30,
    protection: float = 0.50,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Convert a single separated vocal WAV using a local RVC-v2 model."""
    (converted,) = convert_vocals_batch(
        [(input_path, output_path)],
        pitch_shift=pitch_shift,
        model_name=model_name,
        index_rate=index_rate,
        protection=protection,
        progress=progress,
    )
    return converted
