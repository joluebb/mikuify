"""Local Gradio UI for Demucs -> RVC/RMVPE -> MP3 mixing."""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

from pipeline.convert import (
    ConversionError,
    convert_vocals_batch,
    discover_models,
    resolve_model,
    validate_model,
    validate_support_models,
)
from pipeline.lead_split import LeadSplitError, split_lead_and_harmonies
from pipeline.mix import MixError, mix_tracks
from pipeline.separate import SeparationError, separate_audio


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_ROOT = Path(tempfile.gettempdir()) / "opencode"
AVAILABLE_MODELS = [model.name for model in discover_models()]


def _validate_runtime(model_name: str | None) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg fehlt. Installiere ffmpeg und füge es zum Windows-PATH hinzu.")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch fehlt. Aktiviere das Projekt-venv und installiere requirements.txt.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA ist nicht verfügbar. Prüfe den NVIDIA-Treiber und installiere den CUDA-Build von PyTorch."
        )
    validate_support_models()
    validate_model(resolve_model(model_name))


def process_audio(
    uploaded_audio: str | None,
    model_name: str | None,
    pitch_shift: float,
    vocal_volume: float,
    instrumental_volume: float,
    include_harmonies: bool,
    harmony_volume: float,
    index_rate: float,
    protection: float,
    progress=gr.Progress(track_tqdm=True),
):
    """Run all pipeline stages and return an MP3 for playback and download."""
    if not uploaded_audio:
        raise gr.Error("Bitte wähle zuerst eine MP3-Datei aus.")

    source = Path(uploaded_audio).expanduser().resolve()
    if not source.is_file():
        raise gr.Error("Die hochgeladene Audiodatei wurde nicht gefunden. Bitte lade sie erneut hoch.")
    if source.suffix.lower() != ".mp3":
        raise gr.Error("Bitte lade eine MP3-Datei hoch.")

    try:
        _validate_runtime(model_name)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"mikuified_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp3"
    )
    final_output = OUTPUT_DIR / output_name

    try:
        with tempfile.TemporaryDirectory(prefix="mikuify-job-", dir=str(TEMP_ROOT)) as work_dir:
            work = Path(work_dir)
            safe_input = work / "input.mp3"
            shutil.copy2(source, safe_input)

            progress(0.03, desc="Prüfe lokale Modelle und CUDA …")
            progress(0.08, desc="Trenne Vocals und Instrumental mit Demucs …")
            vocals_path, instrumental_path = separate_audio(
                safe_input,
                work / "separated",
                model="htdemucs",
                device="cuda",
            )
            if not vocals_path.is_file() or not instrumental_path.is_file():
                raise SeparationError(
                    "Die Trennung lief durch, aber die Ausgabedateien fehlen. "
                    "Bitte erneut starten."
                )

            harmony_path = None
            if include_harmonies:
                progress(0.42, desc="Trenne Lead-Stimme und Harmonien …")
                vocals_path, harmony_path = split_lead_and_harmonies(
                    vocals_path,
                    work / "split",
                    progress=lambda message: progress(0.46, desc=message),
                )
                progress(0.60, desc="Lead/Harmonie-Trennung fertig.")

            progress(0.62, desc="Starte Miku-Voice-Conversion …")

            def conversion_status(message: str) -> None:
                if "RMVPE" in message:
                    progress(0.68, desc=message)
                else:
                    progress(0.72, desc=message)

            conversions = [(vocals_path, work / "miku_lead.wav")]
            harmony_converted: Path | None = None
            if include_harmonies and harmony_path is not None:
                harmony_converted = work / "miku_harmonies.wav"
                conversions.append((harmony_path, harmony_converted))

            converted_paths = convert_vocals_batch(
                conversions,
                pitch_shift=int(pitch_shift),
                model_name=model_name,
                index_rate=float(index_rate),
                protection=float(protection),
                progress=conversion_status,
            )
            converted_lead = converted_paths[0]
            converted_harmony = converted_paths[1] if len(converted_paths) > 1 else None

            progress(0.88, desc="Mische Vocals und Instrumental und erstelle MP3 …")
            mix_tracks(
                converted_lead,
                instrumental_path,
                final_output,
                harmony_path=converted_harmony,
                vocal_volume=float(vocal_volume),
                instrumental_volume=float(instrumental_volume),
                harmony_volume=float(harmony_volume),
            )

        progress(1.0, desc="Fertig — temporäre Zwischendateien wurden entfernt.")
        return str(final_output), str(final_output)
    except (FileNotFoundError, SeparationError, LeadSplitError, ConversionError, MixError, RuntimeError) as exc:
        if final_output.exists():
            final_output.unlink()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        if final_output.exists():
            final_output.unlink()
        raise gr.Error(f"Unerwarteter Fehler in der lokalen Pipeline: {exc}") from exc


def _cleanup_stale_job_dirs() -> None:
    """Remove leftovers from crashed runs (e.g. killed worker processes).

    Only deletes job folders older than one hour so a still-queued job is
    never wiped while running.
    """
    """Remove leftovers from crashed runs (e.g. killed worker processes)."""
    if not TEMP_ROOT.is_dir():
        return
    cutoff = time.time() - 3600
    for stale in TEMP_ROOT.glob("mikuify-job-*"):
        try:
            if stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            continue
    for stale in TEMP_ROOT.glob("mikuify-faiss-*"):
        try:
            if stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            continue


with gr.Blocks(title="Mikuify — lokale RVC-Konvertierung", analytics_enabled=False) as demo:
    gr.Markdown(
        """
        # Mikuify
        Lade eine MP3 hoch. Die Datei wird lokal mit Demucs getrennt, Lead-Stimme
        und Harmonien werden getrennt und **beide** mit dem lokalen RVC-v2-Modell
        und RMVPE von Miku gesungen, dann mit dem Instrumental gemischt. Es werden
        keine Cloud-APIs verwendet.
        """
    )

    with gr.Row():
        upload = gr.File(
            label="MP3 hochladen",
            type="filepath",
            file_types=[".mp3"],
        )
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None,
            label="Miku-Stimmenmodell",
            info="Gefundene Modelle aus models/. Nach dem Hinzufügen neuer Modelle die Oberfläche neu starten.",
        )
    with gr.Row():
        pitch = gr.Slider(
            minimum=-12,
            maximum=12,
            value=0,
            step=1,
            label="Pitch-Shift (Halbtöne)",
            info="0 erhält die Originalhöhe; höhere Werte hellen die Stimme zusätzlich auf.",
        )

    with gr.Row():
        vocal_gain = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.75,
            step=0.05,
            label="Miku-Vocals (linearer Pegel)",
        )
        instrumental_gain = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.05,
            label="Instrumental (linearer Pegel)",
        )

    with gr.Row():
        include_harmonies = gr.Checkbox(
            value=True,
            label="Miku-Harmonien beimischen",
            info="Lead- und Harmoniestimme werden getrennt und beide von Miku gesungen. Ausschalten entfernt die Harmonien aus dem Ergebnis.",
        )
        harmony_gain = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.05,
            label="Miku-Harmonien (linearer Pegel)",
            info="1.0 = ungefähr Original-Lautstärke der Harmonien.",
        )

    with gr.Accordion("Feineinstellungen gegen Glitches", open=True):
        index_rate = gr.Slider(
            minimum=0.0,
            maximum=0.8,
            value=0.30,
            step=0.05,
            label="Index-Beimischung",
            info="Niedriger (z. B. 0.20–0.35) klingt oft weniger metallisch oder abgehackt.",
        )
        protection = gr.Slider(
            minimum=0.0,
            maximum=0.5,
            value=0.50,
            step=0.05,
            label="Konsonantenschutz",
            info="Höher schützt leise/ungepitchte Konsonanten stärker; bei Bedarf bis 0.50 erhöhen.",
        )

    start = gr.Button("Start", variant="primary")
    with gr.Row():
        result_audio = gr.Audio(label="Ergebnis", type="filepath", format="mp3", interactive=False)
        download = gr.DownloadButton(label="MP3 herunterladen", value=None)

    start.click(
        fn=process_audio,
        inputs=[
            upload,
            model_selector,
            pitch,
            vocal_gain,
            instrumental_gain,
            include_harmonies,
            harmony_gain,
            index_rate,
            protection,
        ],
        outputs=[result_audio, download],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    _cleanup_stale_job_dirs()
    demo.queue(default_concurrency_limit=1, max_size=2).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
