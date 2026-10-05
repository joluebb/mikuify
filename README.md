# Mikuify — lokale RVC-Musikkonvertierung

Eine lokale Windows-Weboberfläche: MP3 hochladen → Demucs trennt Vocals und Instrumental → ein Karaoke-Modell trennt Lead- und Harmoniestimme → RVC-v2/RMVPE konvertiert **beide** Stimmen zu Miku → ffmpeg mischt Miku-Lead, Miku-Harmonien (optional) und Instrumental zu einer MP3. Gradio lauscht nur auf `127.0.0.1`; `share=False`, es werden keine Cloud-APIs verwendet.

## Voraussetzungen

- Windows 10/11 x64
- Python 3.10 x64 (getestet mit 3.10.11)
- NVIDIA-Treiber und CUDA-fähige NVIDIA-GPU (getestet mit RTX 2060, 6 GB VRAM)
- ffmpeg im Windows-`PATH`
- Die lokalen RVC- und Hilfsmodelle wie unten beschrieben

## Einrichtung

PowerShell im Projektordner `E:\code\mikuify` öffnen:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Die CUDA-PyTorch-Wheels kommen über den in `requirements.txt` festgelegten PyTorch-CUDA-12.1-Index. Das venv ist unabhängig von den global installierten Python-Versionen.

Zusätzlich ein zweites, separates venv für die Lead-/Harmonie-Trennung (`audio-separator` benötigt neuere numpy/librosa-Versionen als der RVC-Stack und läuft deshalb als eigener Prozess):

```powershell
py -3.10 -m venv .venv-tools
.\.venv-tools\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tools\Scripts\python.exe -m pip install -r requirements-tools.txt
.\.venv-tools\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
```

Der letzte Schritt stellt ein CUDA- und NumPy-2-kompatibles torch/torchvision-Paar her, das `audio-separator` von sich aus nicht installiert. Beim ersten Split-Lauf lädt die Anwendung das Karaoke-Modell automatisch nach `uvr_models/`.

## Modelle

Modelldateien werden nicht mit dem Projekt ausgeliefert. Jedes Stimmenmodell liegt in einem eigenen Unterordner von `models/`; die Oberfläche listet alle gefundenen Modelle automatisch im Auswahlfeld auf (Anzeigename = Ordnername):

```text
models/
├── miku_default/                                # ein Unterordner pro Modell
│   ├── miku_default_rvc.pth
│   └── added_IVF4457_..._miku_default_rvc_v2.index
├── miku_mellow/
│   ├── miku_mellow_rvc.pth
│   └── added_IVF3160_..._miku_mellow_rvc_v2.index
├── miku_power/
│   ├── miku_power_rvc.pth
│   └── added_IVF10688_..._miku_power_rvc_v2.index
├── rmvpe.pt                                      # gemeinsame Hilfsmodelle
└── hubert/
    ├── config.json
    └── model.safetensors
```

Eine `.index`-Datei ist optional; ohne sie entfällt die Index-Beimischung. Das Stimmenmodell muss ein **RVC-v2-Checkpoint** sein. Nach dem Hinzufügen weiterer Modelle die Oberfläche einmal neu starten. Details stehen in [`models/README.md`](models/README.md).

## Start

```powershell
.\.venv\Scripts\python.exe app.py
```

Die Oberfläche öffnet sich unter <http://127.0.0.1:7860>. Im Auswahlfeld kann zwischen allen gefundenen Stimmenmodellen aus `models/` gewechselt werden. Pitch-Shift ist in Halbtönen einstellbar; Vocal-, Instrumental- und Miku-Harmonienpegel sind lineare Multiplikatoren von 0 bis 2. Die Checkbox schaltet die Miku-Harmonien an/aus; die Feineinstellungen bieten Index-Beimischung und Konsonantenschutz gegen Artefakte. Die Verarbeitung läuft nacheinander, damit GPU-Speicher geschont wird.

Fertige MP3s bleiben unter `outputs/` erhalten. Eingabe-Kopie, getrennte Spuren, umgewandelte Vocals und der temporäre Index-Cache werden nach dem Auftrag entfernt.

## CLI-Tests

Demucs-Trennung:

```powershell
.\.venv\Scripts\python.exe -m pipeline.separate "E:\Audio\song.mp3" --output-dir work
```

Lead-/Harmonie-Trennung einer Demucs-Vocal-Spur:

```powershell
.\.venv\Scripts\python.exe -m pipeline.lead_split "work\htdemucs\song\vocals.wav" --output-dir work\split
```

Mixing vorhandener WAV-Dateien (mit optionaler Harmonie-Spur):

```powershell
.\.venv\Scripts\python.exe -m pipeline.mix `
  "work\miku_vocals.wav" "work\htdemucs\song\no_vocals.wav" `
  --output outputs\mixed.mp3 --vocal-volume 1.0 --instrumental-volume 0.85 `
  --harmony "work\split\vocals_(Instrumental)_6_HP-Karaoke-UVR.wav" --harmony-volume 1.0
```

Ersetze `song` durch den Namen der Eingabedatei ohne Endung.

Die Voice Conversion wird über dieselbe Funktion ausgeführt, die auch die Weboberfläche verwendet, damit Modelle und temporäre Dateien identisch geprüft und bereinigt werden.

## Abhängigkeits- und Windows-Hinweise

- PyTorch, TorchAudio und TorchVision sind gemeinsam auf `2.3.1+cu121` abgestimmt. `soundfile==0.12.1` ergänzt den Audio-Schreib-Backend, der im Windows-TorchAudio-Wheel für Demucs benötigt wird.
- `infer_rvc_python==1.3.1` stellt die Python-API bereit; `transformers==4.44.2` und die übrigen RVC-Abhängigkeiten sind gepinnt. Die Bibliothek zieht `librosa==0.9.1` über `torchcrepe` nach. Da diese Windows-Umgebung `llvmlite.dll` durch eine Anwendungssteuerungsrichtlinie (Smart App Control) blockiert, stellt `pipeline/rvc_compat.py` nur die von der RMVPE-Konfiguration benötigten Audiofunktionen lokal bereit. Der nicht verwendete CREPE-Pfad wird deaktiviert; die Windows-Sicherheitseinstellung wird nicht verändert.
- Der Lead-/Harmonie-Split nutzt standardmäßig das Roformer-Karaoke-Modell `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956` aus dem offiziellen UVR-Modellarchiv (SDR ~10,2) und wird beim ersten Lauf automatisch nach `uvr_models/` geladen. Weil diese Windows-Umgebung `llvmlite.dll` bzw. numba-Teile per Anwendungssteuerungsrichtlinie blockiert, stellt `pipeline/tools_compat.py` im Tools-Prozess einen Pass-Through-Stub bereit, der librosas reine Python-Pfade funktionsfähig hält. Als Fallback ist das numba-leichtere VR-Modell `6_HP-Karaoke-UVR.pth` vorgesehen (`VR_FALLBACK_MODEL` in `pipeline/lead_split.py`).
- FAISS und PyTorch laden in dieser Windows-Umgebung inkompatible OpenMP-DLLs. Der FAISS-Index wird daher in einem kurzlebigen FAISS-only-Prozess gelesen; während der RVC-Inferenz erfolgt die L2-Suche über PyTorch auf der CPU. Das vermeidet `KMP_DUPLICATE_LIB_OK` und dessen bekannte Risiken.
- `fairseq` wird nicht installiert.

## Fehlersuche

- **ffmpeg fehlt:** `ffmpeg -version` in PowerShell prüfen und sicherstellen, dass ffmpeg im `PATH` liegt.
- **CUDA fehlt:** `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` ausführen und NVIDIA-Treiber sowie CUDA-PyTorch-Wheels kontrollieren.
- **Modell fehlt:** Dateinamen und Ordnerstruktur unter `models/` mit obigem Baum vergleichen.
- **Zu wenig GPU-Speicher:** Andere GPU-Anwendungen schließen und den Auftrag erneut starten.

## RVC-Inferenzbibliothek

Verwendet wird `infer_rvc_python`, weil es eine Python-API bietet und RMVPE sowie `.pth`-/`.index`-Modelle direkt unterstützt. Die Abhängigkeiten werden gegenüber unbeschränkten neuesten Versionen gepinnt; Fairseq wird dadurch nicht benötigt.
