# Modelle für die lokale RVC-Inferenz

## Stimmenmodelle

Jedes Stimmenmodell liegt in einem eigenen Unterordner. Die Weboberfläche
listet automatisch jeden Ordner mit einer RVC-v2-Checkpoint-Datei (`.pth`)
als auswählbares Modell auf (Anzeigename = Ordnername):

```text
models/
├── miku_default/
│   ├── miku_default_rvc.pth
│   └── added_IVF4457_Flat_nprobe_1_miku_default_rvc_v2.index
├── miku_mellow/
│   ├── miku_mellow_rvc.pth
│   └── added_IVF3160_Flat_nprobe_1_miku_mellow_rvc_v2.index
└── miku_power/
    ├── miku_power_rvc.pth
    └── added_IVF10688_Flat_nprobe_1_miku_power_rvc_v2.index
```

Hinweise:

- Die `.index`-Datei ist optional. Ohne sie entfällt die Index-Beimischung
  für dieses Modell automatisch.
- Die Zuordnung von `.pth` und `.index` erfolgt über den gemeinsamen
  Namensbestandteil (z. B. `miku_power_rvc`) oder darüber, dass genau eine
  `.index`-Datei im selben Ordner liegt.
- Alternativ kann ein einzelner Checkpoint auch direkt in `models/` liegen
  (Anzeigename dann = Dateiname ohne Endung); die Ordnerform ist übersichtlicher.
- Nach dem Hinzufügen neuer Modelle die Weboberfläche einmal neu starten.

## Gemeinsame Hilfsmodelle

Diese Dateien werden von allen Stimmenmodellen gemeinsam genutzt:

```text
models/rmvpe.pt
models/hubert/config.json
models/hubert/model.safetensors
```
