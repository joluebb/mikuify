"""Export an IVF-Flat FAISS index to vectors in a separate process.

On Windows, the FAISS wheel and PyTorch wheel load incompatible OpenMP DLLs if
they share one process. This small helper imports FAISS without PyTorch,
reconstructs the vectors, saves a temporary NumPy array, and exits before RVC
starts. RVC then uses the same vectors for exact L2 retrieval through PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_vectors(index_path: Path, output_path: Path) -> dict[str, int | str]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS fehlt; installiere faiss-cpu==1.10.0.") from exc

    index_path = index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"FAISS-Index nicht gefunden: {index_path}")

    index = faiss.read_index(str(index_path))
    metric_l2 = getattr(faiss, "METRIC_L2", 1)
    if int(index.metric_type) != int(metric_l2):
        raise RuntimeError(
            f"Nicht unterstützter FAISS-Metriktyp {index.metric_type}; erwartet wird L2."
        )
    vectors = index.reconstruct_n(0, index.ntotal)
    if vectors.ndim != 2 or vectors.shape[0] != index.ntotal:
        raise RuntimeError("FAISS konnte den vollständigen Index nicht rekonstruieren.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors.astype(np.float32, copy=False), allow_pickle=False)
    info = {
        "source": str(index_path),
        "vectors": int(index.ntotal),
        "dimension": int(index.d),
        "metric_type": int(index.metric_type),
        "index_type": type(index).__name__,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        info = export_vectors(args.index, args.output)
    except Exception as exc:
        parser.exit(1, f"Fehler beim Lesen des FAISS-Index: {exc}\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
