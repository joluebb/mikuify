"""Compatibility helpers for the selected RVC inference package on Windows.

infer_rvc_python imports librosa and torchcrepe even when RMVPE is selected.
Those optional paths pull in Numba/llvmlite; on some Windows systems an
application-control policy blocks llvmlite.dll. This app uses RMVPE only, so
provide the small audio API surface used by the RVC package and a clear error
for the unused CREPE path instead of trying to load that DLL.
"""

from __future__ import annotations

import os
import sys
import types
import json
from importlib.machinery import ModuleSpec
from math import gcd
from pathlib import Path


def install_rvc_import_compat() -> None:
    """Install local-only import shims before importing infer_rvc_python."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Avoid loading FAISS' OpenMP DLL in the PyTorch process. Its vectors are
    # exported by pipeline.index_cache in a short-lived, FAISS-only process
    # before the selected voice model's index is used for retrieval.
    import numpy as np

    class TorchL2Index:
        """FAISS-compatible subset using exact L2 top-k on CPU PyTorch."""

        metric_type = 1
        nprobe = 1

        def __init__(self, vectors_path: Path):
            self._vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
            self.ntotal, self.d = self._vectors.shape
            self._torch_vectors = None
            self._squared_norms = None

        def reconstruct_n(self, start: int, count: int):
            return np.asarray(self._vectors[start : start + count])

        def search(self, queries, k: int):
            import torch

            if self._torch_vectors is None:
                # Keep the index on CPU to leave the 6-GB GPU available for
                # HuBERT, RMVPE and the RVC synthesizer.
                values = np.array(self._vectors, dtype=np.float32, copy=True)
                self._torch_vectors = torch.from_numpy(values)
                self._squared_norms = self._torch_vectors.square().sum(dim=1)

            query = torch.as_tensor(np.asarray(queries), dtype=torch.float32)
            k = min(int(k), int(self.ntotal))
            all_scores = []
            all_indices = []
            batch_size = 64
            for start in range(0, query.shape[0], batch_size):
                part = query[start : start + batch_size]
                distances = part @ self._torch_vectors.T
                distances.mul_(-2.0)
                distances.add_(part.square().sum(dim=1, keepdim=True))
                distances.add_(self._squared_norms.unsqueeze(0))
                distances.clamp_min_(0.0)
                scores, indices = torch.topk(
                    distances, k=k, dim=1, largest=False, sorted=True
                )
                # The RVC package converts inverse distances into weights.
                all_scores.append(scores.clamp_min_(1e-12).numpy())
                all_indices.append(indices.numpy().astype(np.int64, copy=False))
            return np.concatenate(all_scores), np.concatenate(all_indices)

    def _faiss_read_index(_path: str) -> TorchL2Index:
        cache_name = os.environ.get("MIKUIFY_FAISS_VECTORS")
        source_name = os.environ.get("MIKUIFY_FAISS_SOURCE")
        if not cache_name or not source_name:
            raise RuntimeError(
                "Der temporäre FAISS-Vektorcache fehlt; der Index des ausgewählten "
                "Stimmenmodells muss vor der RVC-Inferenz vorbereitet werden."
            )
        cache_path = Path(cache_name).resolve()
        metadata_path = cache_path.with_suffix(".json")
        if not cache_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Der temporäre FAISS-Vektorcache ist unvollständig.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if Path(metadata["source"]).resolve() != Path(source_name).resolve():
            raise RuntimeError("Der temporäre FAISS-Cache gehört nicht zum konfigurierten Index.")
        return TorchL2Index(cache_path)

    # Keep the same proxy module object because infer_rvc_python retains a
    # reference to it after its first import; refresh the per-run loader.
    faiss_proxy = sys.modules.get("faiss")
    if faiss_proxy is None:
        faiss_proxy = types.ModuleType("faiss")
        faiss_proxy.__spec__ = ModuleSpec("faiss", loader=None)
        sys.modules["faiss"] = faiss_proxy
    faiss_proxy.METRIC_L2 = 1
    faiss_proxy.read_index = _faiss_read_index

    # RVC calls librosa.resample for array input and librosa.feature.rms only
    # when envelope_ratio differs from 1.0. Keeping these implementations here
    # avoids importing librosa 0.9.1 -> numba -> llvmlite.dll on Windows.
    if "librosa" not in sys.modules:
        import numpy as np
        from scipy.signal import resample_poly

        librosa = types.ModuleType("librosa")
        librosa.__spec__ = ModuleSpec("librosa", loader=None, is_package=True)
        librosa.__path__ = []

        def resample(audio, *, orig_sr: int, target_sr: int, axis: int = -1, **_kwargs):
            if orig_sr <= 0 or target_sr <= 0:
                raise ValueError("Audio sample rates must be positive")
            divisor = gcd(int(orig_sr), int(target_sr))
            return resample_poly(
                np.asarray(audio),
                int(target_sr) // divisor,
                int(orig_sr) // divisor,
                axis=axis,
            )

        def rms(*, y, frame_length: int, hop_length: int, center: bool = True, **_kwargs):
            signal = np.asarray(y, dtype=np.float32)
            if signal.ndim != 1:
                raise ValueError("RVC envelope analysis expects mono audio")
            if center:
                pad = frame_length // 2
                signal = np.pad(signal, (pad, pad))
            if signal.size < frame_length:
                signal = np.pad(signal, (0, frame_length - signal.size))
            frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
            frames = frames[::hop_length]
            return np.sqrt(np.mean(np.square(frames), axis=-1, dtype=np.float64))[None, :]

        def mel(
            *,
            sr: int,
            n_fft: int,
            n_mels: int = 128,
            fmin: float = 0.0,
            fmax: float | None = None,
            htk: bool = False,
            norm: str | None = "slaney",
            dtype=np.float32,
        ):
            # Librosa-compatible mel triangles; RMVPE requests HTK spacing and
            # Slaney area normalization when it constructs its filterbank.
            if fmax is None:
                fmax = float(sr) / 2
            if not 0 <= fmin < fmax <= sr / 2:
                raise ValueError("Invalid mel-filter frequency limits")

            def hz_to_mel(freq):
                freq = np.asarray(freq, dtype=np.float64)
                if htk:
                    return 2595.0 * np.log10(1.0 + freq / 700.0)
                f_sp = 200.0 / 3
                mel_values = freq / f_sp
                min_log_hz = 1000.0
                min_log_mel = min_log_hz / f_sp
                logstep = np.log(6.4) / 27.0
                log_region = freq >= min_log_hz
                mel_values[log_region] = min_log_mel + np.log(
                    freq[log_region] / min_log_hz
                ) / logstep
                return mel_values

            def mel_to_hz(mel_values):
                mel_values = np.asarray(mel_values, dtype=np.float64)
                if htk:
                    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)
                f_sp = 200.0 / 3
                freqs = f_sp * mel_values
                min_log_hz = 1000.0
                min_log_mel = min_log_hz / f_sp
                logstep = np.log(6.4) / 27.0
                log_region = mel_values >= min_log_mel
                freqs[log_region] = min_log_hz * np.exp(
                    logstep * (mel_values[log_region] - min_log_mel)
                )
                return freqs

            mel_edges = np.linspace(
                hz_to_mel([fmin])[0], hz_to_mel([fmax])[0], n_mels + 2
            )
            frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            mel_frequencies = hz_to_mel(frequencies)
            ramps = mel_edges[:, None] - mel_frequencies[None, :]
            mel_deltas = np.diff(mel_edges)
            weights = np.maximum(
                0.0,
                np.minimum(
                    -ramps[:-2] / mel_deltas[:-1, None],
                    ramps[2:] / mel_deltas[1:, None],
                ),
            )
            if norm == "slaney":
                edges_hz = mel_to_hz(mel_edges)
                weights *= (2.0 / (edges_hz[2:] - edges_hz[:-2]))[:, None]
            elif norm is not None:
                raise ValueError(f"Unsupported mel normalization: {norm}")
            return weights.astype(dtype)

        feature = types.ModuleType("librosa.feature")
        feature.__spec__ = ModuleSpec("librosa.feature", loader=None)
        feature.rms = rms
        filters = types.ModuleType("librosa.filters")
        filters.__spec__ = ModuleSpec("librosa.filters", loader=None)
        filters.mel = mel
        librosa.feature = feature
        librosa.filters = filters
        librosa.resample = resample
        sys.modules["librosa"] = librosa
        sys.modules["librosa.feature"] = feature
        sys.modules["librosa.filters"] = filters

    # The app explicitly configures RMVPE. torchcrepe is imported by the
    # package unconditionally but must never be reached for pitch extraction.
    if "torchcrepe" not in sys.modules:
        torchcrepe = types.ModuleType("torchcrepe")

        def crepe_not_enabled(*_args, **_kwargs):
            raise RuntimeError("Diese Anwendung ist auf RMVPE konfiguriert; CREPE ist deaktiviert.")

        torchcrepe.predict = crepe_not_enabled
        torchcrepe.filter = types.SimpleNamespace(
            median=crepe_not_enabled,
            mean=crepe_not_enabled,
        )
        sys.modules["torchcrepe"] = torchcrepe
