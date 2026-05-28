"""Audio-similarity metrics for sample evaluation.

- STFT-L1 distance to a reference set (current paper's metric)
- Per-sample nearest-neighbor distance for memorization detection
- Frechet distance from a pluggable embedding model (CLAP integration in clap_fad.py)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _mel_or_stft_mag(wav: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude STFT for a mono signal."""
    if wav.ndim == 2:
        wav = wav.mean(axis=0)  # downmix
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    wav = np.pad(wav, pad, mode="reflect")
    n_frames = 1 + (len(wav) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n_frames, n_fft),
        strides=(wav.strides[0] * hop, wav.strides[0]),
    ).copy()
    frames *= win
    spec = np.fft.rfft(frames, axis=-1)
    return np.abs(spec).astype(np.float32)


def multires_stft_l1(wav_a: np.ndarray, wav_b: np.ndarray,
                     ffts=(512, 1024, 2048)) -> float:
    """Mean L1 distance between magnitude STFTs at multiple resolutions.
    Both inputs should be 1D mono or 2D (channels, samples).
    """
    n = min(wav_a.shape[-1], wav_b.shape[-1])
    a = wav_a[..., :n]
    b = wav_b[..., :n]
    dists = []
    for nfft in ffts:
        hop = nfft // 4
        sa = _mel_or_stft_mag(a, nfft, hop)
        sb = _mel_or_stft_mag(b, nfft, hop)
        m = min(sa.shape[0], sb.shape[0])
        dists.append(float(np.abs(sa[:m] - sb[:m]).mean()))
    return float(np.mean(dists))


def _stft_mags_batch(wavs: list[np.ndarray], n_fft: int, hop: int,
                     n_frames_cap: int) -> np.ndarray:
    """Stack STFT magnitudes for a list of audio into a (N, F, n_freq) array,
    truncated/padded to n_frames_cap frames for a common shape."""
    out = np.zeros((len(wavs), n_frames_cap, n_fft // 2 + 1), dtype=np.float32)
    for i, w in enumerate(wavs):
        s = _mel_or_stft_mag(w, n_fft, hop)  # (frames, freq)
        f = min(s.shape[0], n_frames_cap)
        out[i, :f, :] = s[:f]
    return out


def _nn_stft_distance_numpy(query_wavs, ref_wavs, ffts):
    """CPU/numpy reference implementation; slow but always works."""
    common_T = min(min(q.shape[-1] for q in query_wavs),
                   min(r.shape[-1] for r in ref_wavs))
    nQ = len(query_wavs); nR = len(ref_wavs)
    dist = np.zeros((nQ, nR), dtype=np.float32)
    for nfft in ffts:
        hop = nfft // 4
        n_frames = 1 + (common_T - nfft) // hop
        if n_frames <= 0:
            continue
        Q = _stft_mags_batch(query_wavs, nfft, hop, n_frames)
        R = _stft_mags_batch(ref_wavs,   nfft, hop, n_frames)
        for i in range(nQ):
            d_i = np.abs(Q[i][None] - R).mean(axis=(1, 2))
            dist[i] += d_i
    dist /= float(len(ffts))
    return dist


def _wavs_to_mono_tensor(wavs, device, target_T):
    """Stack a list of (T,) or (ch, T) numpy arrays into a single (N, target_T)
    float32 GPU tensor, downmixing to mono and truncating to target_T."""
    import torch
    arr = np.zeros((len(wavs), target_T), dtype=np.float32)
    for i, w in enumerate(wavs):
        if w.ndim == 2:
            w = w.mean(axis=0)
        T = min(int(w.shape[0]), target_T)
        arr[i, :T] = w[:T]
    return torch.from_numpy(arr).to(device)


def _nn_stft_distance_gpu(query_wavs, ref_wavs, ffts, device_str="cuda"):
    """torch + CUDA implementation.

    For each FFT size:
      1. torch.stft over all queries → (nQ, freq, frames).abs()
      2. same for refs → (nR, freq, frames).abs()
      3. Pairwise L1 via a Q-loop (B=1 vs nR), accumulated across FFT sizes.

    Loop over queries (not full broadcast) keeps memory bounded at the cost
    of one CUDA stream sync per query (~negligible).
    """
    import torch
    device = torch.device(device_str)
    common_T = min(min(q.shape[-1] for q in query_wavs),
                   min(r.shape[-1] for r in ref_wavs))
    Q_audio = _wavs_to_mono_tensor(query_wavs, device, common_T)
    R_audio = _wavs_to_mono_tensor(ref_wavs,   device, common_T)
    nQ = Q_audio.shape[0]; nR = R_audio.shape[0]
    dist = torch.zeros(nQ, nR, device=device, dtype=torch.float32)

    for nfft in ffts:
        hop = nfft // 4
        if common_T < nfft:
            continue
        win = torch.hann_window(nfft, device=device)
        Q = torch.stft(Q_audio, n_fft=nfft, hop_length=hop, window=win,
                       center=True, return_complex=True).abs()         # (nQ, F, frames)
        R = torch.stft(R_audio, n_fft=nfft, hop_length=hop, window=win,
                       center=True, return_complex=True).abs()         # (nR, F, frames)
        for i in range(nQ):
            # (1, F, frames) - (nR, F, frames) → (nR, F, frames), then mean → (nR,)
            d_i = (Q[i].unsqueeze(0) - R).abs().mean(dim=(1, 2))
            dist[i] += d_i
        del Q, R, win
    dist /= float(len(ffts))
    out = dist.cpu().numpy()
    del Q_audio, R_audio, dist
    torch.cuda.empty_cache()
    return out


def nn_stft_distance(query_wavs: list[np.ndarray],
                     ref_wavs: list[np.ndarray],
                     ffts=(512, 1024, 2048),
                     use_gpu: bool = True) -> dict:
    """Vectorized: precompute STFTs once per audio, then pairwise L1.
    For each query, return the closest ref by mean multires STFT-L1.

    GPU implementation by default; falls back to numpy on any error.
    """
    nQ = len(query_wavs); nR = len(ref_wavs)
    dist = None
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                dist = _nn_stft_distance_gpu(query_wavs, ref_wavs, ffts)
        except Exception as e:
            print(f"  [nn_stft_distance] GPU path failed ({e}); using numpy", flush=True)
            dist = None
    if dist is None:
        dist = _nn_stft_distance_numpy(query_wavs, ref_wavs, ffts)

    nn_i = dist.argmin(axis=1)
    nn_d = dist[np.arange(nQ), nn_i]
    return {
        "nn_distances": nn_d.tolist(),
        "nn_indices":   nn_i.tolist(),
        "mean": float(nn_d.mean()),
        "std":  float(nn_d.std()),
        "min":  float(nn_d.min()),
        "max":  float(nn_d.max()),
    }


def load_wav_dir(dir_path: Path, sr_expected: int = 44100,
                 limit: int | None = None) -> tuple[list[str], list[np.ndarray]]:
    """Load all wavs from a directory as a list of (id, audio[ch, T]) tuples."""
    files = sorted(dir_path.glob("*.wav"))
    if limit:
        files = files[:limit]
    ids = []
    wavs = []
    for f in files:
        a, sr = sf.read(str(f), dtype="float32", always_2d=True)
        if sr != sr_expected:
            raise RuntimeError(f"unexpected sr {sr} in {f}")
        wavs.append(a.T)         # (ch, T)
        ids.append(f.stem)
    return ids, wavs


def frechet_distance(mu1: np.ndarray, sig1: np.ndarray,
                     mu2: np.ndarray, sig2: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between two Gaussians with means/covariances.
    Uses Cholesky-stabilised sqrt for the cross term.
    """
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    # add small ridge for numerical stability
    sig1 = sig1 + eps * np.eye(sig1.shape[0])
    sig2 = sig2 + eps * np.eye(sig2.shape[0])
    covmean, _ = sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))
    return fid


def gaussian_stats(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = emb.mean(axis=0)
    sig = np.cov(emb, rowvar=False)
    return mu, sig
