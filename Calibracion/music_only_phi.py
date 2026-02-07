#!/usr/bin/env python3
# ============================================
# LOCALIZADOR DOA - MUSIC PURO (SIN GRAFOS)
# ============================================

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt, wiener, find_peaks
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import glob
import time
import matplotlib.pyplot as plt

# ============================================
# GEOMETRÍA
# ============================================

USAR_TSHAPE = False

MIC_POSITIONS_TSHAPE = np.array([
    [0.000,  0.000],
    [0.030,  0.000],
    [-0.030, 0.000],
    [0.000,  0.025]
])

MIC_POSITIONS_ORIGINAL = np.array([
    # Izquierdo
    [-0.075, 0.000],   # M1: Izq frontal
    [-0.065, 0.000],   # M2: Izq trasero

    # Derecho
    [ 0.065, 0.000],   # M3: Der trasero
    [ 0.075, 0.000]    # M4: Der frontal
], dtype=np.float64)

mic_positions = MIC_POSITIONS_TSHAPE if USAR_TSHAPE else MIC_POSITIONS_ORIGINAL
GEOMETRY_NAME = "T-Shape" if USAR_TSHAPE else "Original"

c = 343.0
THETA_FIXED = 90.0
MIN_CONFIDENCE = 0.05

# ============================================
# FILTRADO
# ============================================

def apply_filters(signal, fs):
    nyq = fs / 2
    low = 500 / nyq
    high = min(7000, 0.95 * nyq) / nyq

    b, a = butter(5, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    y = wiener(y, mysize=15)
    y = y / (np.std(y) + 1e-10)
    return y

# ============================================
# STEERING VECTOR (SOLO PHI)
# ============================================

def steering_vector_phi(theta, phi, f, mic_positions):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi)
    ])

    delays = (mic_positions @ direction) / c
    return np.exp(-1j * 2 * np.pi * f * delays)

# ============================================
# MUSIC CLÁSICO
# ============================================

def music_phi_only(R, fs):
    phi_range = np.arange(-180, 180, 1.0)

    # Regularización
    eps = 1e-5 * np.linalg.norm(R, ord='fro')
    R = R + eps * np.eye(R.shape[0])

    # Autovalores
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]

    num_sources = 1
    En = eigvecs[:, num_sources:]

    # Frecuencias para MUSIC
    freqs = np.linspace(2000, 6000, 20)

    P = np.zeros(len(phi_range))

    for f in freqs:
        for i, phi in enumerate(phi_range):
            a = steering_vector_phi(THETA_FIXED, phi, f, mic_positions)
            denom = np.real(a.conj() @ En @ En.conj().T @ a)
            P[i] += 1.0 / max(denom, 1e-10)

    P /= np.max(P)

    peaks, props = find_peaks(P, height=0.3, distance=5)

    if len(peaks) == 0:
        return None, 0.0

    idx_max = peaks[np.argmax(props['peak_heights'])]
    phi_est = phi_range[idx_max]

    # Confianza simple
    snr = (P[idx_max] - np.mean(P)) / (np.std(P) + 1e-10)
    confidence = np.clip(snr / 5.0, 0, 1)

    return phi_est, confidence

# ============================================
# LOCALIZACIÓN POR VENTANA
# ============================================

def localize_window(w1, w2, w3, w4, fs):
    signals = [w1, w2, w3, w4]

    with ThreadPoolExecutor() as ex:
        sig_filt = list(ex.map(lambda x: apply_filters(x, fs), signals))

    X = np.array(sig_filt)
    X = X / (np.std(X, axis=1, keepdims=True) + 1e-10)

    # MATRIZ R CONVENCIONAL
    R = (X @ X.T.conj()) / X.shape[1]

    phi, conf = music_phi_only(R, fs)

    if phi is None or conf < MIN_CONFIDENCE:
        return None, 0.0

    return phi, conf

# ============================================
# BATCH PARA CALIBRACIÓN
# ============================================

def procesar_archivo(phi_real):
    pattern = f'/home/arianna/calibracion_*_{phi_real}_mic*.wav'
    files = sorted(glob.glob(pattern))

    if len(files) != 4:
        print("Archivos incompletos")
        return

    mic1, fs = sf.read(files[0])
    mic2, _ = sf.read(files[1])
    mic3, _ = sf.read(files[2])
    mic4, _ = sf.read(files[3])

    win = 1024
    hop = win // 2

    phis = []
    confs = []

    for i in range(0, len(mic1) - win, hop):
        w1 = mic1[i:i+win]
        w2 = mic2[i:i+win]
        w3 = mic3[i:i+win]
        w4 = mic4[i:i+win]

        phi, conf = localize_window(w1, w2, w3, w4, fs)
        if phi is not None:
            phis.append(phi)
            confs.append(conf)

    phis = np.array(phis)
    confs = np.array(confs)

    print(f"\nφ real = {phi_real}°")
    print(f"φ medio = {np.mean(phis):.1f}° ± {np.std(phis):.1f}°")
    print(f"Error = {np.mean(phis) - phi_real:.1f}°")
    print(f"Confianza media = {np.mean(confs):.3f}")

# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("MUSIC PURO – MATRIZ R CONVENCIONAL")
    print(f"Geometría: {GEOMETRY_NAME}")
    print("="*60)

    ANGULOS = [0, 45, 30, 20, 10, -10, -20, -30, -45]

    for ang in ANGULOS:
        procesar_archivo(ang)

if __name__ == "__main__":
    main()
