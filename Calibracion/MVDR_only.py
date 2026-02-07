#!/usr/bin/env python3
# ============================================
# LOCALIZADOR DOA - BEAMFORMER MVDR (solo phi)
# ============================================

import numpy as np
from scipy.signal import butter, filtfilt, wiener
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import glob
import time
import numpy as np
from numba import jit, float64, complex128

USAR_TSHAPE = False

MIC_POSITIONS_TSHAPE = np.array([
    [0.000,  0.000],
    [0.030,  0.000],
    [-0.030, 0.000],
    [0.000,  0.025]
])

MIC_POSITIONS_ORIGINAL = np.array([
    [-0.075, 0.000],   # M1
    [-0.065, 0.000],   # M2
    [ 0.065, 0.000],   # M3
    [ 0.075, 0.000]    # M4
], dtype=np.float64)

mic_positions = MIC_POSITIONS_TSHAPE if USAR_TSHAPE else MIC_POSITIONS_ORIGINAL
GEOMETRY_NAME = "T-Shape" if USAR_TSHAPE else "Original"

c = 343.0
THETA_FIXED = 90.0
MIN_CONFIDENCE = 0.10   # ajustar según pruebas

# ============================================
# FILTRADO (mantenemos igual)
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
# STEERING VECTOR (igual que antes)
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
# MVDR – potencia de salida por dirección
# ============================================

@jit(nopython=True, cache=True)  # cache=True guarda la compilación en disco
def compute_powers(inv_R, freqs, phi_range, theta_fixed, mic_pos, c):
    """
    Calcula la suma de potencias MVDR sobre frecuencias para cada phi.
    inv_R: (n_mic, n_mic) complex
    phi_range: array 1D de ángulos en grados (float64)
    """
    n_phi = phi_range.shape[0]
    n_freq = freqs.shape[0]
    n_mic = mic_pos.shape[0]

    P = np.zeros(n_phi, dtype=float64)

    theta_rad = theta_fixed * (np.pi / 180.0)
    sin_theta = np.sin(theta_rad)

    for i_phi in range(n_phi):
        phi_rad = phi_range[i_phi] * (np.pi / 180.0)
        dir_x = sin_theta * np.cos(phi_rad)
        dir_y = sin_theta * np.sin(phi_rad)

        s = 0.0
        for i_f in range(n_freq):
            f = freqs[i_f]

            # Delays: mic_pos @ direction / c
            delays = (mic_pos[:, 0] * dir_x + mic_pos[:, 1] * dir_y) / c

            # Steering vector a
            arg = -1j * 2.0 * np.pi * f * delays
            a = np.exp(arg)  # shape (n_mic,)

            # MVDR power = 1 / Re(a^H * inv_R * a)
            temp = np.dot(inv_R, a)                # inv_R @ a
            denom = np.real(np.dot(a.conj(), temp)) # Re(a^H @ temp)

            if denom > 1e-10:
                s += 1.0 / denom

        P[i_phi] = s

    return P

# ============================================
# LOCALIZACIÓN MVDR OPTIMIZADA
# ============================================

def localize_with_mvdr(R, fs):
    # Regularización y inversión (solo una vez!)
    eps = 1e-5 * np.trace(R) / R.shape[0]   # más conservador
    R_reg = R + eps * np.eye(R.shape[0], dtype=R.dtype)
    inv_R = np.linalg.inv(R_reg).astype(np.complex128)

    # Pocas frecuencias representativas
    freqs = np.array([1500., 2200., 3000., 4000., 5000., 6500.], dtype=np.float64)

    # Búsqueda gruesa (paso 4° → ~90 evaluaciones)
    phi_coarse = np.arange(-180., 181., 4., dtype=np.float64)
    P_coarse = compute_powers(inv_R, freqs, phi_coarse, THETA_FIXED, mic_positions, c)

    if np.max(P_coarse) < 1e-8:
        return None, 0.0

    # Pico grueso
    i_max_coarse = np.argmax(P_coarse)
    phi_center = phi_coarse[i_max_coarse]

    # Refinamiento ±10° con paso 1° (~21 evaluaciones)
    phi_fine = np.arange(phi_center - 10., phi_center + 10.1, 1., dtype=np.float64)
    P_fine = compute_powers(inv_R, freqs, phi_fine, THETA_FIXED, mic_positions, c)

    # Pico final
    idx_max = np.argmax(P_fine)
    phi_est = phi_fine[idx_max]

    # Normalización para confianza
    P = P_fine / (np.max(P_fine) + 1e-10)
    peak_value = P.max()
    mean_value = P.mean()
    std_value = P.std()
    snr_like = (peak_value - mean_value) / (std_value + 1e-10)
    confidence = np.clip(snr_like / 3.5, 0., 1.)  # factor ajustable

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

    # Matriz de covarianza (igual que antes)
    R = (X @ X.T.conj()) / X.shape[1]

    phi, conf = localize_with_mvdr(R, fs)

    if phi is None or conf < MIN_CONFIDENCE:
        return None, 0.0

    return phi, conf

# ============================================
# El resto (procesar_archivo y main) se mantiene igual
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

    if len(phis) == 0:
        print(f"\nφ real = {phi_real}° → sin detecciones")
        return

    print(f"\nφ real = {phi_real}°")
    print(f"φ medio = {np.mean(phis):.1f}° ± {np.std(phis):.1f}°")
    print(f"Error = {np.mean(phis) - phi_real:.1f}°")
    print(f"Confianza media = {np.mean(confs):.3f}")


def main():
    print("="*60)
    print("LOCALIZADOR – BEAMFORMER MVDR")
    print(f"Geometría: {GEOMETRY_NAME}")
    print("="*60)

    ANGULOS = [0, 45, 30, 20, 10, -10, -20, -30, -45]

    for ang in ANGULOS:
        procesar_archivo(ang)


if __name__ == "__main__":
    main()
