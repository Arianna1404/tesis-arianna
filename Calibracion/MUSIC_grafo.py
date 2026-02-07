#!/usr/bin/env python3
# ============================================
# LOCALIZADOR BINAURAL 4 MICRÓFONOS
# Versión optimizada para geometría binaural
# Compatible con AudioInt_Calibracion.py
# ============================================

import numpy as np
from scipy.linalg import eigh
import soundfile as sf
from scipy.signal import butter, filtfilt, wiener, find_peaks
import time
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import os

# ============================================
# CONFIGURACIÓN DE GEOMETRÍA BINAURAL
# ============================================

# Geometría binaural (metros) - DEBE COINCIDIR CON EL SIMULADOR
MIC_POSITIONS_BINAURAL = np.array([
    # Audífono izquierdo (L)
    [-0.075,  0.005],   # M1: Izquierdo frontal
    [-0.065, -0.005],   # M2: Izquierdo trasero

    # Audífono derecho (R)
    [ 0.065, -0.005],   # M3: Derecho trasero
    [ 0.075,  0.005]    # M4: Derecho frontal
], dtype=np.float64)

mic_positions = MIC_POSITIONS_BINAURAL
GEOMETRY_NAME = "Binaural"
FILE_SUFFIX = "binaural"

# ============================================
# PARÁMETROS
# ============================================

c = 343

GRAFO_MAX_NODES = 20
MIN_CONFIDENCE = 0.05
GRAPH_SCORE_THRESHOLD = 0.35

# Theta fijo (asume fuente en plano horizontal)
THETA_FIXED = 90.0  # grados

# ============================================
# PARÁMETROS DE CRITERIO DE TRANSICIÓN
# ============================================
DELTA_MAX = 15.0          # grados (movimiento humano realista)
ALPHA = 1.0               # peso continuidad angular
BETA = 0.6                # peso confianza MUSIC
PENALTY_JUMP = 50.0       # castigo fuerte pero finito

# Variables globales
doa_graph = []
running_score = None
estimated_doa = None
doa_history = {'phi': [], 'confidence': [], 'score': [], 'cost': []}
R_history = []
phi_prev = None
prev_phi = None

# ============================================
# FUNCIONES ANGULARES
# ============================================

def angle_diff(a, b):
    """Diferencia angular mínima en grados"""
    return ((a - b + 180) % 360) - 180

@jit(nopython=True, fastmath=True, cache=True)
def circular_blend(phi_state, phi_obs, alpha):
    """Mezcla circular entre estado previo y observación nueva"""
    a = np.deg2rad(phi_state)
    b = np.deg2rad(phi_obs)
    z = (1 - alpha) * np.exp(1j * a) + alpha * np.exp(1j * b)
    return np.rad2deg(np.angle(z))

def angular_distance(phi1, phi2):
    """Distancia angular mínima entre dos ángulos en grados"""
    return abs(((phi1 - phi2 + 180) % 360) - 180)

def wrap_angle(phi):
    """Normaliza ángulo a [-180, 180)"""
    return (phi + 180) % 360 - 180

def unwrap_to_reference(phi, phi_ref):
    """Proyecta phi a la representación equivalente más cercana a phi_ref"""
    if phi_ref is None:
        return phi
    options = [phi, phi + 360, phi - 360]
    return min(options, key=lambda x: abs(x - phi_ref))

# ============================================
# FILTRADO
# ============================================

def apply_filters_optimized(signal, fs):
    nyquist = fs / 2

    # Rango optimizado para geometría binaural (1500-6000 Hz)
    low = 1500 / nyquist
    high = min(6000, 0.95 * nyquist) / nyquist

    b_band, a_band = butter(5, [low, high], btype='band')
    signal_filt = filtfilt(b_band, a_band, signal)
    signal_filt = wiener(signal_filt, mysize=15)
    signal_filt = signal_filt / (np.std(signal_filt) + 1e-10)
    return signal_filt

# ============================================
# DETECCIÓN DE VOZ
# ============================================

def detectar_voz_activa(signal, fs, percentil=50):
    window_size = 1024
    hop_size = window_size // 2
    energias = []
    indices_ventanas = []

    for i in range(0, len(signal) - window_size, hop_size):
        ventana = signal[i:i+window_size]
        energia = np.sqrt(np.mean(ventana**2))
        energias.append(energia)
        indices_ventanas.append(i)

    threshold = np.percentile(energias, percentil)
    return np.array(indices_ventanas), np.array(energias) > threshold

# ============================================
# MUSIC OPTIMIZADO - SOLO PHI
# ============================================

@jit(nopython=True, fastmath=True, cache=True)
def fast_steering_vector_phi_only(theta_fixed, phi, f, mic_positions, c):
    # Convención pyroomacoustics: direction = [-sin(φ), cos(φ)]
    phi_rad = np.radians(phi)

    direction = np.array([
        -np.sin(phi_rad),
        np.cos(phi_rad)
    ])

    delays = (mic_positions @ direction) / c
    return np.exp(-1j * 2 * np.pi * f * delays)

@jit(nopython=True, fastmath=True, cache=True)
def fast_noise_subspace(eigenvals, eigenvecs, num_sources):
    noise_eigenvecs = eigenvecs[:, num_sources:]
    return noise_eigenvecs @ noise_eigenvecs.T.conj()

def optimized_music_phi_only(R, theta_fixed=90.0, fs=16000):
    """
    MUSIC optimizado calculando SOLO PHI (azimut)
    Optimizado para geometría binaural
    """
    # Grid angular - 1° en frontal (±90°), 2° en trasero
    phi_frontal = np.arange(-90, 90, 1.0)
    phi_trasero1 = np.arange(-180, -90, 2.0)
    phi_trasero2 = np.arange(90, 180, 2.0)
    phi_range = np.concatenate([phi_trasero1, phi_frontal, phi_trasero2])
    phi_range = np.sort(phi_range)

    # Regularización
    epsilon = 1e-5 * np.linalg.norm(R, ord='fro')
    R_reg = R + epsilon * np.eye(R.shape[0])

    eigenvals, eigenvecs = eigh(R_reg)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    num_sources = 1
    noise_subspace = fast_noise_subspace(eigenvals, eigenvecs, num_sources)

    # Frecuencias optimizadas: 2-6 kHz
    freqs = np.linspace(2000, 6000, 20)

    # Pseudo-spectrum solo para PHI
    pseudo_spectrum = np.zeros(len(phi_range))

    def process_frequency(f):
        ps = np.zeros(len(phi_range))
        for j, phi in enumerate(phi_range):
            a = fast_steering_vector_phi_only(theta_fixed, phi, f, mic_positions, c)
            denom = (a.conj() @ noise_subspace @ a).real
            ps[j] = 1 / max(denom, 1e-10)
        return ps

    # Procesar frecuencias en paralelo
    with ThreadPoolExecutor(max_workers=4) as executor:
        ps_list = list(executor.map(process_frequency, freqs))

    # Sumar sin normalizar individualmente
    pseudo_spectrum = np.sum(ps_list, axis=0)

    # Normalizar final
    ps_max_final = np.max(np.abs(pseudo_spectrum))
    if ps_max_final > 0:
        pseudo_spectrum_norm = pseudo_spectrum / ps_max_final
    else:
        pseudo_spectrum_norm = pseudo_spectrum

    # Peak detection
    peaks, properties = find_peaks(
        pseudo_spectrum_norm,
        height=0.3,
        distance=5
    )

    if len(peaks) == 0:
        return None, 0.0

    # Mejor peak
    max_peak_idx = peaks[np.argmax(properties['peak_heights'])]
    phi_estimated = phi_range[max_peak_idx]

    # Normalizar a [-180, 180)
    phi_estimated = ((phi_estimated + 180) % 360) - 180

    # Confianza mejorada
    peak_value = pseudo_spectrum_norm[max_peak_idx]
    mean_value = np.mean(pseudo_spectrum_norm)
    std_value = np.std(pseudo_spectrum_norm)

    # SNR espectral
    snr_spectral = (peak_value - mean_value) / (std_value + 1e-10)
    confidence = np.clip(snr_spectral / 5.0, 0.0, 1.0)

    return phi_estimated, confidence

# ============================================
# FILTRO DE ACEPTACIÓN CON GRAFO
# ============================================

def stabilize_phi_tolerante(phi_raw, confidence, phi_prev):
    """Filtro de aceptación basado en grafo"""
    global doa_graph, running_score

    node = {
        'phi': float(phi_raw),
        'conf': float(confidence)
    }

    doa_graph.append(node)

    if len(doa_graph) > GRAFO_MAX_NODES:
        doa_graph.pop(0)

    if len(doa_graph) == 1:
        running_score = confidence
        return phi_raw, confidence, running_score

    window = min(12, len(doa_graph))
    recent_nodes = doa_graph[-window:]

    weights = np.array([n['conf'] for n in recent_nodes])
    weights = weights / (np.sum(weights) + 1e-10)

    phis = np.array([n['phi'] for n in recent_nodes])
    confs = np.array([n['conf'] for n in recent_nodes])

    # 🔧 NUEVO: saturación angular suave
    for i in range(len(phis)):
        if abs(phis[i]) > 30 and confs[i] < 0.7:
            phis[i] = np.sign(phis[i]) * 30

    phis_rad = np.deg2rad(phis)
    phi_avg_circular = np.rad2deg(
        np.angle(np.sum(weights * np.exp(1j * phis_rad)))
    )

    angular_dev = angular_distance(phi_raw, phi_avg_circular)

    # 🔧 NUEVO: peso angular adaptativo
    phi_norm = abs(phi_raw) / 90.0        # 0 en frontal, ~0.5 en ±45°
    angular_weight = 1.0 - 0.5 * phi_norm # reduce castigo en extremos

    coherence_factor = np.exp(-angular_dev / (30.0 * angular_weight))
    current_score = confidence * coherence_factor

    if running_score is None:
        running_score = current_score
    else:
        running_score = 0.85 * running_score + 0.15 * current_score

    if running_score < GRAPH_SCORE_THRESHOLD:
        phi_final = phi_prev if phi_prev is not None else phi_raw
    else:
        phi_final = phi_raw

    return phi_final, confidence, running_score

# ============================================
# PROCESAMIENTO PRINCIPAL
# ============================================

def localize_source_optimized(mic1, mic2, mic3, mic4, fs):
    global R_history, estimated_doa, doa_history, phi_prev, prev_phi

    with ThreadPoolExecutor(max_workers=4) as executor:
        filtered = list(executor.map(
            lambda x: apply_filters_optimized(x, fs),
            [mic1, mic2, mic3, mic4]
        ))

    in_data = np.array(filtered)
    in_data_norm = in_data / (np.std(in_data, axis=1, keepdims=True) + 1e-10)
    R_new = np.dot(in_data_norm, in_data_norm.T.conj()) / in_data_norm.shape[1]

    R_history.append(R_new)
    if len(R_history) > 3:
        R_history.pop(0)

    R = np.mean(R_history, axis=0)

    # Calcular solo PHI
    phi_raw, conf = optimized_music_phi_only(R, theta_fixed=THETA_FIXED, fs=fs)

    if phi_raw is None or conf <= MIN_CONFIDENCE:
        return None, 0.0, 0.0, 0.0

    # Aplicar mapeo: invertir signo para coincidir con convención del simulador
    phi_raw_mapped = wrap_angle(-phi_raw)

    # Criterio de transición
    if prev_phi is None:
        phi_selected = phi_raw_mapped
        cost = 0.0
    else:
        dphi = abs(angle_diff(phi_raw_mapped, prev_phi))
        cost = (
            ALPHA * dphi +
            BETA * (1.0 - conf)
        )
        if dphi > DELTA_MAX:
            cost += PENALTY_JUMP
        phi_selected = phi_raw_mapped

    # Unwrapping
    phi_unwrapped = unwrap_to_reference(phi_selected, phi_prev)

    # Grafo
    phi_graph, conf_graph, score = stabilize_phi_tolerante(
        phi_unwrapped,
        conf,
        phi_prev
    )

    # NUEVO: usar salida del grafo cuando es confiable
    if score >= GRAPH_SCORE_THRESHOLD:
        phi_final = phi_graph
    elif phi_prev is not None:
        phi_final = phi_prev
    else:
        phi_final = phi_unwrapped

    # Actualizar estados
    prev_phi = phi_selected
    phi_prev = phi_final

    estimated_doa = wrap_angle(phi_final)

    # Guardar en historial
    doa_history['phi'].append(phi_final)
    doa_history['confidence'].append(conf)
    doa_history['score'].append(score)
    doa_history['cost'].append(cost)

    return estimated_doa, conf, score, cost

# ============================================
# VISUALIZACIÓN
# ============================================

def visualizar_evolucion_phi(phis, confs, scores, costs, fs, hop_size, phi_real=None):
    phis_wrapped = np.array([wrap_angle(p) for p in phis])

    n_samples = len(phis_wrapped)
    tiempo = np.arange(n_samples) * (hop_size / fs)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Evolución de PHI
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(tiempo, phis_wrapped, 'g-', linewidth=2, alpha=0.8, label='φ estimado')
    ax1.axhline(np.mean(phis_wrapped), color='r', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(phis_wrapped):.1f}°')

    if phi_real is not None:
        ax1.axhline(phi_real, color='blue', linestyle='--',
                   linewidth=2, label=f'Real: {phi_real}°')

    ax1.fill_between(tiempo,
                     np.mean(phis_wrapped) - np.std(phis_wrapped),
                     np.mean(phis_wrapped) + np.std(phis_wrapped),
                     alpha=0.2, color='red', label=f'±1σ: {np.std(phis_wrapped):.1f}°')
    ax1.set_xlabel('Tiempo (s)', fontsize=12)
    ax1.set_ylabel('φ - Azimut (grados)', fontsize=12)
    ax1.set_title(f'Evolución del Azimut - {GEOMETRY_NAME} - θ fijo = {THETA_FIXED}°',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-180, 180])

    # Confianza
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(tiempo, confs, c=confs, cmap='RdYlGn',
                         s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.plot(tiempo, confs, 'k-', linewidth=0.8, alpha=0.4)
    ax2.axhline(np.mean(confs), color='blue', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(confs):.3f}')
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Confianza', fontsize=11)
    ax2.set_title('Confianza de Estimación', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(1.0, np.max(confs)*1.1)])
    plt.colorbar(scatter, ax=ax2, label='Confianza')

    # Score
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(tiempo, scores, 'purple', linewidth=2, alpha=0.7)
    ax3.fill_between(tiempo, 0, scores, alpha=0.3, color='purple')
    ax3.axhline(np.mean(scores), color='red', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(scores):.3f}')
    ax3.set_xlabel('Tiempo (s)', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Score de Estabilidad', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Costo
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(tiempo, costs, 'orange', linewidth=2, alpha=0.7, label='Costo')
    ax4.axhline(np.mean(costs), color='red', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(costs):.2f}')
    ax4.axhline(PENALTY_JUMP, color='purple', linestyle=':',
                linewidth=1.5, alpha=0.6, label=f'Umbral salto: {PENALTY_JUMP}')
    ax4.fill_between(tiempo, 0, costs, alpha=0.3, color='orange')
    ax4.set_xlabel('Tiempo (s)', fontsize=11)
    ax4.set_ylabel('Costo', fontsize=11)
    ax4.set_title(f'Costo de Transición (α={ALPHA}, β={BETA}, Δmax={DELTA_MAX}°)',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Localización de Azimut - {GEOMETRY_NAME}',
                 fontsize=14, fontweight='bold', y=0.98)

    error_text = ""
    if phi_real is not None:
        error = angle_diff(np.mean(phis_wrapped), phi_real)
        error_text = f' | ERROR: {error:.1f}°'

    stats_text = (f'Estadísticas: φ = {np.mean(phis_wrapped):.1f}° ± {np.std(phis_wrapped):.1f}° | '
                 f'Confianza = {np.mean(confs):.3f}{error_text} | '
                 f'N = {len(phis_wrapped)} frames')
    fig.text(0.5, 0.94, stats_text, ha='center', fontsize=10, style='italic')

    filename = f'evolucion_phi_{FILE_SUFFIX}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: {filename}")
    plt.tight_layout()
    plt.show()

def visualizar_polar(phis, confs, phi_real=None):
    phis_wrapped = np.array([wrap_angle(p) for p in phis])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    phis_rad = np.deg2rad(phis_wrapped)

    scatter = ax.scatter(phis_rad, confs, c=confs, cmap='RdYlGn',
                        s=100, alpha=0.6, edgecolors='black', linewidth=1)

    mean_phi_rad = np.deg2rad(np.mean(phis_wrapped))
    ax.plot([mean_phi_rad, mean_phi_rad], [0, 1], 'r-', linewidth=3,
            label=f'Media: {np.mean(phis_wrapped):.1f}°')
    ax.plot(mean_phi_rad, np.mean(confs), 'r*', markersize=30,
            label='Posición media')

    if phi_real is not None:
        phi_real_rad = np.deg2rad(phi_real)
        ax.plot([phi_real_rad, phi_real_rad], [0, 1], 'b--', linewidth=3,
               label=f'Real: {phi_real}°')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Confianza', fontsize=11)
    ax.set_title(f'Distribución Polar - {GEOMETRY_NAME}\n(θ = {THETA_FIXED}°)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Confianza', pad=0.1)
    filename = f'polar_phi_{FILE_SUFFIX}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Gráfica polar guardada: {filename}")
    plt.tight_layout()
    plt.show()

# ============================================
# MAIN CON BATCH PROCESSING
# ============================================

def procesar_archivo(phi_real):
    """Procesa un conjunto de archivos para un ángulo dado"""
    global phi_prev, prev_phi, doa_graph, running_score, doa_history, R_history

    # Resetear variables globales
    phi_prev = None
    prev_phi = None
    doa_graph = []
    running_score = None
    doa_history = {'phi': [], 'confidence': [], 'score': [], 'cost': []}
    R_history = []

    # Buscar archivos - FORMATO COMPATIBLE CON SIMULADOR
    pattern = f'calibracion_{FILE_SUFFIX}_{phi_real}_mic*.wav'
    files = sorted(glob.glob(pattern))

    if len(files) != 4:
        print(f"⚠️  Se esperaban 4 archivos para φ={phi_real}°, encontrados: {len(files)}")
        return None

    print(f"\n{'='*70}")
    print(f"🔍 Procesando φ_real = {phi_real}°")
    print(f"{'='*70}")

    try:
        mic1, fs = sf.read(files[0])
        mic2, _ = sf.read(files[1])
        mic3, _ = sf.read(files[2])
        mic4, _ = sf.read(files[3])
        print(f"✅ Archivos cargados: {len(mic1)/fs:.1f}s @ {fs}Hz")
    except Exception as e:
        print(f"❌ Error cargando archivos: {e}")
        return None

    print("🎤 Detectando voz activa...")
    indices, mask_voz = detectar_voz_activa(mic1, fs, percentil=50)
    indices_voz = indices[mask_voz]

    window_size = 1024
    hop_size = window_size // 2

    start_time = time.time()
    processed = 0

    print("🎯 Procesando...")
    for i, start_idx in enumerate(indices_voz):
        end_idx = start_idx + window_size
        w1 = mic1[start_idx:end_idx]
        w2 = mic2[start_idx:end_idx]
        w3 = mic3[start_idx:end_idx]
        w4 = mic4[start_idx:end_idx]

        phi, conf, score, cost = localize_source_optimized(w1, w2, w3, w4, fs)
        processed += 1

    elapsed = time.time() - start_time

    if len(doa_history['phi']) > 0:
        phis = np.array(doa_history['phi'])
        confs = np.array(doa_history['confidence'])
        phis_wrapped = np.array([wrap_angle(p) for p in phis])

        error = angle_diff(np.mean(phis_wrapped), phi_real)

        print(f"\n✅ RESULTADO:")
        print(f"   φ medio: {np.mean(phis_wrapped):.1f}° ± {np.std(phis_wrapped):.1f}°")
        print(f"   φ real: {phi_real}°")
        print(f"   ERROR: {error:.1f}°")
        print(f"   Confianza: {np.mean(confs):.3f}")
        print(f"   Tiempo: {elapsed:.1f}s ({processed/elapsed:.1f} ventanas/s)")

        return {
            'phi_real': phi_real,
            'phi_mean': np.mean(phis_wrapped),
            'phi_std': np.std(phis_wrapped),
            'error': error,
            'confidence': np.mean(confs),
            'phis': phis,
            'confs': confs,
            'scores': doa_history['score'],
            'costs': doa_history['cost'],
            'fs': fs,
            'hop_size': hop_size
        }

    return None

def main():
    print("="*70)
    print(f"LOCALIZADOR {GEOMETRY_NAME.upper()}")
    print("="*70)
    print(f"Geometría: {GEOMETRY_NAME}")
    print(f"Posiciones de micrófonos (mm):")
    labels = ["Izq frontal (M1)", "Izq trasero (M2)", "Der trasero (M3)", "Der frontal (M4)"]
    for i, (pos, label) in enumerate(zip(mic_positions * 1000, labels)):
        print(f"  M{i+1}: ({pos[0]:+.0f}, {pos[1]:+.0f}) - {label}")

    # Calcular separaciones
    separations = []
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            d = np.linalg.norm(mic_positions[i] - mic_positions[j]) * 1000
            separations.append(d)

    print(f"\nSeparaciones: min={min(separations):.1f}mm, max={max(separations):.1f}mm")
    print("="*70)

    # Ángulos a probar - DEBEN COINCIDIR CON LOS DEL SIMULADOR
    ÁNGULOS = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

    resultados = []

    for angulo in ÁNGULOS:
        resultado = procesar_archivo(angulo)
        if resultado:
            resultados.append(resultado)

    # Resumen final
    if resultados:
        print("\n" + "="*70)
        print(f"📊 RESUMEN {GEOMETRY_NAME.upper()}")
        print("="*70)
        print("\nφ_real | φ_medio  | Error   | Confianza")
        print("-" * 50)
        for r in resultados:
            print(f"{r['phi_real']:+6.0f}° | {r['phi_mean']:+7.1f}° | {r['error']:+7.1f}° | {r['confidence']:.3f}")

        # Estadísticas globales
        errores = [abs(r['error']) for r in resultados]
        print("\n" + "="*70)
        print("📈 ESTADÍSTICAS GLOBALES")
        print("="*70)
        print(f"Error absoluto medio: {np.mean(errores):.1f}°")
        print(f"Error máximo: {max(errores):.1f}°")
        print(f"Error mínimo: {min(errores):.1f}°")
        print(f"Desviación estándar: {np.std(errores):.1f}°")

        if np.mean(errores) < 10:
            print("\n✅ EXCELENTE: Error medio <10°")
        elif np.mean(errores) < 20:
            print("\n✅ BUENO: Error medio <20°")
        elif np.mean(errores) < 40:
            print("\n⚠️  ACEPTABLE: Error medio <40°")
        else:
            print("\n❌ REVISAR: Error medio >40°")

        print("="*70)

        # Guardar resultados
        np.savez(f'resultados_{FILE_SUFFIX}.npz',
                resultados=resultados,
                geometry=GEOMETRY_NAME,
                mic_positions=mic_positions)
        print(f"\n✅ Resultados guardados: resultados_{FILE_SUFFIX}.npz")

        # Visualizar un caso (ángulo 0° o el primero disponible)
        caso_ejemplo = next((r for r in resultados if r['phi_real'] == 0), resultados[0])
        print(f"\n📊 Generando visualizaciones para φ={caso_ejemplo['phi_real']}°...")
        visualizar_evolucion_phi(
            caso_ejemplo['phis'],
            caso_ejemplo['confs'],
            caso_ejemplo['scores'],
            caso_ejemplo['costs'],
            caso_ejemplo['fs'],
            caso_ejemplo['hop_size'],
            phi_real=caso_ejemplo['phi_real']
        )
        visualizar_polar(
            caso_ejemplo['phis'],
            caso_ejemplo['confs'],
            phi_real=caso_ejemplo['phi_real']
        )

    else:
        print("\n⚠️  No se obtuvieron resultados válidos")

if __name__ == "__main__":
    main()
