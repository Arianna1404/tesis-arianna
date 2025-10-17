"""
Simulación acústica con Pyroomacoustics
Versión corregida: genera audio válido en los micrófonos.
"""

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample
import colorednoise  # pip install colorednoise

# ===============================================================
# === PARÁMETROS DE SIMULACIÓN ===
# ===============================================================
room_dim = [6.0, 4.0, 2.5]      # [ancho, largo, alto] en metros
use_rt60 = True                # Si True, calcula absorción a partir de RT60
rt60_target = 0.3               # Segundos (solo si use_rt60=True)
absorption = 0.2                # Coeficiente de absorción si no se usa RT60
fs = 16000                      # Frecuencia de muestreo
max_order = 3                   # Número máximo de reflexiones

azimuth = 45                    # Ángulo horizontal en grados
elevation = 0                   # Ángulo vertical en grados
distance_src = 0.5              # Distancia fuente–Mic1 (m)

add_noise = False                # Activar ruido
snr_target = 30                 # Nivel de SNR (dB)
noise_type = 'white'             # 'white', 'pink', 'brown'

input_audio = "/home/arianna/Audios/Mujer/Audio_mujer.wav"
output_prefix = "Nivel1_Mujer_mic"  # Prefijo de salida
# ===============================================================


def direction_doa(azimuth, elevation):
    """Calcula vector unitario según el DOA (azimut y elevación)."""
    az = np.radians(azimuth)
    el = np.radians(elevation)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])


# === Calcular absorción si se define RT60 ===
if use_rt60:
    absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
    print(f"[INFO] Absorción calculada para RT60={rt60_target}s: {absorption:.3f}")

# === Crear la sala ===
room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption, max_order=max_order)

# === Posiciones de micrófonos ===
mic1_pos = np.array([2.100, 0.500, 1.400])
mic_positions = np.array([
    [2.100, 0.500, 1.400],
    [2.120, 0.510, 1.400],
    [2.140, 0.500, 1.400],
    [2.160, 0.510, 1.400],
])
mic_positions_absolute = mic_positions.T  # (3 x N)
room.add_microphone_array(pra.MicrophoneArray(mic_positions_absolute, fs))

# === Definir fuente según DOA ===
doa_vector = direction_doa(azimuth, elevation)
src_pos = mic1_pos + distance_src * doa_vector

# === Cargar señal ===
signal, input_fs = sf.read(input_audio)
if input_fs != fs:
    signal = resample(signal, int(len(signal) * fs / input_fs))
if signal.ndim > 1:
    signal = signal[:, 0]

# === Agregar fuente CON señal desde el inicio ===
room.add_source(src_pos, signal=signal)

# === Simular propagación ===
room.simulate()
signals = np.copy(room.mic_array.signals)

# === Agregar ruido si está activado ===
if add_noise:
    n_samples = signals.shape[1]
    if noise_type == 'white':
        noise = np.random.randn(n_samples)
    elif noise_type == 'pink':
        noise = colorednoise.powerlaw_psd_gaussian(1, n_samples)
    elif noise_type == 'brown':
        noise = colorednoise.powerlaw_psd_gaussian(2, n_samples)
    else:
        raise ValueError("Tipo de ruido no válido. Usa 'white', 'pink' o 'brown'.")

    noise = noise / np.std(noise)  # Normaliza energía

    for i in range(signals.shape[0]):
        sig_power = np.mean(signals[i] ** 2)
        noise_power = np.mean(noise ** 2)
        k = np.sqrt(sig_power / (10 ** (snr_target / 10) * noise_power))
        signals[i] += k * noise

# === Guardar archivos ===
for i in range(signals.shape[0]):
    sf.write(f"{output_prefix}{i+1}.wav", signals[i], fs)

# === Visualizar sala ===
room.plot()
plt.title("Simulación acústica con parámetros configurables")
plt.show()

# === Mostrar resumen ===
print("\n=== RESUMEN DE LA SIMULACIÓN ===")
print(f"Dimensiones de la sala: {room_dim} m")
print(f"Coef. de absorción: {absorption:.3f}")
print(f"DOA: azimuth={azimuth}°, elevation={elevation}°")
print(f"Distancia fuente-mic1: {distance_src} m")
print(f"Ruido activado: {add_noise}, tipo: {noise_type}, SNR={snr_target} dB")
print(f"Fuente en posición: {src_pos}")
print("Archivos generados:")
for i in range(signals.shape[0]):
    print(f"  -> {output_prefix}{i+1}.wav")
