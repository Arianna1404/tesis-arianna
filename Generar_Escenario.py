"""
Simulación acústica optimizada para  ARREGLO BINAURAL 4 MICRÓFONOS
Versión con RUIDO MODULADO (murmullo simulado)
"""

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample
import os

# ===============================================================
# === CONFIGURACIÓN ARREGLO BINAURAL 4 MICRÓFONOS ===
# ===============================================================

# Geometría binaural realista (metros) - 2 mic por audífono
# Convención: x positivo = derecha, y positivo = frontal
MIC_POSITIONS_BINAURAL = np.array([
    # Audífono izquierdo (L) - lado negativo x
    [-0.075,  0.005],   # M1: Izquierdo frontal
    [-0.065, -0.005],   # M2: Izquierdo trasero

    # Audífono derecho (R) - lado positivo x
    [ 0.065, -0.005],   # M3: Derecho trasero
    [ 0.075,  0.005]    # M4: Derecho frontal
], dtype=np.float64)

GEOMETRY_TYPE = "Binaural 4 mics"
mic_layout_2d = MIC_POSITIONS_BINAURAL
output_suffix = "F—Interferencia"         #Numero de Escenario Prueba

# ===============================================================
# === PARÁMETROS OPTIMIZADOS PARA CALIBRACIÓN ===
# ===============================================================

room_dim = [4.0, 3.0, 2.5]      # [ancho, largo, alto] en metros
use_rt60 = False                # Usar absorción directa
rt60_target = 0.05              # No usado si use_rt60=False
absorption = 0.2
fs = 16000
max_order = 0                   # REFLEXIONES (0 para escenarios sin reverberacion)

# === PARÁMETROS DE MOVIMIENTO (DOA) ===
# Para calibración completa, probar múltiples ángulos
ÁNGULOS_CALIBRACIÓN = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

# Configuración individual (cambiar según necesidad)
azimuth_start = 0               # Ángulo inicial azimut (grados)
azimuth_end = 0                 # Ángulo final azimut (grados)
elevation_start = 0             # SIEMPRE 0° para calibración (plano horizontal)
elevation_end = 0               # SIEMPRE 0° para calibración
distance_src = 1.2              # m de distancia
num_positions = 1               # cambios de posición durante la simulación

# ===============================================================
# === 🔉 PARÁMETROS DE RUIDO MODULADO (MURMULLO) ===
# ===============================================================
add_noise_source = False         # Activar fuente de ruido modulado
noise_azimuth = 135             # Ángulo de la fuente de ruido (grados)
noise_distance = 1.5            # Distancia de la fuente de ruido (metros)
noise_modulation_freq = 3.0     # Frecuencia de modulación (Hz) - simula ritmo del habla
noise_modulation_depth = 0.7    # Profundidad de modulación (0-1)
noise_amplitude = 0.15          # Amplitud del ruido (relativo a señal)
noise_lowpass_freq = 3000       # Frecuencia de corte para simular voz humana (Hz)

# === PARÁMETROS DE INTERFERENCIA ===
add_interference = True        # Agregar Interferencia (audio adicional)
interference_azimuth = -90      # Ángulo de la fuente de interferencia (grados)
interference_distance = 1     # Distancia de la fuente de interferencia (metros)
interference_start_time = 15.0   # Tiempo de inicio (segundos) - 0 para toda la duración
interference_audio = "/home/arianna/Audios/Interferencias/Interferencia_1.wav"

# === PARÁMETROS DE RUIDO ADICIONAL (blanco, estacionario) ===
add_noise = False               # Ruido blanco estacionario adicional
snr_target = 10                 # Si se activa, usar SNR alto
noise_type = 'white'

# === ARCHIVOS ===
input_audio = "/home/arianna/Audios/Mujer/Audio_mujer.wav"
output_prefix = f"escenario_{output_suffix}_0_mic"  # Cambiar según ángulo

# ===============================================================
# === MODO DE OPERACIÓN ===
# ===============================================================
MODO_BATCH = True  # True = genera todos los ángulos automáticamente

# ===============================================================

def generar_ruido_modulado(duracion, fs,
                          freq_modulacion=3.0,
                          profundidad=0.7,
                          freq_corte=3000):
    """
    Genera ruido blanco modulado en amplitud para simular murmullo.

    Parámetros:
    - duracion: duración en segundos
    - fs: frecuencia de muestreo
    - freq_modulacion: frecuencia de la envolvente (Hz)
    - profundidad: profundidad de modulación (0-1)
    - freq_corte: frecuencia de corte del filtro pasa-bajos (Hz)

    Returns:
    - señal de ruido modulado
    """
    num_samples = int(duracion * fs)

    # 1. Generar ruido blanco
    ruido_blanco = np.random.randn(num_samples)

    # 2. Aplicar filtro pasa-bajos para simular espectro de voz
    from scipy.signal import butter, filtfilt
    nyquist = fs / 2
    cutoff_norm = freq_corte / nyquist
    b, a = butter(4, cutoff_norm, btype='low')
    ruido_filtrado = filtfilt(b, a, ruido_blanco)

    # 3. Crear envolvente de modulación lenta
    t = np.arange(num_samples) / fs

    # Usar múltiples frecuencias para hacer la modulación más natural
    envolvente = 0
    for i, f_mult in enumerate([1.0, 1.5, 0.7, 2.3]):
        fase_aleatoria = np.random.rand() * 2 * np.pi
        amplitud = 1.0 / (i + 1)  # Armónicos decrecientes
        envolvente += amplitud * np.sin(2 * np.pi * freq_modulacion * f_mult * t + fase_aleatoria)

    # Normalizar y escalar la envolvente
    envolvente = envolvente / np.max(np.abs(envolvente))
    envolvente = (1 - profundidad) + profundidad * (envolvente + 1) / 2

    # 4. Modular el ruido filtrado con la envolvente
    ruido_modulado = ruido_filtrado * envolvente

    # 5. Normalizar
    ruido_modulado = ruido_modulado / np.max(np.abs(ruido_modulado))

    return ruido_modulado

def direction_doa(azimuth, elevation):
    """Calcula vector unitario según el DOA (azimut y elevación)."""
    az = np.radians(azimuth)
    el = np.radians(elevation)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ])

def crear_simulacion(azimuth, elevation, distance, output_name):
    """
    Crea una simulación para un ángulo específico
    """
    # Configurar absorción
    if use_rt60:
        try:
            absorption_calc, max_order_calc = pra.inverse_sabine(rt60_target, room_dim)
            max_order_calc = max_order  # Forzar sin reflexiones
            print(f"   Absorción calculada: {absorption_calc:.4f}")
        except ValueError as e:
            print(f"   ⚠️  No se puede calcular RT60, usando absorción fija")
            absorption_calc = 0.99
            max_order_calc = max_order
    else:
        absorption_calc = absorption
        max_order_calc = max_order

    # Crear la sala
    room = pra.ShoeBox(room_dim, fs=fs, absorption=absorption_calc, max_order=max_order_calc)

    # Posiciones de micrófonos (centradas en la sala)
    center_x, center_y = room_dim[0]/2, room_dim[1]/2
    mic_height = 1.5  # Altura en el centro vertical

    # Construir posiciones 3D desde layout 2D
    mic_positions = np.zeros((len(mic_layout_2d), 3))
    for i, (x, y) in enumerate(mic_layout_2d):
        mic_positions[i] = [center_x + x, center_y + y, mic_height]

    mic1_pos = mic_positions[0]
    mic_positions_absolute = mic_positions.T
    room.add_microphone_array(pra.MicrophoneArray(mic_positions_absolute, fs))

    # Cargar señal principal
    signal, input_fs = sf.read(input_audio)
    if input_fs != fs:
        signal = resample(signal, int(len(signal) * fs / input_fs))
    if signal.ndim > 1:
        signal = signal[:, 0]

    # ============================================
    # FUENTE PRINCIPAL (voz)
    # ============================================
    phi_rad = np.radians(azimuth)

    direction = np.array([
        -np.sin(phi_rad),  # X_pra = -sin(φ)
        np.cos(phi_rad),   # Y_pra = cos(φ)
        0.0
    ])

    src_pos = mic1_pos + distance * direction

    # Verificación
    delta = src_pos - mic1_pos
    az_check = np.degrees(np.arctan2(-delta[0], delta[1]))
    print(f"   ✓ Az señal: {azimuth}°, verificado: {az_check:.1f}°")

    # Verificar que la fuente esté dentro de la sala
    if not (0 < src_pos[0] < room_dim[0] and
            0 < src_pos[1] < room_dim[1] and
            0 < src_pos[2] < room_dim[2]):
        print(f"⚠️  ADVERTENCIA: Fuente fuera de la sala!")
        print(f"   Posición: {src_pos}")
        print(f"   Límites sala: {room_dim}")
        return None

    # Agregar fuente principal
    room.add_source(src_pos, signal=signal, delay=0)

    # ============================================
    # 🎵 FUENTE DE INTERFERENCIA (Audio adicional)
    # ============================================
    interference_src_pos = None
    if add_interference and interference_audio and os.path.exists(interference_audio):
        # Cargar audio de interferencia
        interference_signal, interference_fs = sf.read(interference_audio)
        if interference_fs != fs:
            interference_signal = resample(interference_signal,
                                          int(len(interference_signal) * fs / interference_fs))
        if interference_signal.ndim > 1:
            interference_signal = interference_signal[:, 0]

        # Ajustar duración
        if len(interference_signal) < len(signal):
            # Repetir si es más corto
            num_repeats = int(np.ceil(len(signal) / len(interference_signal)))
            interference_signal = np.tile(interference_signal, num_repeats)[:len(signal)]
        else:
            # Truncar si es más largo
            interference_signal = interference_signal[:len(signal)]

        # Aplicar delay si se especifica
        if interference_start_time > 0:
            delay_samples = int(interference_start_time * fs)
            if delay_samples < len(signal):
                padded_signal = np.zeros(len(signal))
                padded_signal[delay_samples:] = interference_signal[:len(signal)-delay_samples]
                interference_signal = padded_signal

        # Calcular posición de la fuente de interferencia
        phi_interference_rad = np.radians(interference_azimuth)
        direction_interference = np.array([
            -np.sin(phi_interference_rad),
            np.cos(phi_interference_rad),
            0.0
        ])
        interference_src_pos = mic1_pos + interference_distance * direction_interference

        # Verificar que esté dentro de la sala
        if not (0 < interference_src_pos[0] < room_dim[0] and
                0 < interference_src_pos[1] < room_dim[1] and
                0 < interference_src_pos[2] < room_dim[2]):
            print(f"⚠️  Fuente de interferencia fuera de la sala, ajustando posición...")
            interference_src_pos = np.clip(interference_src_pos,
                                          [0.2, 0.2, 0.5],
                                          [room_dim[0]-0.2, room_dim[1]-0.2, room_dim[2]-0.2])

        # Agregar fuente de interferencia
        room.add_source(interference_src_pos, signal=interference_signal, delay=0)

        az_interference_check = np.degrees(np.arctan2(-direction_interference[0],
                                                       direction_interference[1]))
        print(f"   🎵 Az interferencia: {interference_azimuth}°, verificado: {az_interference_check:.1f}°")
        print(f"   📂 Audio: {os.path.basename(interference_audio)}")

    # ============================================
    # 🔉 FUENTE SECUNDARIA: RUIDO MODULADO (MURMULLO)
    # ============================================
    noise_src_pos = None
    if add_noise_source:
        # Generar señal de ruido modulado
        duracion_ruido = len(signal) / fs
        ruido_modulado = generar_ruido_modulado(
            duracion_ruido,
            fs,
            freq_modulacion=noise_modulation_freq,
            profundidad=noise_modulation_depth,
            freq_corte=noise_lowpass_freq
        )

        # Escalar amplitud
        ruido_modulado = ruido_modulado * noise_amplitude

        # Ajustar longitud para que coincida con la señal
        if len(ruido_modulado) < len(signal):
            ruido_modulado = np.pad(ruido_modulado, (0, len(signal) - len(ruido_modulado)))
        else:
            ruido_modulado = ruido_modulado[:len(signal)]

        # Calcular posición de la fuente de ruido
        phi_noise_rad = np.radians(noise_azimuth)
        direction_noise = np.array([
            -np.sin(phi_noise_rad),
            np.cos(phi_noise_rad),
            0.0
        ])
        noise_src_pos = mic1_pos + noise_distance * direction_noise

        # Verificar que esté dentro de la sala
        if not (0 < noise_src_pos[0] < room_dim[0] and
                0 < noise_src_pos[1] < room_dim[1] and
                0 < noise_src_pos[2] < room_dim[2]):
            print(f"⚠️  Fuente de ruido fuera de la sala, ajustando posición...")
            noise_src_pos = np.clip(noise_src_pos,
                                   [0.2, 0.2, 0.5],
                                   [room_dim[0]-0.2, room_dim[1]-0.2, room_dim[2]-0.2])

        # Agregar fuente de ruido
        room.add_source(noise_src_pos, signal=ruido_modulado, delay=0)

        az_noise_check = np.degrees(np.arctan2(-direction_noise[0], direction_noise[1]))
        print(f"   🔉 Az ruido: {noise_azimuth}°, verificado: {az_noise_check:.1f}°")
        print(f"   📊 Amplitud ruido: {noise_amplitude:.2f}")
        print(f"   🌊 Modulación: {noise_modulation_freq}Hz, profundidad {noise_modulation_depth:.2f}")

    # Simular
    room.simulate()
    signals = np.copy(room.mic_array.signals)

    # Guardar archivos
    for i in range(signals.shape[0]):
        sf.write(f"{output_name}{i+1}.wav", signals[i], fs)

    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'distance': distance,
        'src_pos': src_pos,
        'noise_src_pos': noise_src_pos,
        'noise_azimuth': noise_azimuth if add_noise_source else None,
        'interference_src_pos': interference_src_pos,
        'interference_azimuth': interference_azimuth if add_interference else None,
        'mic1_pos': mic1_pos,
        'mic_positions': mic_positions,
        'absorption': absorption_calc,
        'max_order': max_order_calc,
        'output_name': output_name,
        'geometry': GEOMETRY_TYPE
    }

def generar_batch_calibracion():
    """
    Genera automáticamente todos los ángulos de calibración
    """
    print("="*70)
    print(f"GENERACIÓN BATCH - CALIBRACIÓN {GEOMETRY_TYPE.upper()}")
    print("="*70)
    print(f"\nGeometría de array: {GEOMETRY_TYPE}")
    print("\nPosiciones binaural (metros, relativas al centro):")
    labels = ["Izq frontal (M1)", "Izq trasero (M2)", "Der trasero (M3)", "Der frontal (M4)"]
    for i, pos in enumerate(mic_layout_2d):
        print(f"  M{i+1}: ({pos[0]:+.3f}, {pos[1]:+.3f}) - {labels[i]}")

    print(f"\nÁngulos a generar: {ÁNGULOS_CALIBRACIÓN}")
    print(f"Distancia señal: {distance_src}m")
    if add_noise_source:
        print(f"\n🔉 RUIDO MODULADO ACTIVO:")
        print(f"  • Posición: {noise_azimuth}° @ {noise_distance}m")
        print(f"  • Modulación: {noise_modulation_freq}Hz (profundidad {noise_modulation_depth})")
        print(f"  • Amplitud: {noise_amplitude}")
        print(f"  • Filtro: <{noise_lowpass_freq}Hz")
    if add_interference:
        print(f"\n🎵 INTERFERENCIA ACTIVA:")
        print(f"  • Posición: {interference_azimuth}° @ {interference_distance}m")
        print(f"  • Audio: {os.path.basename(interference_audio) if interference_audio else 'No especificado'}")
        print(f"  • Inicio: {interference_start_time}s")
    print(f"Max order (reflexiones): {max_order}")
    print("="*70)

    resultados = []

    for azimuth in ÁNGULOS_CALIBRACIÓN:
        print(f"\n🔧 Generando azimuth = {azimuth}°...")
        output_name = f"escenario_{output_suffix}_{azimuth}_mic"

        resultado = crear_simulacion(
            azimuth=azimuth,
            elevation=0,
            distance=distance_src,
            output_name=output_name
        )

        if resultado:
            resultados.append(resultado)
            print(f"   ✅ Generado: {output_name}*.wav")
        else:
            print(f"   ❌ Error generando {azimuth}°")

    visualizar_todas_posiciones(resultados)
    print_resumen(resultados)
    return resultados

def print_resumen(resultados):
    print("\n" + "="*70)
    print("📊 RESUMEN DE GENERACIÓN")
    print("="*70)
    print(f"\nGeometría: {GEOMETRY_TYPE}")
    print(f"Archivos generados: {len(resultados)}/{len(ÁNGULOS_CALIBRACIÓN)}")
    print("\nLista de archivos:")
    for r in resultados:
        print(f"  • escenario_{output_suffix}_{r['azimuth']}_mic*.wav")

    print("\n" + "="*70)
    print("📐 ANÁLISIS DE GEOMETRÍA")
    print("="*70)

    separations = []
    for i in range(len(mic_layout_2d)):
        for j in range(i + 1, len(mic_layout_2d)):
            d = np.linalg.norm(mic_layout_2d[i] - mic_layout_2d[j])
            separations.append(d * 1000)  # mm

    print(f"\nSeparaciones entre micrófonos:")
    print(f"  Mínima: {min(separations):.1f} mm")
    print(f"  Máxima: {max(separations):.1f} mm (inter-aural)")
    print(f"  Media: {np.mean(separations):.1f} mm")

    d_max = max(separations) / 1000  # metros
    d_min = min(separations) / 1000
    c = 343

    f_nyquist = c / (2 * d_max)
    f_alias = c / d_min

    print(f"\nRango de frecuencias teórico (sin aliasing):")
    print(f"  Hasta: {f_nyquist:.0f} Hz (por apertura máxima)")
    print(f"  Límite superior: {f_alias:.0f} Hz (por espaciamiento mínimo)")
    print(f"  Rango recomendado para DOA/MUSIC: 1500-6000 Hz")

    if add_noise_source:
        print(f"\n🔉 CONFIGURACIÓN DE RUIDO:")
        print(f"  Tipo: Murmullo simulado (ruido blanco modulado)")
        print(f"  Posición: {noise_azimuth}° @ {noise_distance}m")
        print(f"  Frecuencia modulación: {noise_modulation_freq}Hz")
        print(f"  Profundidad: {noise_modulation_depth}")
        print(f"  Amplitud relativa: {noise_amplitude}")

    if add_interference:
        print(f"\n🎵 CONFIGURACIÓN DE INTERFERENCIA:")
        print(f"  Posición: {interference_azimuth}° @ {interference_distance}m")
        print(f"  Audio: {os.path.basename(interference_audio) if interference_audio else 'No especificado'}")
        print(f"  Tiempo inicio: {interference_start_time}s")

def visualizar_todas_posiciones(resultados):
    if not resultados:
        return

    fig = plt.figure(figsize=(18, 7))

    # Vista superior
    ax1 = fig.add_subplot(131)
    mic_pos = resultados[0]['mic_positions']
    mic1_pos = resultados[0]['mic1_pos']

    ax1.scatter(mic_pos[:, 0], mic_pos[:, 1], c='blue', s=150,
                marker='^', edgecolors='black', linewidth=2, label='Micrófonos', zorder=5)

    for i, pos in enumerate(mic_pos):
        ax1.text(pos[0], pos[1] - 0.15, f'M{i+1}',
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.scatter([mic1_pos[0]], [mic1_pos[1]], c='purple', s=250,
                marker='*', edgecolors='black', linewidth=2, label='M1 (ref)', zorder=6)

    # Fuentes de señal (colores)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(resultados)))
    for r, color in zip(resultados, colors):
        src = r['src_pos']
        az = r['azimuth']
        ax1.scatter([src[0]], [src[1]], c=[color], s=150,
                   edgecolors='black', linewidth=1.5, zorder=4)
        ax1.plot([mic1_pos[0], src[0]], [mic1_pos[1], src[1]],
                color=color, alpha=0.5, linewidth=2, zorder=3)
        ax1.text(src[0], src[1], f' {az}°', fontsize=10,
                fontweight='bold', ha='left')

    # Fuente de ruido (si existe)
    if add_noise_source and resultados[0]['noise_src_pos'] is not None:
        noise_pos = resultados[0]['noise_src_pos']
        ax1.scatter([noise_pos[0]], [noise_pos[1]], c='red', s=200,
                   marker='s', edgecolors='black', linewidth=2,
                   label=f'🔉 Ruido ({noise_azimuth}°)', zorder=4)
        ax1.plot([mic1_pos[0], noise_pos[0]], [mic1_pos[1], noise_pos[1]],
                color='red', alpha=0.3, linewidth=2, linestyle='--', zorder=3)

    # Fuente de interferencia (si existe)
    if add_interference and resultados[0]['interference_src_pos'] is not None:
        interference_pos = resultados[0]['interference_src_pos']
        ax1.scatter([interference_pos[0]], [interference_pos[1]], c='orange', s=200,
                   marker='D', edgecolors='black', linewidth=2,
                   label=f'🎵 Interferencia ({interference_azimuth}°)', zorder=4)
        ax1.plot([mic1_pos[0], interference_pos[0]], [mic1_pos[1], interference_pos[1]],
                color='orange', alpha=0.3, linewidth=2, linestyle='-.', zorder=3)

    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), room_dim[0], room_dim[1],
                     linewidth=2, edgecolor='black',
                     facecolor='none', linestyle='--', alpha=0.5)
    ax1.add_patch(rect)

    arrow_len = distance_src * 0.5
    ax1.arrow(mic1_pos[0], mic1_pos[1], arrow_len, 0,
             head_width=0.2, head_length=0.2, fc='green', ec='green',
             linewidth=2, alpha=0.7, label='+X (Az=0°)', zorder=2)
    ax1.arrow(mic1_pos[0], mic1_pos[1], 0, arrow_len,
             head_width=0.2, head_length=0.2, fc='orange', ec='orange',
             linewidth=2, alpha=0.7, label='+Y (Az=90°)', zorder=2)

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Vista Superior - {GEOMETRY_TYPE}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Vista polar
    ax2 = plt.subplot(132, projection='polar')
    for r, color in zip(resultados, colors):
        az_rad = np.deg2rad(r['azimuth'])
        ax2.scatter([az_rad], [1], c=[color], s=200,
                   edgecolors='black', linewidth=2)
        ax2.plot([az_rad, az_rad], [0, 1], color=color,
                alpha=0.5, linewidth=3)
        ax2.text(az_rad, 1.15, f"{r['azimuth']}°",
                fontsize=10, ha='center', fontweight='bold')

    # Añadir fuente de ruido en vista polar
    if add_noise_source and resultados[0]['noise_azimuth'] is not None:
        noise_az_rad = np.deg2rad(resultados[0]['noise_azimuth'])
        ax2.scatter([noise_az_rad], [0.8], c='red', s=250,
                   marker='s', edgecolors='black', linewidth=2)
        ax2.text(noise_az_rad, 0.95, f"🔉{resultados[0]['noise_azimuth']}°",
                fontsize=10, ha='center', fontweight='bold', color='red')

    # Añadir fuente de interferencia en vista polar
    if add_interference and resultados[0]['interference_azimuth'] is not None:
        interference_az_rad = np.deg2rad(resultados[0]['interference_azimuth'])
        ax2.scatter([interference_az_rad], [0.65], c='orange', s=250,
                   marker='D', edgecolors='black', linewidth=2)
        ax2.text(interference_az_rad, 0.78, f"🎵{resultados[0]['interference_azimuth']}°",
                fontsize=10, ha='center', fontweight='bold', color='orange')

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Distribución Polar de Ángulos', fontsize=13, fontweight='bold', pad=20)

    # Zoom del array (en mm)
    ax3 = fig.add_subplot(133)
    mic_rel = (mic_pos - mic1_pos) * 1000

    ax3.scatter(mic_rel[:, 0], mic_rel[:, 1], c='blue', s=300,
               marker='^', edgecolors='black', linewidth=2)

    labels = ["Izq front", "Izq tras", "Der tras", "Der front"]
    for i, pos in enumerate(mic_rel):
        ax3.text(pos[0], pos[1] + 3, f'M{i+1}\n{labels[i]}',
                fontsize=10, ha='center', fontweight='bold')

    for i in range(len(mic_rel)):
        for j in range(i + 1, len(mic_rel)):
            ax3.plot([mic_rel[i, 0], mic_rel[j, 0]],
                    [mic_rel[i, 1], mic_rel[j, 1]],
                    'k--', alpha=0.3, linewidth=1)
            d = np.linalg.norm(mic_rel[i] - mic_rel[j])
            mid_x = (mic_rel[i, 0] + mic_rel[j, 0]) / 2
            mid_y = (mic_rel[i, 1] + mic_rel[j, 1]) / 2
            ax3.text(mid_x, mid_y, f'{d:.1f}mm',
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax3.axhline(0, color='red', linewidth=1, alpha=0.5, linestyle='-')
    ax3.axvline(0, color='green', linewidth=1, alpha=0.5, linestyle='-')
    ax3.scatter([0], [0], c='purple', s=400, marker='*',
               edgecolors='black', linewidth=2, zorder=5)

    ax3.set_xlabel('X (mm)', fontsize=12)
    ax3.set_ylabel('Y (mm)', fontsize=12)
    ax3.set_title(f'Detalle arreglo {GEOMETRY_TYPE}', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    plt.tight_layout()
    filename = f'posiciones_escenario_{output_suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: {filename}")
    plt.show()

# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"SIMULADOR CON RUIDO MODULADO - {GEOMETRY_TYPE.upper()}")
    print("="*70)
    print("\n⚙️  CONFIGURACIÓN:")
    print(f"  • Geometría: {GEOMETRY_TYPE}")
    print(f"  • Sala: {room_dim[0]}x{room_dim[1]}x{room_dim[2]}m")
    print(f"  • Absorción: {absorption if not use_rt60 else 'calculada'}")
    print(f"  • Max order: {max_order} (sin reflexiones)")
    print(f"  • Distancia señal: {distance_src}m")
    print(f"  • Frecuencia: {fs}Hz")
    print(f"  • Elevación: {elevation_start}° (plano horizontal)")

    if add_noise_source:
        print(f"\n🔉 RUIDO MODULADO (Murmullo):")
        print(f"  • Activado: Sí")
        print(f"  • Posición: {noise_azimuth}° @ {noise_distance}m")
        print(f"  • Modulación: {noise_modulation_freq}Hz")
        print(f"  • Profundidad: {noise_modulation_depth}")
        print(f"  • Amplitud: {noise_amplitude}")
        print(f"  • Filtro: <{noise_lowpass_freq}Hz")

    if add_interference:
        print(f"\n🎵 INTERFERENCIA (Audio):")
        print(f"  • Activado: Sí")
        print(f"  • Posición: {interference_azimuth}° @ {interference_distance}m")
        print(f"  • Audio: {os.path.basename(interference_audio) if interference_audio else 'No especificado'}")
        print(f"  • Inicio: {interference_start_time}s")

    if MODO_BATCH:
        print(f"\n🔄 MODO: Generación batch (todos los ángulos)")
        print("="*70)
        resultados = generar_batch_calibracion()
    else:
        print("\n🎯 MODO: Generación individual")
        print(f"  • Azimuth: {azimuth_start}°")
        print("="*70)

        print(f"\n🔧 Generando simulación para azimuth = {azimuth_start}°...")
        resultado = crear_simulacion(
            azimuth=azimuth_start,
            elevation=elevation_start,
            distance=distance_src,
            output_name=output_prefix
        )

        if resultado:
            print("\n" + "="*70)
            print("✅ SIMULACIÓN COMPLETADA")
            print("="*70)
            print(f"\nGeometría: {resultado['geometry']}")
            print(f"Posición Mic1: {resultado['mic1_pos']}")
            print(f"Posición fuente: {resultado['src_pos']}")
            if resultado['noise_src_pos'] is not None:
                print(f"Posición ruido: {resultado['noise_src_pos']}")
            if resultado['interference_src_pos'] is not None:
                print(f"Posición interferencia: {resultado['interference_src_pos']}")
            print(f"Distancia real: {np.linalg.norm(resultado['src_pos'] - resultado['mic1_pos']):.3f}m")
            print(f"Azimuth: {resultado['azimuth']}°")

            print(f"\nArchivos generados:")
            for i in range(4):
                filename = f"{output_prefix}{i+1}.wav"
                if os.path.exists(filename):
                    size_kb = os.path.getsize(filename) / 1024
                    print(f"  ✓ {filename} ({size_kb:.1f} KB)")

            visualizar_todas_posiciones([resultado])

    print("\n✅ Proceso completado")
