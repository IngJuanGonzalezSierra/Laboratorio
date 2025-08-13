# =======================================
# Laboratorio de Procesamiento Digital de Señales
# Señal + Estadísticos + Histogramas + PDF + Ruidos + SNR
# =======================================

# ====== IMPORTAR LIBRERÍAS ======
import math
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# ====== CARGAR LA SEÑAL ======
signal = wfdb.rdrecord('biomedicalsignal')  # Archivos .dat y .hea en la misma carpeta
valores = signal.p_signal.flatten()

# ====== MEDIA ======
media_manual = sum(valores) / len(valores)
media_np = np.mean(valores)

# Graficar medias
plt.figure(figsize=(10, 5))
plt.plot(valores, label='Señal')
plt.axhline(y=media_manual, color='r', linestyle='--', label='Media Manual')
plt.axhline(y=media_np, color='g', linestyle='--', label='Media Numpy')
plt.title('Medias')
plt.legend()
plt.show()

# ====== DESVIACIÓN ESTÁNDAR ======
desv_manual = math.sqrt(sum((x - media_manual) ** 2 for x in valores) / len(valores))
desv_np = np.std(valores)

# Graficar desviaciones
plt.figure(figsize=(10, 5))
plt.plot(valores, label='Señal')
plt.axhline(y=desv_manual, color='r', linestyle='--', label='Desv Std Manual')
plt.axhline(y=desv_np, color='g', linestyle='--', label='Desv Std Numpy')
plt.title('Desviaciones estándar')
plt.legend()
plt.show()

# ====== COEFICIENTE DE VARIACIÓN ======
cv_manual = (desv_manual / media_manual) * 100
cv_np = (desv_np / media_np) * 100
print(f"CV Manual: {cv_manual:.2f}% | CV Numpy: {cv_np:.2f}%")

# ====== HISTOGRAMA ======
N_bins = 20
minimo, maximo = np.min(valores), np.max(valores)
intervalos = (maximo - minimo) / N_bins
freq = np.zeros(N_bins, dtype=int)

for valor in valores:
    idx = int((valor - minimo) // intervalos)
    if 0 <= idx < N_bins:
        freq[idx] += 1

bins = np.linspace(minimo, maximo, N_bins + 1)

plt.bar(bins[:-1], freq, width=intervalos, align='edge', edgecolor='black')
plt.title('Histograma Manual')
plt.show()

plt.hist(valores, bins=N_bins, color='yellow', edgecolor='black')
plt.title('Histograma con Numpy')
plt.show()

# ====== PDF ======
pdf = norm.pdf(valores, media_np, desv_np)
plt.plot(valores, pdf, label='PDF')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.legend()
plt.show()

# ====== FUNCIÓN DE POTENCIA ======
def pot(signal):
    return np.mean(signal**2)

potSenal = pot(valores)

# ====== RUIDO GAUSSIANO ======
ruido_gauss = np.random.normal(0, 1, len(valores))
ruido_gauss_norm = ruido_gauss / np.max(np.abs(ruido_gauss)) * (np.max(valores) - np.min(valores))
senal_gauss = valores + ruido_gauss_norm
snr_gauss = 10 * np.log10(potSenal / pot(ruido_gauss_norm))

# ====== RUIDO IMPULSO ======
ruido_impulso = np.zeros(len(valores))
n_impulsos = int(0.05 * len(valores))
pos_impulsos = np.random.randint(0, len(valores), n_impulsos)
ruido_impulso[pos_impulsos] = np.max(valores) * 0.5
senal_impulso = valores + ruido_impulso
snr_impulso = 10 * np.log10(potSenal / pot(ruido_impulso))

# ====== RUIDO ARTEFACTO (onda sinusoidal) ======
frecuencia = 5  # Hz ficticia para ejemplo
t = np.arange(len(valores))
ruido_artefacto = 0.1 * np.sin(2 * np.pi * frecuencia * t / len(valores))
senal_artefacto = valores + ruido_artefacto
snr_artefacto = 10 * np.log10(potSenal / pot(ruido_artefacto))

# ====== GRAFICAR TODAS LAS SEÑALES ======
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(valores)
plt.title('Señal Original')

plt.subplot(4, 1, 2)
plt.plot(senal_gauss)
plt.title(f'Señal con Ruido Gaussiano (SNR: {snr_gauss:.2f} dB)')

plt.subplot(4, 1, 3)
plt.plot(senal_impulso)
plt.title(f'Señal con Ruido Impulso (SNR: {snr_impulso:.2f} dB)')

plt.subplot(4, 1, 4)
plt.plot(senal_artefacto)
plt.title(f'Señal con Ruido Artefacto (SNR: {snr_artefacto:.2f} dB)')

plt.tight_layout()
plt.show()
