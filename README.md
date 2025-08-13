# =======================================
# Laboratorio de Procesamiento Digital de Señales
# =======================================

# ====== IMPORTAR LIBRERÍAS ======
import math
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# ====== CARGAR LA SEÑAL ======
# Asegúrate de tener los archivos 'biomedicalsignal.dat' y 'biomedicalsignal.hea' en la misma carpeta
signal = wfdb.rdrecord('biomedicalsignal')
valores = signal.p_signal.flatten()

# ====== VARIABLES INICIALES ======
suma = 0
cont = 0
cuadrado = 0

# ====== CALCULO DE MEDIA ======
for valor in valores:
    suma += valor
    cont += 1
media = suma / cont
media2 = np.mean(valores)

# Graficar medias
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(valores, label='Señal')
plt.axhline(y=media, color='r', linestyle='--', label='Media Manual')
plt.title('Promedio Manual')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(valores, label='Señal')
plt.axhline(y=media2, color='g', linestyle='--', label='Media Numpy')
plt.title('Promedio Numpy')
plt.legend()
plt.tight_layout()
plt.show()

# ====== DESVIACIÓN ESTÁNDAR ======
for valor in valores:
    resta = valor - media
    cuadrado += resta ** 2
varianza = cuadrado / cont
desv = math.sqrt(varianza)
desv2 = np.std(valores)

# Graficar desviaciones estándar
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(valores)
plt.axhline(y=desv, color='r', linestyle='--', label='Desv Std Manual')
plt.title('Desviación estándar Manual')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(valores)
plt.axhline(y=desv2, color='g', linestyle='--', label='Desv Std Numpy')
plt.title('Desviación estándar Numpy')
plt.legend()
plt.tight_layout()
plt.show()

# ====== COEFICIENTE DE VARIACIÓN ======
coe = (desv / media) * 100
coe2 = (desv2 / media2) * 100

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(valores)
plt.text(0.05, 0.9, f"CV Manual: {coe:.2f}%", transform=plt.gca().transAxes)
plt.title('Coeficiente de variación Manual')

plt.subplot(2, 1, 2)
plt.plot(valores)
plt.text(0.05, 0.9, f"CV Numpy: {coe2:.2f}%", transform=plt.gca().transAxes)
plt.title('Coeficiente de variación Numpy')
plt.tight_layout()
plt.show()

# ====== HISTOGRAMA ======
minimo = np.min(valores)
maximo = np.max(valores)
N_intervalos = 20
intervalos = (maximo - minimo) / N_intervalos
freq = np.zeros(N_intervalos, dtype=int)

for valor in valores:
    index = int((valor - minimo) // intervalos)
    if 0 <= index < N_intervalos:
        freq[index] += 1

bins = np.linspace(minimo, maximo, N_intervalos + 1)
plt.bar(bins[:-1], freq, width=intervalos, align='edge', edgecolor='black')
plt.title('Histograma Manual')
plt.show()

plt.hist(valores, bins=N_intervalos, color='yellow', edgecolor='black')
plt.title('Histograma con Numpy')
plt.show()

# ====== FUNCIÓN DE PROBABILIDAD ======
probabilidad = freq / (cont * intervalos)
plt.plot(bins[:-1], probabilidad, label='Función Probabilidad Manual')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.legend()
plt.show()

pdf = norm.pdf(valores, media, desv)
plt.plot(valores, pdf, label='PDF')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')
plt.legend()
plt.show()

# ====== RUIDO GAUSSIANO ======
vectores = valores
ruido_gaussiano = np.random.normal(0, 1, len(vectores))
ruido_normalizado = ruido_gaussiano / np.max(np.abs(ruido_gaussiano)) * (np.max(vectores) - np.min(vectores))
senal_ruidosa = vectores + ruido_normalizado

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(vectores)
plt.title('Señal Original')

plt.subplot(2, 1, 2)
plt.plot(senal_ruidosa)
plt.title('Señal con Ruido Gaussiano')
plt.tight_layout()
plt.show()

# ====== FUNCIÓN PARA POTENCIA Y SNR ======
def pot(signal):
    return np.mean(signal**2)

potSenal = pot(vectores)
potGaussN = pot(ruido_normalizado)
snrGaussN = 10 * np.log10(potSenal / potGaussN)

print(f"SNR Señal / Ruido Gaussiano Normalizado: {snrGaussN:.2f} dB")
