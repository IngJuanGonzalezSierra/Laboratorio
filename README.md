# ===============================
# Laboratorio de Procesamiento Digital de Señales
# ===============================

# IMPORTAR LIBRERÍAS
import math
import wfdb
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# ===============================
# CARGA DE LA SEÑAL DESDE ARCHIVO
# Asegúrate de tener 'biomedicalsignal.dat' y 'biomedicalsignal.hea' en la misma carpeta
# ===============================
signal = wfdb.rdrecord('biomedicalsignal')  
valores = signal.p_signal.flatten()  # Convertimos a 1D

# VARIABLES INICIALES
suma = 0
cont = 0
cuadrado = 0
varianza = 0
minimo = 0
maximo = 0
N_intervalos = 0
bins = 0
N = 1024  # Número de datos

# ===============================
# MEDIA (MANUAL Y CON FUNCIÓN)
# ===============================
for valor in valores:
    suma += valor
    cont += 1
media = suma / cont
media2 = np.mean(valores)

# Graficar media
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(valores, label='Señal')
plt.axhline(y=media, color='r', linestyle='--', label='Media Manual')
plt.title('Promedio Manual')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(valores, label='Señal')
plt.axhline(y=media2, color='g', linestyle='--', label='M














