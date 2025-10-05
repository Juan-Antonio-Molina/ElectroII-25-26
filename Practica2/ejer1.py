"""
ejer1.py - Ejercicio 1: Evalue el ECM

Objetivos:
-

Created on Sun Sept 21 14:07:23 2025
@autors: juanan, manuelpi
"""

from em2um import SemiCajaPotencial
import numpy as np
import matplotlib.pyplot as plt

# Datos iniciales
V0 = 2
L = 2
Nx = 50 # Número de puntos de la particion
NumModos = np.arange(1, Nx/2, 2) # M < N/2

# Calculamos el ECM usando 50 puntos sobre el conductor.
x = np.linspace(0.05*L, 0.95*L, Nx)# Usamos 50 puntos sin tomar las esquinas
y = np.zeros_like(x)
ECMs = np.zeros_like(NumModos, dtype=float) # si no pongo float el 0.1923310 es 0
V_teorico = V0*np.ones(Nx)

# Graficamos los potenciales a lo largo de la recta y = 0
cmap = plt.cm.get_cmap('viridis') # paleta de colores jeje
colores_norm = np.linspace(0, 1, len(NumModos))
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(NumModos)):
    V_numerico = SemiCajaPotencial(NumModos[i], V0, L, x, y)
    ECMs[i] = np.mean((V_numerico - V_teorico)**2)
    ax.plot(x, V_numerico, color=cmap(colores_norm[i]), label=f'Modo {NumModos[i]}')  # label dinamico

ax.set_xlabel('x [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $x= 0$ para diferentes modos')
ax.set_xlim(0.05*L, 0.95*L)
ax.set_ylim(0, 3)
ax.legend(title='Número de Modo', loc='best')  # Mejorar la leyenda
ax.grid(True, linestyle='--')
plt.show()

# Dibujar los puntos y el ajuste en un nuevo subplot
fig_ajuste, ax_ajuste = plt.subplots(figsize=(6, 6))
ax_ajuste.scatter(NumModos, ECMs, label='Puntos de datos')
ax_ajuste.plot(NumModos, ECMs, color='red', label='Funcion')

ax_ajuste.set_xlabel('Número de modos')
ax_ajuste.set_ylabel(r'ECM [$V^2$]')
ax_ajuste.set_title('Error cuadrático medio en función del número de modos')
ax_ajuste.legend()
ax_ajuste.grid(True, linestyle='--')
plt.show()


# Encontramos el numero de modos mínimo a lo bruto
Nx = 500 # Numero de puntos de la particion
x = np.linspace(0.05*L,0.95*L,Nx)
y = np.zeros_like(x)
NumModos = np.arange(1, Nx/2, 2) # M < N/2
err_rel_V = 1
contador = 0
V_teorico = V0*np.ones(Nx)

while err_rel_V > 0.01:
    if contador >= len(NumModos):
        print("No se puede alcanzar ese error relativo para la partición dada")
        contador = 0
        break

    V_numerico = SemiCajaPotencial(NumModos[contador], V0, L, x, y)
    err_rel_V = np.max(np.abs(V_numerico/V_teorico - 1))
    contador += 1

if contador > 0:
    print("El número de modos mínimo es:", NumModos[contador])



