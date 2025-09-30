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


# Calculamos el ECM usando 50 puntos sobre el conductor.
# Es decir, calculamos el potencial cuando y=0 y lo comparamos con V0:
V0 = 2 # Por ejemplo
L = 2 # Por ejemplo
Nx = 50 # Numero de puntos de la particion
NumModos = np.arange(1, Nx/2, 2) # M < N/2

# Usamos 50 puntos sin tomar las esquinas, pues estas inducirán un error
# significativo respecto al valor esperado (V0) al no haber continuidad en el
# potencial en un entorno de estas.

x = np.linspace(0.05*L, 0.95*L, Nx)
y = np.zeros_like(x)
ECMs = np.zeros_like(NumModos, dtype=float) # sino pongo float el 0.1923310 es 0
V_teorico = V0*np.ones(Nx)

for i in range(len(NumModos)):
    V_numerico = SemiCajaPotencial(NumModos[i], V0, L, x, y)
    ECMs[i] = np.mean((V_numerico - V_teorico)**2)

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
Nx = 200 # Numero de puntos de la particion
NumModos = np.arange(1, Nx/2, 2) # M < N/2
err_rel_V = 1
contador = 0

while err_rel_V > 0.01:
    if contador >= len(NumModos):
        print("No se puede alcanzar ese error relativo para la partición dada")
        break

    V_numerico = SemiCajaPotencial(NumModos[contador], V0, L, x, y)
    err_rel_V = np.max(np.abs(V_numerico/V_teorico - 1))
    contador += 1

if contador > 0:
    print("El número de modos mínimo es:", NumModos[contador])


# Para determinar el número mínimo de modos necesario para obtener
# un error aceptable, seguiremos el criterio de que el ECM sea menor
# que N

# Error aceptable cuando error relativo <1%

# Se necesitan más modos para una aproximación mejor porque la serie converge
# a la solución analítica (exacta)




