"""
ejer5.py - Ejercicio 5: Densidad de carga superficial

Objetivos:
-

Created on Sun Sept 21 14:07:23 2025
@autors: juanan, manuelpi
"""

from em2um import SemiCajaCampo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits import mplot3d # Contornos de Potencial en 3D

E_0 = 8.8541878188e-12  # permitividad vacio
V0 = 2 # Por ejemplo
L = 2 # Por ejemplo
Ny = 400 # Numero de puntos de la particion
NumModos = Ny/2 # M < N/2

# Cara Y=0
x0 = np.linspace(0.05*L, 0.95*L, Ny)
y0 = np.zeros_like(x0)
Ex0, Ey0 = SemiCajaCampo(NumModos, V0, L, x0, y0)
sigma0 = E_0*Ey0

# Cara X = 0
y_abajo = np.linspace(0.05*L,2*L,Ny)
x_abajo = np.zeros_like(y_abajo)
Ex_abajo, Ey_abajo = SemiCajaCampo(NumModos, V0, L, x_abajo, y_abajo)
sigma_abajo = E_0*Ex_abajo


# Cara X = L
y_arriba = np.linspace(0.05*L,2*L,Ny)
x_arriba = L*np.ones(Ny)
Ex_arriba, Ey_arriba = SemiCajaCampo(NumModos, V0, L, x_arriba, y_arriba)
sigma_arriba = (-1)*E_0*Ex_arriba


# Representar graficamente
# Cara Y=0
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x0, sigma0, label='Puntos de datos')
ax.plot(x0, sigma0, label='funcion')

ax.set_xlabel('X [m]')
ax.set_ylabel(r' $\sigma$ [N / C m]')
ax.set_title('Densidad de carga en la cara Y = 0')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# Cara X=0
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_abajo, sigma_abajo, label='Puntos de datos')
ax.plot(y_abajo, sigma_abajo, label='funcion')

ax.set_xlabel('Y [m]')
ax.set_ylabel(r' $\sigma$ [N / C m]')
ax.set_title('Densidad de carga en la cara X = 0')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# Cara X=L
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_arriba, sigma_arriba, label='Puntos de datos')
ax.plot(y_arriba, sigma_arriba, label='funcion')

ax.set_xlabel('Y [m]')
ax.set_ylabel(r' $\sigma$ [N / C m]')
ax.set_title('Densidad de carga en la cara X = L')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# Vamos a integrar numéricamente la densidad de carga superficial
# en cada conductor.

carga_arriba = sum((y_arriba[1] - y_arriba[0]) * sigma_arriba)
carga_abajo = sum((y_abajo[1] - y_abajo[0]) * sigma_abajo)
carga_0 = sum((x0[1] - x0[0]) * sigma0)

# Para ver que se cumple la conservación de la carga, debemos comprobar
# que la carga total es nula, como en su estado inicial:

# Para ello, fijamos una tolerancia del 5% de la carga
#  del conductor en la izquierda:
tol = 0.05 * carga_0
suma_cargas = carga_arriba + carga_abajo + carga_0

if np.abs(suma_cargas) > tol:
    print('La carga no se conserva: Q_TOTAL = ', suma_cargas)
else :
    print('La carga se conserva: Q_TOTAL = ', suma_cargas)

# Para relacionar este resultado con la geometría del sistema, observemos
# que la simetría del sistema justifica que
# el condensador de arriba tenga la misma carga que el de abajo,
# haciendo que estos dos tengan, cada uno, la mitad de carga que
# el condensador de la izquierda.

