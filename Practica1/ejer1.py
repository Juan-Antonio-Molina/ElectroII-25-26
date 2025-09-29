"""
ejer1.py - Ejercicio 1: Campo de una carga puntual

Objetivos:
- Calcule y visualice el potencial y campo elécitrco de una carga puntual
- Demuestre numéricamente que el campo cumple la ley de Coulomb, i.e., E = K 1/r^2
- Genere una gráfica del campo vs. distancia e interprete la relación (1/r)

Created on Fri Sept 19 22:07:51 2025
@autors: juanan, manuelpi
"""
import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga
from scipy import stats

#Definimos la carga puntual y su posición
q = 1.0e-9; r = [0, 0]
colors=['blue']

#Creamos una malla en el espacio
x = np.linspace(-2.5, 2.5, 50)
y = np.linspace(-2.5, 2.5, 50)
X,Y = np.meshgrid(x, y)

#Vectores que recogen la info
V = potencial_carga(q, r, X, Y)
Ex, Ey = campo_carga(q, r, X, Y)

#Visualizamos el potencial y el campo
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(r[0], r[1], c=colors[0])

# Potencial
niveles=[-50, -10, -5, -1, 0, 1, 5, 10, 50]
contours = ax.contour(X, Y, V, levels=niveles, colors='black',
                     alpha=0.7)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
ax.clabel(contours, inline=True, fontsize=8)

# Campo
magnitude = np.sqrt(Ex**2 + Ey**2)
streamplot = ax.streamplot(X, Y, Ex, Ey, color=magnitude,
                          cmap='viridis', density=1.,
                          linewidth=1, arrowsize=1.2)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Potencial y Campo Eléctrico de una carga puntual')
ax.set_aspect('equal')
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------
# Tomamos puntos de la horizontal (simetría radial)
x2 = np.linspace(0.5, 2.5, 50)
y2 = np.zeros(len(x2))
E2x, E2y = campo_carga(q, r, x2, y2)
inv_dist_cuadrado = x2 ** (-2)
res = stats.linregress(inv_dist_cuadrado, E2x) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")

# Dibujamos el ajuste
y_ajuste = res.slope * inv_dist_cuadrado + res.intercept

fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(inv_dist_cuadrado, E2x, label='Puntos de datos')
ax_ajuste.plot(inv_dist_cuadrado, y_ajuste, color='red', label='Ajuste Lineal')


ax_ajuste.set_xlabel('$1/r^2$ [$\mathrm{m}^{-2}$]')
ax_ajuste.set_ylabel('$E$ [$\mathrm{N/C}$]')
ax_ajuste.set_title('Ajuste Lineal de $E$ frente a $1/r^2$')
ax_ajuste.legend()
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------
# Dibujar los puntos y el ajuste en un nuevo subplot
fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(x2, E2x, label='Puntos de datos')
ax_ajuste.plot(x2, E2x, color='red', label='funcion')

ax_ajuste.set_xlabel('r [m]')
ax_ajuste.set_ylabel('$E$ [$\mathrm{N/C}$]')
ax_ajuste.set_title('Comparación del campo eléctrico frente a la distancia')
ax_ajuste.legend()
plt.show()