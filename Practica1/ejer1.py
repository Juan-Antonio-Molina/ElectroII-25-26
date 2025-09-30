"""
ejer3.py - Ejercicio 3: Dipolo eléctrico

Estudie el potenical y líneas de campo de un dipolo eléctrrico formado por dos cargas
puntuales. Para ello:
- Modele un dipolo con dos cargas separadas por una distancia d
- Compare la solución numérica con la aproximación analítica del dipolo puntual
- Identifique y comente los puntos donde el campo se anula
- Analice el comportamiento asintótico del potencial a grandes distancias

Created on Fri Sept 19 22:07:51 2025
@autors: juanan, manuelpi
"""

import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga
from scipy import stats

#Definir cargas y posiciones
q1 = 1.0e-9; r1 = [0, 0.5]
q2 = -1.0e-9; r2 = [0, -0.5]
vec_cargas = [q1, q2]
vec_posiciones = [r1, r2]
colors=['blue', 'red']

# Creamos el mallado del espacio
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X,Y = np.meshgrid(x, y)

#Vectores que almacenan los valores del dipolo
V = np.zeros_like(X)
Ex = np.zeros_like(X)
Ey = np.zeros_like(X)

for i in range(len(vec_cargas)):
  Ex_i, Ey_i = campo_carga(vec_cargas[i], vec_posiciones[i], X, Y)
  Ex += Ex_i
  Ey += Ey_i
  V += potencial_carga(vec_cargas[i], vec_posiciones[i], X, Y)

# Vemos el potencial y el campo creado por las cargas
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vec_cargas)):
  ax.scatter(vec_posiciones[i][0], vec_posiciones[i][1], c=colors[i])

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
ax.set_title('Potencial y campo eléctrico de dos cargas')
ax.set_aspect('equal')
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------
# Constantes universales
pi = np.pi
E_0 = 8.8541878188e-12  # permitividad vacio
k_0 = 1/(4 * pi * E_0) # constante de Coulomb

def potencial_dipolo(p, pos, x, y):
    """
    Calcula el potencial electrico de un dipolo puntual.
    Desarrollada en la practica 1

    - p: array de dos elementos, vector momento dipolar
    - pos: array de dos elementos, posición de la carga
    - x,y: array/matrices, puntos donde evaluar el potencial
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r_cubo = (dx ** 2 + dy ** 2)**(3/2)
    r_cubo = np.where(r_cubo < 1e-10, 1e-10, r_cubo) # Evitar division por cero
    return k_0 * (p[0]*dx + p[1]*dy) / r_cubo

def campo_dipolo(p, pos, x, y):
    """
    Calcula el campo electrico de un diolo puntual..
    Desarrollada en P1

    - p: array de dos elementos, vector momento dipolar
    - pos: array de dos elementos, posición de la carga
    - x,y: array/matrices, puntos donde evaluar el potencial
    """
    dx = x - pos[0]
    dy = y - pos[1]
    r_cubo = (dx ** 2 + dy ** 2) ** (3 / 2)
    r_quinta = (dx ** 2 + dy ** 2) ** (5 / 2)
    r_cubo = np.where(r_cubo < 1e-10, 1e-10, r_cubo) # Evitar division por cero
    r_quinta = np.where(r_quinta < 1e-10, 1e-10, r_quinta) # Evitar division por cero

    Ex = k_0 * (3*dx*(p[0]*dx + p[1]*dy) / r_quinta - p[0] / r_cubo)
    Ey = k_0 * (3*dy*(p[0]*dx + p[1]*dy) / r_quinta - p[1] / r_cubo)
    return Ex, Ey

# Definimos el vector momento dipolar de nuestro problema
p = [q1*(-r2[0] + r1[0]), q1*(-r2[1] + r1[1])]

#Vectores que almacenan los valores del dipolo
V_dip = potencial_dipolo(p, [0,0], X, Y)
Ex_dip, Ey_dip = campo_dipolo(p, [0,0], X, Y)

#Visualizar el potencial
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vec_cargas)):
  ax.scatter(vec_posiciones[i][0], vec_posiciones[i][1], c=colors[i])

niveles=[-50, -10, -5, -1, 0, 1, 5, 10, 50]

# Contornos de dos cargas
cont1 = ax.contour(X, Y, V, levels=niveles, colors='indigo', alpha=0.7)
ax.clabel(cont1, inline=True, fontsize=8, fmt='%.1f')

# Contornos del dipolo
cont2 = ax.contour(X, Y, V_dip, levels=niveles, colors='deeppink', alpha=0.7)
ax.clabel(cont2, inline=True, fontsize=8, fmt='%.1f')

# Crear artistas falsos para la leyenda
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='indigo', lw=1, label='Dos cargas'),
    Line2D([0], [0], color='deeppink', lw=1, label='Dipolo'),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + legend_elements, labels + [e.get_label() for e in legend_elements])

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Comparación numérica del dipolo puntual')
ax.set_aspect('equal')
plt.show()

#Visualizar el campo eléctrico
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vec_cargas)):
  ax.scatter(vec_posiciones[i][0], vec_posiciones[i][1], c=colors[i])

# Flechas de dos cargas
streamplot1 = ax.streamplot(X, Y, Ex, Ey, color='indigo',
                          cmap='viridis', density=1.,
                          linewidth=1, arrowsize=1.2)
# Flechas del dipolo
streamplot2 = ax.streamplot(X, Y, Ex_dip, Ey_dip, color='deeppink',
                          cmap='viridis', density=1.,
                          linewidth=1, arrowsize=1.2)
# Añadir leyendas
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + legend_elements, labels + [e.get_label() for e in legend_elements])

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Comparación numérica del dipolo puntual')
ax.set_aspect('equal')
plt.show()

# Error absoluto (norma infinito)
e_abs_V = np.abs(V_dip - V)
e_abs_Ex = np.abs(Ex_dip - Ex)
e_abs_Ey = np.abs(Ey_dip - Ey)
e_rel_V = e_abs_V / np.abs(V)
e_rel_Ex = e_abs_Ex / np.abs(Ex)
e_rel_Ey = e_abs_Ey / np.abs(Ey)

err_promedio_V = np.mean(e_rel_V)
err_promedio_V_abs = np.mean(e_abs_V)
err_promedio_Ex = np.mean(e_rel_Ex)
err_promedio_Ex_abs = np.mean(e_abs_Ex)
err_promedio_Ey = np.mean(e_rel_Ey)
err_promedio_Ey_abs = np.mean(e_abs_Ey)

print('El error promedio (relativo) en V es: ', err_promedio_V)
print('El error promedio (absoluto) en V es: ', err_promedio_V_abs)
print('El error promedio (relativo) en la Ex es: ', err_promedio_Ex)
print('El error promedio (absoluto) en la Ex es: ', err_promedio_Ex_abs)
print('El error promedio (relativo) en la Ey es: ', err_promedio_Ey)
print('El error promedio (absoluto) en la Ey es: ', err_promedio_Ey_abs)


# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Usamos 'magma' que es muy visual para intensidad, y 'shading='gouraud'' para un mejor suavizado
shadow_plot = ax.pcolormesh(X, Y, magnitude,
                            cmap='magma', # Prueba con 'magma' o 'inferno'
                            shading='gouraud')
cbar = fig.colorbar(shadow_plot, ax=ax, label='Magnitud del Campo Eléctrico $||\mathbf{E}||$')
ax.streamplot(X, Y, Ex, Ey, color='gray', density=0.8, linewidth=0.5, arrowsize=0.8)

ax.set_title('Mapa de Magnitud del Campo Eléctrico')
ax.set_xlabel('Coordenada X [m]')
ax.set_ylabel('Coordenada Y [m]')
ax.set_aspect('equal')
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------

# Para el estudio del comportamiento asintotico tomamos la recta y=x
x_asin = np.linspace(0.5,5,100)
y_asin = np.linspace(0.5,5,100)
r_asin = np.sqrt(x_asin ** 2 + y_asin ** 2)
V_asin = potencial_dipolo(p, [0,0], x_asin, y_asin)

# Dibujar los puntos y el ajuste en un nuevo subplot
fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(r_asin, V_asin, label='Puntos de datos')

ax_ajuste.set_xlabel('r [m]')
ax_ajuste.set_ylabel('V [V]')
ax_ajuste.set_title('Comportamiento asintótico del potencial')
ax_ajuste.legend()
plt.show()

# Comprobemos que el comportamiento es de 1/r^2 tomando logaritmos
r_log = np.log(r_asin)
V_log = np.log(abs(V_asin))
res = stats.linregress(r_log, V_log) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")

# Dibujar los puntos y el ajuste en un nuevo subplot
y_ajuste = res.slope * r_log + res.intercept
fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(r_log, V_log, label='Puntos de datos')
ax_ajuste.plot(r_log, y_ajuste, color='red', label='Ajuste Lineal')

ax_ajuste.set_xlabel('log(r)')
ax_ajuste.set_ylabel('log(V)')
ax_ajuste.legend()
ax_ajuste.set_title('Comportamiento asintótico del potencial en logaritmos')
ax_ajuste.legend()
plt.show()
