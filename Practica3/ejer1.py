"""
ejer1.py - Ejercicio 1: Carga puntual y esfera conductora a tierra

Created on Fri Sept 19 22:07:51 2025
@autors: juanan, manuelpi
"""

import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga

#Datos iniciales
a = 1.0
d = 3.0
q1 = 1.0e-9; r1 = [d, 0.0]

#Carga imagen
q2 = -q1*a/d
d2 = a**2/d
r2 = [d2, 0]
print(f"La carga imagen tiene de características q_i = {q2} y r_i = [{d2:.3f},0]")

# Las almacenamos
vec_cargas = [q1, q2]
vec_posiciones = [r1, r2]

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------

# Calculamos el potencial y el campo eléctrico del sistema
L = 4.0 # Límite del eje x/y para el gráfico
N = 100 # Número de puntos PONER PARES PORFAVOR GRACIAS

x = np.linspace(-L+1, L+1, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

V = np.zeros_like(X)
Ex = np.zeros_like(X)
Ey = np.zeros_like(X)

for i in range(len(vec_cargas)):
  Ex_i, Ey_i = campo_carga(vec_cargas[i], vec_posiciones[i], X, Y)
  Ex += Ex_i
  Ey += Ey_i
  V += potencial_carga(vec_cargas[i], vec_posiciones[i], X, Y)

### Dibujamos el potencial
fig, ax = plt.subplots(figsize=(8, 6))

# Creamos un círculo para sombrear la región de la esfera.
theta = np.linspace(0, 2*np.pi, 100)
plt.fill(a * np.cos(theta), a * np.sin(theta),
         color='gray', alpha=0.5, label='Esfera conductora ($V=0$)')
plt.plot(a * np.cos(theta), a * np.sin(theta), color='black', linewidth=1)

# Posiciones de las cargas
plt.plot(r1[0], r1[1], 'ro', markersize=6, label=f'$q_1 = {q1/1e-9:.0f}$ nC',
         markeredgecolor='black', markerfacecolor='red') # Carga real (positiva, rojo)
plt.plot(r2[0], r2[1], 'bo', markersize=6, label=f'$q_2 = {q2/1e-9:.2f}$ nC',
         markeredgecolor='black', markerfacecolor='blue') # Carga imagen (negativa, azul)

# Líneas equipotenciales
levels = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]
contours = ax.contour(X, Y, V, levels=levels, colors='darkcyan',
                     alpha=0.7)
ax.clabel(contours, inline=True, fontsize=9, fmt='%1.2f')

# Configuración del gráfico
plt.title('Líneas equipotenciales del sistema')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.gca().set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--')
plt.legend(loc='lower left')
plt.show()

### Verificar numéricmaente que es cero en la superficie
x2 = a*np.cos(theta)
y2 = a*np.sin(theta)

V_super = np.zeros_like(x2)
Ex_super = np.zeros_like(x2)
Ey_super = np.zeros_like(x2)

for i in range(len(vec_cargas)):
  Ex_i, Ey_i = campo_carga(vec_cargas[i], vec_posiciones[i], x2, y2)
  Ex_super += Ex_i
  Ey_super += Ey_i
  V_super += potencial_carga(vec_cargas[i], vec_posiciones[i], x2, y2)

abs_V_super = np.abs(V_super)
err_V_super = np.max(abs_V_super)
print(f'El error máximo cometido ha sido {err_V_super}')

# Dibujamos el error del potencial
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(theta, abs_V_super, label='Puntos de datos')
ax.plot(theta, abs_V_super, color='darkcyan')

ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'V [V]')
ax.set_title('Potencial en la superficie de la esfera')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------

# Comprobación de la condicion de E
def comp_normal(campo, x, y):
    """
    Calcula la componente normal de un campo vectorial
    cualquiera en un punto (x,y) del plano.

    - campo: campo que queremos manipular
    - x: coordenada x del punto
    - y: coordenada y del punto
    """
    # Realizaremos el producto escalar del vector campo
    # con el vector normal en el punto (x,y):
    dist = np.sqrt(x**2 + y**2)
    campo_n = campo[0] * x/dist + campo[1] * y/dist
    return campo_n

def comp_tang(campo, x, y):
    """
    Calcula la componente tangencial de un campo vectorial
    cualquiera en un punto (x,y) del plano.

    - campo: campo que queremos manipular
    - x: coordenada x del punto
    - y: coordenada y del punto
    """
    # Realizaremos el producto escalar del vector campo
    # con el vector normal en el punto (x,y):
    dist = np.sqrt(x**2 + y**2)
    campo_tg = campo[0] * (-y)/dist + campo[1] * x/dist
    return campo_tg

E_super = [Ex_super, Ey_super]
E_normal_super = comp_normal(E_super, x2, y2)
E_tan_super = comp_tang(E_super, x2, y2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(theta, E_normal_super, label = 'Normal')
ax.plot(theta, E_tan_super, label = 'Tangencial')

ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'E [N/C]')
ax.set_title('Componentes del campo eléctrico en la superficie de la esfera')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

### Dibujamos el campo
fig, ax = plt.subplots(figsize=(8, 6))

# Normalizamos la paleta de colores
from matplotlib.colors import Normalize
vmin_enfocado = 0.05
vmax_enfocado = 2
norm_personalizada = Normalize(vmin=vmin_enfocado, vmax=vmax_enfocado)

# Creamos un círculo para sombrear la región de la esfera.
plt.fill(a * np.cos(theta), a * np.sin(theta),
         color='gray', alpha=0.5, label='Esfera conductora ($V=0$)')
plt.plot(a * np.cos(theta), a * np.sin(theta), color='black', linewidth=1)

# Posiciones de las cargas
plt.plot(r1[0], r1[1], 'ro', markersize=6, label=f'$q_1 = {q1/1e-9:.0f}$ nC',
         markeredgecolor='black', markerfacecolor='red') # Carga real (positiva, rojo)
plt.plot(r2[0], r2[1], 'bo', markersize=6, label=f'$q_2 = {q2/1e-9:.2f}$ nC',
         markeredgecolor='black', markerfacecolor='blue') # Carga imagen (negativa, azul)

# Flechas
E_magn = np.sqrt(Ex**2 + Ey**2)
streamplot = ax.streamplot(X, Y, Ex, Ey, color=E_magn, cmap='jet', density=1.5,
                          norm=norm_personalizada, linewidth=1, arrowsize=1.2)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Campo eléctrico del sistema')
ax.set_aspect('equal')
fig.colorbar(streamplot.lines, label='Magnitud de E', norm=norm_personalizada)
plt.show()

### Normalizamos para ver mejor
Exn = Ex/E_magn
Eyn = Ey/E_magn

fig, ax = plt.subplots(figsize=(8, 6))

# Creamos un círculo para sombrear la región de la esfera.
plt.fill(a * np.cos(theta), a * np.sin(theta),
         color='gray', alpha=0.5, label='Esfera conductora ($V=0$)')
plt.plot(a * np.cos(theta), a * np.sin(theta), color='black', linewidth=1)

# Posiciones de las cargas
plt.plot(r1[0], r1[1], 'ro', markersize=6, label=f'$q_1 = {q1/1e-9:.0f}$ nC',
         markeredgecolor='black', markerfacecolor='red') # Carga real (positiva, rojo)
plt.plot(r2[0], r2[1], 'bo', markersize=6, label=f'$q_2 = {q2/1e-9:.2f}$ nC',
         markeredgecolor='black', markerfacecolor='blue') # Carga imagen (negativa, azul)

# Usamos 'slice' para tomar solo 1 de cada 5 puntos en X y 1 de cada 5 en Y
skip = (slice(None, None, 5), slice(None, None, 5))
quiver_plot = ax.quiver(X[skip], Y[skip], Exn[skip], Eyn[skip], E_magn[skip],
                        cmap='jet', pivot='mid', scale=40, headlength=4, norm=norm_personalizada)

ax.set_xlabel('Y [m]')
ax.set_ylabel('X [m]')
ax.set_title('Vectores de campo eléctrico normalizados')
ax.set_aspect('equal')
fig.colorbar(quiver_plot, ax=ax, label='Magnitud de E', norm=norm_personalizada)
plt.show()

# -----------------------------------------------------------------------
#                          NUEVO OBJETIVO
# -----------------------------------------------------------------------

# Calculamos la densidad superficial
e_0 = 8.8541878188e-12  # permitividad vacio
sigma = e_0*E_normal_super

# Hacemos un pequeño plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(theta, sigma, label = 'Puntos datos')
ax.plot(theta, sigma, color='darkcyan')

ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'$\sigma$ [C/$m^2$]')
ax.set_title('Densidad superficial de la esfera')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# Integramos numéricamente desde theta 0 a pi
# HE ESTADO CONSIDERANDO THETA DE 0 A 2PI PERO ESO NO ES ASI MY FOOLS
theta = theta[:50] # cambiar en funcion de la particion de theta
sigma = sigma[:50]
carga_superficie = a**2 * 2 * np.pi * (theta[1] - theta[0]) * sum(sigma * np.sin(theta))
print(f'El valor de la integración de sigma a lo largo de la superficie es {carga_superficie:}')
