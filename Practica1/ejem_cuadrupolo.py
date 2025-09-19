"""
pl.py - código practical: cuadrupolo eléctrico
Created on Tue Feb 26 09:58:51 2025
@Author: gregomc
"""
import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga

#Definir cargas y posiciones
q1 = 1.0e-9; r1 = [-0.5, 0]
q2 = 1.0e-9; r2 = [0.5, 0]
q3 = -1.0e-9; r3 = [0, -0.5]
q4 = -1.0e-9; r4 = [0, 0.5]

cargas = [q1, q2, q3, q4]
posiciones = [r1, r2, r3, r4]
colors=['blue', 'blue', 'red', 'red']

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X,Y = np.meshgrid(x, y)

V = np.zeros_like(X)
Ex = np.zeros_like(X)
Ey = np.zeros_like(X)

for i in range(len(cargas)):
  Ex_i, Ey_i = campo_carga(cargas[i], posiciones[i], X, Y)
  Ex += Ex_i
  Ey += Ey_i
  V += potencial_carga(cargas[i], posiciones[i], X, Y)

#Visualizar
fig, ax = plt.subplots(figsize=(8, 6))

#pintar cargas
for i in range(len(cargas)):
  ax.scatter(posiciones[i][0], posiciones[i][1], c=colors[i])

#pintar potencial
niveles=[-50, -10, -5, -1, 0, 1, 5, 10, 50]

contours = ax.contour(X, Y, V, levels=niveles, colors='black',
                     alpha=0.7)
ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
ax.clabel(contours, inline=True, fontsize=8)

#pintar campo
magnitude = np.sqrt(Ex**2 + Ey**2)
streamplot = ax.streamplot(X, Y, Ex, Ey, color=magnitude,
                          cmap='viridis', density=1.,
                          linewidth=1, arrowsize=1.2)

#Aspecto
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Potencial y Campo Eléctrico de Cuatro Cargas')
ax.set_aspect('equal')

# Nos muestra la imágen
plt.show()

# Ajustes en pyhton
from scipy import stats

res = stats.linregress(x, y) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")

# Cálculo de las componentes normal y tangencial de E
radio = 1
theta = np.linspace(0, 2*np.pi, 50) # Ángulos de 0 a 2pi. 50 puntos
x = radio*np.cos(theta)
y = radio*np.sin(theta)
Ex, Ey = campo_carga(q1, [0,0], x, y)
u_rho_x = x/radio # Componente x del versor normal
u_rho_y = y/radio # Componente y del versor normal
u_phi_x = -Y/radio # Componente x del versor tangencial [9]
u_phi_y = x/radio # Componente y del versor tangencial [9]
E_rho = Ex*u_rho_x + Ey*u_rho_y # Componente normal E_rho
E_phi = Ex*u_phi_x + Ey*u_phi_y # Componente tangencial E_phi