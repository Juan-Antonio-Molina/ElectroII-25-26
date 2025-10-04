"""
ejer4.py - Ejercicio 4: Visualiza las líneas de campo

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

# Sección para definir los puntos que graficaremos
V0 = 2
Lx, Ly = 5, 10 # Tamaño a graficar
Nx, Ny = 100, 200 # N'umero de puntos a graficar
x=np.linspace(0, Lx, Nx)
y=np.linspace(0, Ly, Ny)
X,Y = np.meshgrid(x,y) # Matriz de puntos

# Calculamos el campo eléctrico
NumModos=50
Ex, Ey = SemiCajaCampo(NumModos, V0, Lx, X, Y)

# Dibujamos
# Normalizamos la paleta de colores
vmin_enfocado = 0.05
vmax_enfocado = 0.25
norm_personalizada = Normalize(vmin=vmin_enfocado, vmax=vmax_enfocado)

fig, ax=plt.subplots()
Norm = (Ex**2 + Ey**2)**(.5) # norma de E
strm = ax.streamplot(X, Y, Ex, Ey,
                     color=Norm,
                     cmap='viridis',
                     norm=norm_personalizada,
                     density=1.5,
                     linewidth=1)

# Nota: La barra de color también debe usar esta normalización para ser coherente.
ax.set_ylabel('Y [m]')
ax.set_xlabel('X [m]')
ax.set_title('Líneas de Campo Eléctrico E')
fig.colorbar(strm.lines, label='Magnitud de E', norm=norm_personalizada)

# Para usar "quiver" hay que normalizar
Exn = Ex/Norm
Eyn = Ey/Norm
fig, ax=plt.subplots(figsize=(8, 6))

# Usamos 'slice' para tomar solo 1 de cada 5 puntos en X y 1 de cada 5 en Y
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(Y[skip], X[skip], Eyn[skip], Exn[skip],
          Norm[skip],
          cmap='jet',
          pivot='mid',
          scale=40,
          headlength=4)

# Etiquetas y título
ax.set_xlabel('Y [m]')
ax.set_ylabel('X [m]')
ax.set_title('Vectores de Campo Eléctrico Normalizados')
plt.show()

