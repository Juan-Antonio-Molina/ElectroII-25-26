import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del modelo (Estimados de la imagen)
L = 1
Nz, Nt = 100, 200  # Número de puntos espaciales y temporales
er, mur = 2, 1  # Permeabilidad relativa y permeabilidad magnética relativa
w = 4 * np.pi  # Frecuencia angular
Q = 10
sigma = w * er / Q
v = (er * mur) ** (-0.5)  # Velocidad
n = (er / mur) ** (0.5)  # Impedancia

# Inicialización
z = np.linspace(0, L, Nz + 1)
dz = z[2] - z[1]
dt = 0.99 * dz / v  # Paso de tiempo, usando el criterio de estabilidad CFL
v_dt_dz = v * dt / dz

# Inicialización de la matriz F (Campo eléctrico o magnético)
# F tiene dimensiones [Nz+1, Nt+1] para incluir los límites y el tiempo inicial
F = np.zeros([Nz + 1, Nt + 1])

# Parámetros para el esquema de diferencias finitas (Estimados de la imagen)
dz_over_dt = dz / dt
A = mur * sigma * dz ** 2 / (2 * dt)
B = mur * er * (dz_over_dt) ** 2

# Condición de contorno en z=0 (Fuente sinusoidal)
Fo = 1
# La línea F[0,1] en la imagen está incompleta o mal formateada.
# Asumo que es el valor inicial de la fuente en el primer paso de tiempo,
# aunque típicamente la fuente se aplica en F[0, j+1].
# Basándome en la línea 32 y 33, reinterpreto la inicialización de la fuente:

# F[0, 1] (Estimado, la línea 30 es confusa)
F[0, 1] = Fo * np.sin(-w * 1 * dt)  # Usando 1*dt para el primer paso de tiempo
t = np.arange(0, Nt * dt, dt)  # Vector de tiempo
# F[0, j] (Condición de contorno de la fuente)
for j in np.arange(1, Nt):  # Rango de 1 a Nt para los pasos de tiempo
    F[0, j] = Fo * np.sin(-w * j * dt)

# Esquema de diferencias finitas (Pasos de tiempo)
for j in np.arange(1, Nt):  # Itera sobre el tiempo (j)
    for i in np.arange(1, Nz):  # Itera sobre el espacio (i)
        # Esquema de Crank-Nicolson o similar (líneas 35-37)
        F[i, j + 1] = 1 / (A + B) * (F[i + 1, j] + F[i - 1, j] + \
                                     2 * (B - 1) * F[i, j] + \
                                     (A - B) * F[i, j - 1])  # Esta es una interpretación del código incompleto

    # Condiciones de contorno en z=L (Líneas 38-40) - Asumiendo absorción (PML o simple)
    # Estas líneas están truncadas/confusas, uso una interpretación razonable:
    # F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz-1, j] ) # Línea 38 - No usada, la reinterpreto abajo
    F[Nz, j + 1] = F[Nz - 1, j] * (r - 1) / (r + 1) + (r / (r + 1)) * F[
        Nz - 1, j + 1]  # Interpretación alternativa para CL

    # Línea 40
    F[Nz, j + 1] = 0  # Anulación del borde, una condición simple

# --- Bloque de Visualización (Animación) ---
fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot(z, F[:, 0], lw=2)  # Inicializa la línea con el tiempo j=0
ax.set_xlim(0, L), ax.set_ylim(-1.5, 1.5), ax.set_xlabel('z'), ax.set_ylabel('F')

for j in range(Nt + 1):
    # Actualiza los datos de la línea con la columna de tiempo 'j' de la matriz F
    line.set_ydata(F[:, j])

    # Actualiza el título con el tiempo transcurrido (j*dt)
    ax.set_title(f"t={j * dt:.4f} s")

    # Pausa para visualizar la animación
    plt.pause(0.03)  # ~30 ms por frame

plt.show()  # Muestra la figura al final si el bucle termina