import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del modelo (Estimados de la imagen)
L = 1
Nz, Nt = 100, 200  # Número de puntos espaciales y temporales
er, mur = 2, 1  # Permeabilidad relativa y permeabilidad magnética relativa
w = 4 * np.pi  # Frecuencia angular, periodo de T = 2pi/w
Q = 0.1

# Características del modelo
sigma = w * er / Q # Conductividad
v = (er * mur) ** (-0.5)  # Velocidad
n = (er / mur) ** (0.5)  # Impedancia
alpha = w * (er * mur/2) ** (0.5) * ((1+Q**(-2))**(0.5) + 1)**(0.5)
beta = w * (er * mur/2) ** (0.5) * ((1+Q**(-2))**(0.5) - 1)**(0.5)
T = 2 * np.pi/w # Periodo
delta = 1/beta # Distancia de atenuación
landa = 2*np.pi/alpha #Longitud de onda
print(f" Características para Q = {Q: .2f}")
print(" ")
print(f"Constante de propagación alpha = {alpha:.4f}")
print(f"Constante beta = {beta:.4f}")
print(f"Distancia de atenuación 1/beta = {delta:.4f} ")
print(f"Longitud de onda = {landa:.4f}")
print(f"Velocidad de onda (no fase) = {v:.4f}")
print(f"Conductividad = {sigma:.4f}")
print(f"Periodo = {T:.4f}")

# Inicialización
z = np.linspace(0, L, Nz + 1)
dz = z[2] - z[1]
dt = 0.99 * dz / v  # Paso de tiempo, usando el criterio de estabilidad CFL
r = v * dt / dz

# Inicialización de la matriz F (Campo eléctrico o magnético)
F = np.zeros([Nz + 1, Nt + 1])
F_teo = np.zeros([Nz + 1, Nt + 1])

# Parámetros para el esquema de diferencias finitas (Estimados de la imagen)
A = mur * sigma * dz ** 2 / (2 * dt)
B = mur * er * (dz/dt) ** 2

# Condición de contorno en z=0 (Fuente sinusoidal)
Fo = 1
# La línea F[0,1] en la imagen está incompleta o mal formateada.
# Asumo que es el valor inicial de la fuente en el primer paso de tiempo,
# aunque típicamente la fuente se aplica en F[0, j+1].
# Basándome en la línea 32 y 33, reinterpreto la inicialización de la fuente:

F[0, 1] = Fo * np.sin(-w * 1 * dt)  # Usando 1*dt para el primer paso de tiempo
for j in np.arange(1, Nt):  # Rango de 1 a Nt para los pasos de tiempo
    F[0, j+1] = Fo * np.sin(-w * j * dt)
    F_teo[0, j+1] = Fo * np.sin(-w*j*dt)
    for i in np.arange(1, Nz):  # Itera sobre el espacio (i)
        F[i, j + 1] = 1 / (A + B) * (F[i + 1, j] + F[i - 1, j] +
                                     2 * (B - 1) * F[i, j] +
                                     (A - B) * F[i, j - 1])
        F_teo[i, j+1] = Fo*np.exp(-beta*i*dz)*np.cos(alpha*i*dz - w*j*dt - np.pi/2)

    # Condiciones de contorno en z=L (Líneas 38-40) - Asumiendo absorción (PML o simple)
    # Estas líneas están truncadas/confusas, uso una interpretación razonable:
    # F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz-1, j] ) # Línea 38 - No usada, la reinterpreto abajo
    F[Nz, j + 1] = F[Nz - 1, j] * (r - 1) / (r + 1)*(F[Nz -1, j+1] - F[Nz , j])

# --- Bloque de Visualización (Animación) ---
fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot(z, F[:, 0], lw=2)  # Inicializa la línea con el tiempo j=0
line2, = ax.plot(z, F_teo[:, 0], 'r:', lw=2)
ax.set_xlim(0, L), ax.set_ylim(-1.5, 1.5), ax.set_xlabel('z'), ax.set_ylabel('F')
for j in range(Nt + 1):
    # Actualiza los datos de la línea con la columna de tiempo 'j' de la matriz F
    line.set_ydata(F[:, j])
    line2.set_ydata(F_teo[:, j])
    ax.set_title(f"Q ={Q: .2f}  t={j * dt:.4f} s")
    plt.pause(0.03)  # ~30 ms por frame

plt.show()

# Ejercicio 8, dibujamos para t = T. Como T = j_T * dt..
fig, ax = plt.subplots(figsize=(6, 3))
j_T = int(T/dt)
ax.plot(z, F[:, j_T], color='darkslategray',label='Numérica')
ax.plot(z, F_teo[:, j_T], 'r:', label = "Analítica")
ax.set_xlabel(r'$z$ [m]')
ax.set_ylabel(r'$F(z,T)$ [V/m]')
ax.set_title(f'Evolución espacial del campo eléctrico para Q = {Q: .2f} y t={T:.2f} s')
ax.grid(True, linestyle='--')
ax.legend()
plt.show()


# Ejercicio 9, dibujamos para z = landa/2. Como landa = i_landa * dz..
fig, ax = plt.subplots(figsize=(6, 3))
j_landa = int(landa/(2*dz))
t = np.arange(0,(Nt+1)*dt,dt)
ax.plot(t, F[j_landa, :], color='darkslategray',label='Numérica')
ax.plot(t, F_teo[j_landa, :], 'r:', label = "Analítica")
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$F(\lambda/2,t)$ [V/m]')
ax.set_title(f'Evolución temporal del campo eléctrico para Q = {Q: .2f} y z={landa/2:.2f} m')
ax.grid(True, linestyle='--')
ax.legend()
plt.show()
