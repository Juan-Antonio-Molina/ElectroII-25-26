import numpy as np
import matplotlib.pyplot as plt

# Parámetros del medio
pi = np.pi
epr = 1  # epsilon_r
mur = 1  # mu_r
v = 1 / np.sqrt(epr * mur)

# Discretización espacial
L = 20
Nz = 300
z = np.linspace(0, L, Nz + 1)
dz = z[2] - z[1]

# Discretización temporal
dt = 0.99 * dz / v
Nt = 2 * Nz
T = 10
loon = v * T
k = 2 * pi / loon
w = v * k
T = 2 * pi / w

# Caracteristicas del medio
print(f" Características para medio con e_r = {epr :.2f} y u_r = {mur :.2f}")
print(" ")
print(f"Periodo = {T:.4f}")
print(f"Longitud de onda = {loon:.4f}")
print(f"Constante de propagacion = {k:.4f} ")
print(f"Frecuencia = {w:.4f}")
print(f"Velocidad de onda = {v:.4f}")


# Inicialización
F = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F_teo = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F0 = 1  # Amplitud de la onda de excitación
F[0, 1] = F0 * np.sin(-w * 1 * dt)  # Inicializacion

# Parámetros del esquema
A = (v * dt / dz) ** 2
r = v * dt / dz

# Cálculo
for j in np.arange(1, Nt):
    F[0, j+1] = F0*np.sin(-w*j*dt)
    F_teo[0, j + 1] = F0 * np.sin(-w * j * dt)
    for i in np.arange(1, Nz):
        F[i,j + 1] = A * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A) * F[i, j] - F[i, j - 1]
        F_teo[i, j+1] = F0 * np.cos(k * i*dz - w * j*dt - pi / 2)

    # Condición de pared absorbente
    F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz, j] )


# Bloque de animacion
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot(z, F[:, 0], lw=2)  # Inicializa la línea con el tiempo j=0
line2, = ax.plot(z, F_teo[:, 0], 'r:', lw=2)
ax.set_xlim(0, L), ax.set_ylim(-1.5, 1.5), ax.set_xlabel('z'), ax.set_ylabel('F')

for j in range(Nt + 1):
    # Actualiza los datos de la línea con la columna de tiempo 'j' de la matriz F
    line.set_ydata(F[:, j])
    line2.set_ydata(F_teo[:, j])
    ax.set_title(f"t={j * dt:.4f} s")

    # Pausa para visualizar la animación
    plt.pause(0.03)  # ~30 ms por frame

plt.show()

# Dibujamos tres periodo
fig, ax = plt.subplots(figsize=(6, 3))
j_T = int(3*T/dt)
ax.plot(z, F[:, j_T], color='darkslategray',label='Numérica')
ax.plot(z, F_teo[:, j_T], 'r:', label = "Analítica")
ax.set_xlabel(r'$z$ [m]')
ax.set_ylabel(r'$F(z,3T)$ [V/m]')
ax.set_title(f'Evolución espacial del campo eléctrico t={3*T:.2f} s')
ax.grid(True, linestyle='--')
ax.legend()
plt.show()


# Comprobacion excitacion inicial
fig, ax = plt.subplots(figsize=(6, 3))
j_landa = 0
t = np.arange(0,(Nt+1)*dt,dt)
ax.plot(t, F[j_landa, :], color='darkslategray',label='Numérica')
ax.plot(t, F_teo[j_landa, :], 'r:', label = "Analítica")
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$F(0,t)$ [V/m]')
ax.set_title(f'Evolución temporal del campo eléctrico en z={0:.2f} m')
ax.grid(True, linestyle='--')
ax.legend()
plt.show()