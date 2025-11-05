import numpy as np
import matplotlib.pyplot as plt

# Parámetros del medio
pi = np.pi
epr = 1  # epsilon_r
mur = 1  # mu_r
v = 1 / np.sqrt(epr * mur)

# Discretización espacial
L = 1
Nz = 100
z = np.linspace(0, L, Nz + 1)
dz = z[2] - z[1]
loon = L / 3  # pintamos 3 longitudes de onda
k = 2 * pi / loon

# Discretización temporal
dt = 0.99 * dz / v
Nt = 2 * Nz
w = v * k
T = 2 * pi / w

# Inicialización
F = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F0 = 1  # Amplitud de la onda de excitación
F[0, 1] = F0 * np.sin(-w * 1 * dt)  # Inicializacion

# Parámetros del esquema
A = (v * dt / dz) ** 2
r = v * dt / dz

# Cálculo
for j in np.arange(1, Nt):
    F[0, j+1] = F0*np.sin(-w*j*dt)
    for i in np.arange(1, Nz):
        F[i,j + 1] = A * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A) * F[i, j] - F[i, j - 1]

    # Condición de pared absorbente
    F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz, j] )

# Solucion real
def solucion_analitica(z, t, F0, w):
    """
    Psi_analitica(z, t) = Psi_0 * sin(omega * z / v - omega * t)
    """
    return F0 * np.cos(k*z - w*t - pi/2)


# Pintar
fig, ax = plt.subplots(15, 1, sharex=True)
for it in np.arange(0, 15):
    ax[it].plot(z, F[:, it * 10], label=str(it * 10) + 'dt')
    ax[it].plot(z, solucion_analitica(z,it * 10, F0,w), label=str(it * 10) + 'dt')
    ax[it].set(xlim=(0, L), ylim=(-F0 * 1.1, F0 * 1.1))
    ax[it].legend()

fig.set_size_inches(5, 25)
plt.show()

from matplotlib.animation import FuncAnimation
def solucion_analitica(z, t, F0, w):
    """
    Psi_analitica(z, t) = Psi_0 * sin(omega * z / v - omega * t)
    """
    return F0 * np.cos(k*z - w*t - pi/2)


fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title('Propagación de Onda (FDTD vs. Analítica) en Medio Sin Pérdidas')
ax.set_xlabel('Posición z')
ax.set_ylabel('Función de Onda $\Psi$')
ax.set_ylim(-F0 * 1.1, F0 * 1.1)
ax.set_xlim(0, L)
ax.grid(True)

# Inicializar líneas
line_num, = ax.plot(z, F[:, 0], label='Numérica (FDTD)', color='blue')
line_ana, = ax.plot(z, solucion_analitica(z, 0, F0, w,), label='Analítica (Ec. 5.35)', color='red', linestyle='--')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='upper right')


# La función que actualiza el gráfico en cada frame
def update(j_idx):
    t_actual = j_idx * dt

    # Actualizar datos de la solución numérica
    line_num.set_ydata(F[j_idx, :])

    # Actualizar datos de la solución analítica
    line_ana.set_ydata(solucion_analitica(z, t_actual, F0, w))

    # Actualizar el texto del tiempo
    time_text.set_text(f'Tiempo: t={t_actual:.2f} s ({t_actual / T:.1f} T)')

    return line_num, line_ana, time_text


# Crear la animación: Mostramos un frame cada 5 pasos de tiempo para reducir el tamaño y el tiempo de renderizado.
ani = FuncAnimation(fig, update, frames=np.arange(0, Nt, 5), interval=50, blit=True)

plt.show()