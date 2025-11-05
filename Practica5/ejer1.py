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
loon = L / 3  # pintamos 3 longitudes de onda
k = 2 * pi / loon

# Discretización temporal
dt = 0.99 * dz / v
Nt = 2 * Nz
w = v * k
T = 2 * pi / w

# Inicialización
F = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F_teo = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F0 = 1  # Amplitud de la onda de excitación
F[0, 1] = F0 * np.sin(-w * 1 * dt)  # Inicializacion

# Parámetros del esquema
A = (v * dt / dz) ** 2
r = v * dt / dz

# Solucion real
def solucion_analitica(z, t):
    """
    Psi_analitica(z, t) = Psi_0 * sin(omega * z / v - omega * t)
    """
    return F0 * np.cos(k*z - w*t - pi/2)

# Cálculo
for j in np.arange(1, Nt):
    F[0, j+1] = F0*np.sin(-w*j*dt)
    for i in np.arange(1, Nz):
        F[i,j + 1] = A * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A) * F[i, j] - F[i, j - 1]
        F_teo[i, j+1] = solucion_analitica(i*dz, j*dt)

    # Condición de pared absorbente
    F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz, j] )


from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot(z, F[:, 0], lw=2)  # Inicializa la línea con el tiempo j=0
line2, = ax.plot(z, F_teo[:, 0], 'r:', lw=2)
ax.set_xlim(0, L), ax.set_ylim(-1.5, 1.5), ax.set_xlabel('z'), ax.set_ylabel('F')

for j in range(Nt + 1):
    # Actualiza los datos de la línea con la columna de tiempo 'j' de la matriz F
    line.set_ydata(F[:, j])
    line2.set_ydata(F_teo[:, j])

    # Actualiza el título con el tiempo transcurrido (j*dt)
    ax.set_title(f"t={j * dt:.4f} s")

    # Pausa para visualizar la animación
    plt.pause(0.03)  # ~30 ms por frame

plt.show()  # Muestra
# OJALA LAS ARDILLAS SEAN FEMBOYS