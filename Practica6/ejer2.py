import numpy as np
import matplotlib.pyplot as plt

# Constantes del problema
pi=np.pi
L=3                              #espacio a discretizar
Nz=150                           #N=numero de ptos
w=2*pi                           #frec. angular excitacion
A=1                              #Amplitud del campo
mur = 1
T = 2*np.pi/w
ratio=.99

# Distribución espacial
z = np.linspace(0, L, Nz+1)
dz = z[1]-z[0]

# --- Definición de la interfaz ---
L_int = L / 2  # La interfaz se ubica a la mitad del dominio
i_int = int(L_int / dz)  # Índice donde comienza el dieléctrico (i_int + 1)
print(f"La interfaz se ubica en el índice i={i_int}, z={z[i_int]:.2f}")

# --- Definición de las propiedades del material ---
epr1 = 1 # Región 1 (0 < z <= L_int)
epr2 = 4 # Región 2 (L_int < z <= L)
eps_r_array = epr1* np.ones(Nz + 1)
eps_r_array[i_int + 1:] = epr2  # El dieléctrico comienza en el índice i_int + 1

# Array de velocidades de onda
v_array = 1 / np.sqrt(eps_r_array * mur)
n = 1/v_array

# --- Discretización temporal ---
v_max = v_array.max()
n_max = n.max()
dt = 0.99 * dz / v_max
Nt = 1000
tiempos = np.arange(0., Nt * dt, dt)
nu = dt / (dz * n_max) # ESTO ESTA BIEN????????

# Inicialización
E = np.zeros([Nz + 1, Nt + 1])
B = np.zeros_like(E)
S = np.zeros_like(E)

for jt in range(Nt):
    t = (jt+1)*dt
    # E[0, jt + 1] = A * np.sin(-w * t)
    if t < T/2: # Mandamos un solo pulso
        E[0, jt + 1] = A * np.sin(-w * t)

    # Cálculo del camapo eléctrico
    for it in range(1, Nz):
        E[it, jt + 1] = E[it, jt] - dt / (dz * n[it] * n[it]) * (B[it + 1, jt] - B[it, jt])

    # Condición pared libre (transmision)
    # E[Nz, jt + 1] = E[Nz - 1, jt] - nu * (B[Nz, jt] - B[Nz - 1, jt])

    # Condicion pared metálica (reflexion)
    #E[Nz, jt + 1] = 0

    # Condición pared absorbente (onda libre)
    E[Nz, jt + 1] = E[Nz - 1, jt] + (nu - 1) / (nu + 1) * (E[Nz - 1, jt + 1] - E[Nz, jt])


    for it in range(1, Nz + 1):
        B[it, jt + 1] = B[it, jt] - dt / dz * (E[it, jt + 1] - E[it - 1, jt + 1])

    B[0, jt + 1] = B[1, jt + 1]

    S[:, jt + 1] = E[:, jt + 1] * B[:, jt + 1]

# Bloque para visualizar
fig, (ax1, ax2) = plt.subplots(2, 1)

lineE, = ax1.plot(z, E[:, 0], lw=2)
ax1.set(xlim=(0, L), ylim=(-1.5, 1.5), xlabel='z', ylabel='E')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axvline(x=L_int, color='r', linestyle='--', label='Interfaz Dieléctrica')
ax1.legend(loc='lower left')

lineB, = ax2.plot(z, B[:, 0], lw=2)
ax2.set(xlim=(0, L), ylim=(-1.5, 1.5), xlabel='z', ylabel='B')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axvline(x=L_int, color='r', linestyle='--', label='Interfaz Dieléctrica')
ax2.legend(loc='lower left')

for j in range(Nt):
    ax1.set_title(f't = {t:.4f} s')
    lineE.set_ydata(E[:, j + 1])
    lineB.set_ydata(B[:, j + 1])
    plt.pause(0.003)