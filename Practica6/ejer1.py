import numpy as np
import matplotlib.pyplot as plt

# Constantes del problema
pi=np.pi
L=3                              #espacio a discretizar
Nz=150                           #N=numero de ptos
w=2*pi                           #frec. angular excitacion
A=1                              #Amplitud del campo
T = 2*np.pi/w
er,mur=1, 1                       #permitividad y permeabilidad relativa
n =np.sqrt(er*mur)               #indice de refraccion
ratio=.99

# Distribución espacial
z = np.linspace(0, L, Nz+1)
dz = z[1]-z[0]

# Distribución temporal
dt = ratio*dz/n
Nt = 1000
tiempos = np.arange(0., Nt * dt, dt)
nu = dt / (dz * n)

# Inicialización
E = np.zeros([Nz + 1, Nt + 1])
B = np.zeros_like(E)
S = np.zeros_like(E)

for jt in range(Nt):
    t = (jt+1)*dt
    # E[0, jt + 1] = A * np.sin(-w * t)
    if t < T/2: # Mandamos un solo pulso
        E[0, jt + 1] = A * np.sin(-w * t)

    # Cálculo del cmapo eléctrico
    for it in range(1, Nz):
        E[it, jt + 1] = E[it, jt] - dt / (dz * n * n) * (B[it + 1, jt] - B[it, jt])

    # Condición pared libre (transmision)
    E[Nz, jt + 1] = E[Nz - 1, jt] - nu * (B[Nz, jt] - B[Nz - 1, jt])

    # Condicion pared metálica (reflexion)
    #E[Nz, jt + 1] = 0

    # Condición pared absorbente (onda libre)
    #E[Nz, jt + 1] = E[Nz - 1, jt] + (nu - 1) / (nu + 1) * (E[Nz - 1, jt + 1] - E[Nz, jt])


    for it in range(1, Nz + 1):
        B[it, jt + 1] = B[it, jt] - dt / dz * (E[it, jt + 1] - E[it - 1, jt + 1])

    B[0, jt + 1] = B[1, jt + 1]

    S[:, jt + 1] = E[:, jt + 1] * B[:, jt + 1]

# Bloque para visualizar
fig, (ax1, ax2) = plt.subplots(2, 1)
lineE, = ax1.plot(z, E[:, 0], lw=2)
lineB, = ax2.plot(z, B[:, 0], lw=2)
ax1.set(xlim=(0, L), ylim=(-1.5, 1.5), xlabel='z', ylabel='E')
ax2.set(xlim=(0, L), ylim=(-1.5, 1.5), xlabel='z', ylabel='B')
for j in range(Nt):
    ax1.set_title(f't = {t:.4f} s')
    lineE.set_ydata(E[:, j + 1])
    lineB.set_ydata(B[:, j + 1])
    plt.pause(0.003)