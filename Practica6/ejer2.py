import numpy as np
import matplotlib.pyplot as plt

# Constantes del problema
pi=np.pi
L=3                              #espacio a discretizar
Nz=150                           #N=numero de ptos
w=2*pi                           #frec. angular excitacion
A=1                            #Amplitud del campo
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
tiempos = np.arange(0.,(Nt +1) * dt, dt)
nu = dt / (dz * n_max) # ESTO ESTA BIEN????????

# Inicialización
E = np.zeros([Nz + 1, Nt + 1])
B = np.zeros_like(E)
S = np.zeros_like(E)

# Miramos cuando llega la onda
t_interfaz = []
t_derecha = []
tol = 0.001

for jt in range(Nt):
    t = (jt+1)*dt
    # E[0, jt + 1] = A * np.sin(-w * t)
    if t < T/2: # Mandamos un solo pulso
        E[0, jt + 1] = A * np.sin(-w * t)

    # Cálculo del camapo eléctrico
    for it in range(1, Nz):
        E[it, jt + 1] = E[it, jt] - dt / (dz * n[it] * n[it]) * (B[it + 1, jt] - B[it, jt])

    # Condición pared libre (transmision)
    #E[Nz, jt + 1] = E[Nz - 1, jt] - nu * (B[Nz, jt] - B[Nz - 1, jt])

    # Condicion pared metálica (reflexion)
    # E[Nz, jt + 1] = 0

    # Condición pared absorbente (onda libre)
    E[Nz, jt + 1] = E[Nz - 1, jt] + (nu - 1) / (nu + 1) * (E[Nz - 1, jt + 1] - E[Nz, jt])


    for it in range(1, Nz + 1):
        B[it, jt + 1] = B[it, jt] - dt / dz * (E[it, jt + 1] - E[it - 1, jt + 1])

    B[0, jt + 1] = B[1, jt + 1]

    S[:, jt + 1] = E[:, jt + 1] * B[:, jt + 1]

    # Miramos cuando llega a la interfaz la fase:
    if np.abs(E[i_int, jt]) >tol:
        t_interfaz.append(jt)
    if np.abs(E[Nz, jt]) > tol:
        t_derecha.append(jt)

# Bloque para visualizar
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

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

lineS, = ax3.plot(z, S[:, 0], lw=2)
ax3.set(xlim=(0, L), ylim=(-1.5, 1.5), xlabel='z', ylabel='S')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.axvline(x=L_int, color='r', linestyle='--', label='Interfaz Dieléctrica')
ax3.legend(loc='lower left')

"""for j in range(160): # cuando rebota
    ax1.set_title(f't = {j*dt:.4f} s')
    lineE.set_ydata(E[:, j + 1])
    lineB.set_ydata(B[:, j + 1])
    lineS.set_ydata(S[:, j + 1])
    plt.pause(0.003)"""


# Ejercicio 2c), dibujamos para z = landa/2. Como landa = i_landa * dz..
fig, ax = plt.subplots(figsize=(10, 5))
j_landa = int(0.3*L/dz)
ax.plot(tiempos[:160], E[j_landa, :160], color='b',label='Numérica')
ax.set_xlabel(r'$t$ [s]')
ax.set_ylabel(r'$E(0.3L,t) $ [V/m]')
ax.set_title(f'Evolución temporal del campo eléctrico z={0.3*L:.2f} m')
ax.set_ylim(-1.25, 1.25)
ax.grid(True, linestyle='--')
ax.legend()
plt.show()

# Ejercicio 2d)
E_ref = np.max(np.abs(E[j_landa, t_interfaz[0]:160]))
r_exp = E_ref/A
r_teo = (n[-1] - n[0])/(n[-1] + n[0])
err_rel_r = np.abs(r_exp - r_teo)/r_teo
print(f"El factor de reflexion experimental es r={r_exp}")
print(f"El factor de reflexion teorico es r={r_teo}")
print(f"El error relativo es de err={err_rel_r}")


# Ejercicio 2e)
j_landa2 = int(0.7*L/dz)
E_tra = np.max(np.abs(E[j_landa2,:160]))
t_exp = E_tra/A
t_teo = 2*(n[0])/(n[-1] + n[0])
err_rel_t = np.abs(t_exp - t_teo)/t_teo
print(f"El factor de transmision experimental es t={t_exp}")
print(f"El factor de transmision teorico es t={t_teo}")
print(f"El error relativo es de err={err_rel_t}")

# Ejercicio 2d)
R_exp = r_exp**2
T_exp =  n[-1]/n[0]*t_exp**2
print(f"La suma de los coeficientes R + T={R_exp + T_exp}")


#Ejer 3
Nizq = int(i_int/2)
Nder = int(3*i_int/2)

S_izq = S[Nizq, :190]
S_der = S[Nder, :190]

# Configuración de la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(tiempos[:190], S_izq, 'darkcyan', lw=2, label='Solución Numérica')
ax.set_ylim(-1.25, 1.25)
ax.set_xlabel('t [s]'), ax.set_ylabel(r'S(t) [W/$m^2$]')
ax.set_title(r'Propagación del vector de Poynting $\ \vec{S}(t)$ para $\ z = 0.25L <z_0$ fijo')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left')

# Configuración de la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(tiempos[:190], S_der, 'g', lw=2, label='Solución Numérica')
ax.set_ylim(-1.25, 1.25)
ax.set_xlabel('t [s]'), ax.set_ylabel(r'S(t) [W/$m^2$]')
ax.set_title(r'Propagación del vector de Poynting $\ \vec{S}(t)$ para $\ z = 0.75L >z_0$ fijo')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left')
plt.show()

