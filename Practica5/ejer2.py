import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros del medio ---
pi = np.pi
mur = 1  # mu_r constante en todo el dominio

# --- Discretización espacial ---
L = 20
Nz = 300
z = np.linspace(0, L, Nz + 1)
dz = z[1] - z[0]

# --- Definición de la interfaz ---
L_int = L / 2  # La interfaz se ubica a la mitad del dominio
i_int = int(L_int / dz)  # Índice donde comienza el dieléctrico (i_int + 1)
print(f"La interfaz se ubica en el índice i={i_int}, z={z[i_int]:.2f}")

# --- Definición de las propiedades del material ---
# Región 1: Vacío (0 < z <= L_int)
epr1 = 1
# Región 2: Dieléctrico (L_int < z <= L)
epr2 = 4

# Array de permitividades relativas por posición (Nz + 1 puntos)
eps_r_array = np.ones(Nz + 1)
eps_r_array[i_int + 1:] = epr2  # El dieléctrico comienza en el índice i_int + 1

# Array de velocidades de onda
v_array = 1 / np.sqrt(eps_r_array * mur)

# Parámetros de la onda incidente (definidos en el vacío)
loon = L / 3
k = 2 * pi / loon
w_inc = v_array[0] * k  # Frecuencia angular en el vacío

# --- Discretización temporal ---
# Usamos la velocidad máxima (la del vacío, v=1) para garantizar la estabilidad
v_max = v_array.max()
dt = 0.99 * dz / v_max
Nt = 2 * Nz
T_periodo = 2 * pi / w_inc  # Periodo en el vacío

# --- Inicialización ---
F = np.zeros([Nz + 1, Nt + 1])  # F es Psi (función de onda)
F0 = 1  # Amplitud de la onda de excitación

# --- Parámetros del esquema por posición ---
# Array A(i) = (v(i) * dt / dz)^2
A_array = (v_array * dt / dz) ** 2

t_interfaz = []
t_derecha = []
tol = 0.001
amplitud = F0/2
t_amplitud = []
t_amplitud_diel = []
amplitud_diel = F0/3
tol_amp = 0.01

# --- Bucle de Cálculo ---
# Inicialización de los dos primeros pasos de tiempo (requerido por el esquema)
for j in np.arange(0, 2):
    F[0, j] = F0 * np.sin(-w_inc * j * dt)  # Excitación en z=0
    # No se propaga onda en el primer paso (F[i, 0] = 0)

# Propagación principal (pasos de tiempo j=1 hasta Nt-1)
for j in np.arange(1, Nt):
    # Condición de excitación (fuente) en z=0
    F[0, j + 1] = F0 * np.sin(-w_inc * j * dt)

    v_abc = v_array[1]
    r_abc = v_abc * dt / dz

    # Ecuación de onda de diferencia finita para i=1 hasta Nz-1
    for i in np.arange(1, i_int):
        A_i = A_array[i]  # Usamos el factor A local
        F[i, j + 1] = A_i * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A_i) * F[i, j] - F[i, j - 1]

    #F[i_int, j + 1] = F[i_int - 1, j] + (r_abc - 1) / (r_abc + 1) * (F[i_int - 1, j + 1] - F[i_int, j])

    v_abc = v_array[Nz]
    r_abc = v_abc * dt / dz

    for i in np.arange(i_int, Nz):
        A_i = A_array[i]  # Usamos el factor A local
        F[i, j + 1] = A_i * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A_i) * F[i, j] - F[i, j - 1]

    # Condición de Contorno Absorbente (ABC) de primer orden en z=L
    # Se utiliza la velocidad del último punto (la del dieléctrico)

    # F[Nz, j+1] = F[Nz-1, j] + (r-1)/(r+1) * ( F[Nz-1, j+1] - F[Nz, j] )
    # Despejando F[Nz, j+1] de la ecuación original (simplificación para r=1):
    # F[Nz, j+1] = F[Nz, j] + (r/(r+1)) * (F[Nz-1, j] - F[Nz, j]) + (1/(r+1)) * (F[Nz-1, j+1] - F[Nz, j-1])
    # Usaremos la versión Mur (primer orden) para evitar iteración

    # Simple primer orden Mur:
    F[Nz, j + 1] = F[Nz - 1, j] + (r_abc - 1) / (r_abc + 1) * (F[Nz - 1, j + 1] - F[Nz, j])
    if np.abs(F[i_int, j]) >tol:
        t_interfaz.append(j)
    if np.abs(F[Nz, j]) > tol:
        t_derecha.append(j)

    if np.abs(F[2,j]+amplitud)<tol_amp:
        t_amplitud.append(dt * j)

    if np.abs(F[i_int+2,j]+amplitud_diel)<tol_amp:
        t_amplitud_diel.append(dt * j)


# Procedemos a calcular la velocidad de propagación
# en cada medio dividiendo espacio recorrido por tiempo empleado:
T_interfaz = t_interfaz[0]*dt
T_derecha = t_derecha[0]*dt

velocidad_vac = (z[i_int] - z[0]) / T_interfaz
velocidad_diel = (z[-1] - z[i_int]) / (T_derecha-T_interfaz)

print(f"La velocidad de propagación en el vacío es v={velocidad_vac:.2f}")
print(f"La velocidad de propagación en el dieléctrico es v={velocidad_diel:.2f}")

# Ahora queremos hallar el periodo en cada caso.
# Para ello, fijaremos una cierta amplitud y una tolerancia.
# Veremos cuánto tiempo transcurre entre dos veces consecutivas
# en las que se alcanza dicha amplitud:
# Tomaremos el primer valor del periodo que obtengamos, pues después
# se ven involucrados los efectos de reflexión y transmisión.
periodo_exp = t_amplitud[2]-t_amplitud[0]
periodo_diel_exp = t_amplitud_diel[2]-t_amplitud_diel[0]
print(f"El periodo de nuestra onda en el vacío es T={periodo_exp:.2f}")
print(f"El periodo de nuestra onda en el dieléctrico es T={periodo_diel_exp:.2f}")

# Para hallar la longitud de onda, fijaremos un instante de tiempo
# en vez de una posición. Para ese instante de tiempo, veremos cuánta
# distancia hay entre dos puntos espaciales consecutivos donde se alcance
# la misma amplitud:
# A la hora de fijar un instante de tiempo, debemos asegurarnos de que
# la onda ya haya alcanzado el dieléctrico lo suficiente.

onda_vacio = F[0:i_int, t_interfaz[0]]
onda_diel = F[i_int:Nz, t_derecha[0]]
pos_vacio = []
pos_diel = []
tol_long = 0.05 # Ponemos amplitudes negativas para que las encuentre
amplitud_vac = onda_vacio[1]
amplitud_dielec = onda_diel[1]
for i in range(i_int):
    if np.abs(onda_vacio[i]-amplitud_vac) < tol_long:
        pos_vacio.append(z[i])
for i in np.arange(i_int, Nz):
    if np.abs(onda_diel[i-i_int]-amplitud_dielec) < tol_long:
        pos_diel.append(z[i])

long_vac = pos_vacio[3]-pos_vacio[1]
long_diel = pos_diel[3]-pos_diel[1]
print(f"La longitud de onda en el vacío es L={long_vac:.2f}")
print(f"La longitud de onda en el dieléctrico es L={long_diel:.2f}")
# ESTO SALE MAL
# Y SI MEJOR USAMOS LAMBDA=v*T Y YA ESTÁ???

# --- Visualización de la Animación ---
from matplotlib.animation import FuncAnimation

# Configuración de la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(z, F[:, 0], 'b-', lw=2, label='Solución Numérica')
ax.set_xlim(0, L), ax.set_ylim(-1.5, 1.5),
ax.set_xlabel('Posición z [unidades]'), ax.set_ylabel('Amplitud F(z, t)')
ax.set_title('Propagación de Onda en Interfaz Dieléctrica ($\\epsilon_r$: 1 $\\to$ 4)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.axvline(x=L_int, color='r', linestyle='--', label='Interfaz Dieléctrica (z=10)')
ax.legend(loc='lower left')

for j in range(Nt):
    ax.set_title(f't = {j*dt:.4f} s')
    line.set_ydata(F[:, j + 1])
    plt.pause(0.003)
