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

# --- Bucle de Cálculo ---
# Inicialización de los dos primeros pasos de tiempo (requerido por el esquema)
for j in np.arange(0, 2):
    F[0, j] = F0 * np.sin(-w_inc * j * dt)  # Excitación en z=0
    # No se propaga onda en el primer paso (F[i, 0] = 0)

# Propagación principal (pasos de tiempo j=1 hasta Nt-1)
for j in np.arange(1, Nt):
    # Condición de excitación (fuente) en z=0
    F[0, j + 1] = F0 * np.sin(w_inc * j * dt)

    v_abc = v_array[1]
    r_abc = v_abc * dt / dz

    # Ecuación de onda de diferencia finita para i=1 hasta Nz-1
    for i in np.arange(1, i_int):
        A_i = A_array[i]  # Usamos el factor A local
        F[i, j + 1] = A_i * (F[i + 1, j] + F[i - 1, j]) + 2 * (1 - A_i) * F[i, j] - F[i, j - 1]

    F[i_int, j + 1] = F[i_int - 1, j] + (r_abc - 1) / (r_abc + 1) * (F[i_int - 1, j + 1] - F[i_int, j])

    v_abc = v_array[Nz]
    r_abc = v_abc * dt / dz

    for i in np.arange(i_int+1, Nz):
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


# Función de actualización para la animación
def animate(j):
    """Actualiza la gráfica para el paso de tiempo j."""
    line.set_ydata(F[:, j])
    ax.set_title(f"Propagación de Onda en Interfaz Dieléctrica | t={j * dt:.4f} s | Paso={j}/{Nt}")
    return line,


# Crear la animación
# Reducimos los frames para que sea más rápida y visible
frames_to_show = np.arange(0, Nt, 10)
ani = FuncAnimation(fig, animate, frames=frames_to_show, interval=50, blit=True)

plt.show()

print("\n--- Resultados de la Simulación ---")
print(f"Factor Courant máximo (Vacío): {np.sqrt(A_array[0]):.4f}")
print(f"Factor Courant en Dieléctrico (er=4): {np.sqrt(A_array[Nz]):.4f}")
print(
    "¡Observa cómo la onda se refleja parcialmente en la interfaz y la longitud de onda se acorta al entrar al dieléctrico!")