import numpy as np
import matplotlib.pyplot as plt

# Definición de parámetros
N = 50  # número de iteraciones
Nx, Ny = 15, 10  # número de puntos en los ejes x, y
h = 1  # discretización
Lx = Nx * h  # longitud en el eje x
Ly = Ny * h  # longitud en el eje y
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)

# Potenciales en los límites (Condiciones de Contorno de Dirichlet)
V1 = 0
V2 = 0
V3 = 10
V4 = 20

# Inicialización de la matriz de potencial V
V = np.zeros((Nx + 1, Ny + 1))
Vn = np.zeros((Nx + 1, Ny + 1))  # V para la siguiente iteración
# Establecer las condiciones de contorno
V[0, :] = V1
V[-1, :] = V2
V[:, 0] = V3
V[:, -1] = V4
# Esquinas (promedio de los potenciales adyacentes)
V[0, 0] = (V1 + V3) / 2
V[Nx, 0] = (V2 + V3) / 2
V[0, Ny] = (V1 + V4) / 2
V[Nx, Ny] = (V2 + V4) / 2

# Método de Jacobi
VJ = V.copy()  # Potencial usando Jacobi
Vn = VJ.copy()  # Jacobi requiere de dos matrices de potencial
RJ = np.zeros(N)  # Residuo usando Jacobi
for n in np.arange(0, N):
    # Potencial
    for jt in np.arange(1, Ny):
        for it in np.arange(1, Nx):
            Vn[it, jt] = 0.25 * (VJ[it + 1, jt] + VJ[it - 1, jt] + VJ[it, jt + 1] + VJ[it, jt - 1])

    # Residuo
    RJ[n] = np.sum(abs(VJ - Vn))
    # print("Residuo ({}) = {:.4f}".format(n,R[n]))
    VJ = Vn.copy()

# Método de Gauss-Seidel
VGS = V.copy()
RGS = np.zeros(N)  # Residuo usando Gauss-Seidel
for n in np.arange(0, N):
    V_old = VGS.copy()
    # Potencial
    for jt in np.arange(1, Ny):
        for it in np.arange(1, Nx):
            VGS[it, jt] = 0.25 * (VGS[it + 1, jt] + VGS[it - 1, jt] + VGS[it, jt + 1] + VGS[it, jt - 1])

    # Residuo
    RGS[n] = np.sum(abs(V_old - VGS))

# Dibujar el potencial
fig, ax1 = plt.subplots(num=1)
ax1.semilogy(RJ, label='Jacobi')
ax1.semilogy(RGS, label='Gauss-Seidel')
ax1.legend()
ax1.set_title('Residuo tras {} iteraciones'.format(N))
ax1.set_xlabel('Iteración')
ax1.set_ylabel('R')

fig2, ax2 = plt.subplots(num=2)
X, Y = np.meshgrid(x, y)
cont = ax2.contourf(X, Y, np.transpose(VGS))
fig2.colorbar(cont, ax=ax2)
ax2.set_title('Potencial Gauss-Seidel: {} iter.'.format(N))
ax2.set_xlabel('x')
ax2.set_ylabel('y')

fig3, ax3 = plt.subplots(num=3)
X, Y = np.meshgrid(x, y)
cont = ax3.contourf(X, Y, np.transpose(VJ))
fig3.colorbar(cont, ax=ax3)
ax3.set_title('Potencial Jacobi {} iter.'.format(N))
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.show()  # Añadido para mostrar las gráficas al ejecutar el script


# Solución directa pag 34
import numpy as np
np.set_printoptions(precision=3)

V1, V2 = 1, 2
# Matriz del vector de potenciales conocidos (V1 en filas 1, 2, V2 en filas 3, 4, 5, 6)
B = np.array([[V1], [V1], [V2], [V2], [V2], [V2]])

# Matriz A del sistema lineal Ax = B para los 6 nodos internos (Va...Vf)
# Obtenida de la "molécula de 5 puntos" aplicada a cada nodo
A = np.array([
    [-4, 1, 0, 1, 0, 0],
    [1, -4, 1, 0, 1, 0],
    [0, 1, -4, 0, 0, 1],
    [1, 0, 0, -4, 1, 0],
    [0, 1, 0, 1, -4, 1],
    [0, 0, 1, 0, 1, -4]
])

print("Determinante de A ({:.4f})".format(np.linalg.det(A)))

# Inversión de la matriz A
A_inv = np.linalg.inv(A)

# Cálculo del vector de potenciales X = A_inv * B
X = np.dot(A_inv, B)
print("X*={}".format(np.transpose(X)))