import numpy as np
import matplotlib.pyplot as plt

# Definición de parámetros
N = 50  # número de iteraciones
h = 0.25  # misma discretización en los ejes x e y
Lx = 2  # longitud en el eje x
Ly = 2  # longitud en el eje y
Nx, Ny = int(Lx/h), int(Ly/h)  # número de puntos en los ejes x, y
x = np.linspace(0, Lx, Nx + 1)
y = np.linspace(0, Ly, Ny + 1)

# Potenciales en los límites (Condiciones de Contorno de Dirichlet)
V1 = 10
V2 = -40
V3 = 50
V4 = 80

# Inicialización de la matriz de potencial V
V = np.zeros((Nx + 1, Ny + 1))
Vn = np.zeros((Nx + 1, Ny + 1))  # V para la siguiente iteración

# Establecer las condiciones de contorno
V[0, :] = V1
V[-1, :] = V3
V[:, 0] = V2
V[:, -1] = V4

# Esquinas (promedio de los potenciales adyacentes) para que no haya mucha singularidad
V[0, 0] = (V1 + V3) / 2
V[Nx, 0] = (V2 + V3) / 2
V[0, Ny] = (V1 + V4) / 2
V[Nx, Ny] = (V2 + V4) / 2

# Método de Jacobi
VJ = V.copy()  # Potencial usando Jacobi
Vn = VJ.copy()  # Jacobi requiere de dos matrices de potencial
RJ = np.zeros(N)  # Residuo usando Jacobi
for n in np.arange(1, N+1):
    # Potencial
    for jt in np.arange(1, Ny):
        for it in np.arange(1, Nx):
            Vn[it, jt] = 0.25 * (VJ[it + 1, jt] + VJ[it - 1, jt] + VJ[it, jt + 1] + VJ[it, jt - 1])

    # Residuo
    RJ[n-1] = np.sum(abs(VJ - Vn))
    # print(f"V con Jacobi en la iteración {n} = {Vn}")
    VJ = Vn.copy()

# Método de Gauss-Seidel
VGS = V.copy()
RGS = np.zeros(N)  # Residuo usando Gauss-Seidel
for n in np.arange(1, N+1):
    V_old = VGS.copy()
    # Potencial
    for jt in np.arange(1, Ny):
        for it in np.arange(1, Nx):
            VGS[it, jt] = 0.25 * (VGS[it + 1, jt] + VGS[it - 1, jt] + VGS[it, jt + 1] + VGS[it, jt - 1])

    # Residuo
    RGS[n-1] = np.sum(abs(V_old - VGS))

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
plt.show()

print(f"V en el centro con Jacobi {VJ[int(Nx/2), int(Ny/2)]}")
print(f"V en el centro con Gauss-Seidel {VGS[int(Nx/2), int(Ny/2)]}")