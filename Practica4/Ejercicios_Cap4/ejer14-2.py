import numpy as np
import matplotlib.pyplot as plt

# Definición de parámetros
I = 5 # numero de iteraciones
h = 0.25
L = 1
N = int(L/h)
x = np.linspace(0, L, N + 1) # contamos desde el 0 ahora

# Potenciales en los límites (Condiciones de Contorno de Dirichlet)
V1 = 0
V2 = 1

# Inicialización de la matriz de potencial V con sentido fisico
V = np.zeros(N + 1)
for i in np.arange(0, N + 1):
    V[i] = V1 + (V2-V1)/(L - 0) * (i*h - 0)
Vn = np.zeros_like(V)  # V para la siguiente iteración

# Método de Jacobi
VJ = V.copy()  # Potencial usando Jacobi
Vn = VJ.copy()  # Jacobi requiere de dos matrices de potencial
RJ = np.zeros(I)  # Residuo usando Jacobi
for n in np.arange(1, I+1):
    for i in np.arange(1, N): # Cuenta desde el 1 hasta el N-1.
        Vn[i] = 0.5 * (VJ[i+1] + VJ[i-1]) - 0.5 * h**2 * (i*h + 1)
    # Residuo
    RJ[n-1] = np.sum(abs(VJ - Vn))
    # print("Residuo ({}) = {:.4f}".format(n,R[n]))
    print(f"V con Jacobi en la iteración {n} = {Vn}")
    VJ = Vn.copy()

# Método de Gauss-Seidel
VGS = V.copy()
RGS = np.zeros(I)  # Residuo usando Gauss-Seidel
for n in np.arange(1, I+1):
    V_old = VGS.copy()
    # Potencial
    for i in np.arange(1, N):  # Cuenta desde el 1 hasta el N-1.
        VGS[i] = 0.5 * (VGS[i + 1] + VGS[i - 1]) - 0.5 * h**2 * (i * h + 1)

    # Residuo
    RGS[n-1] = np.sum(abs(V_old - VGS))
    print(f"V con G-S en la iteración {n} = {VGS}")

# Dibujar el potencial
fig, ax1 = plt.subplots(num=1)
ax1.semilogy(RJ, label='Jacobi')
ax1.semilogy(RGS, label='Gauss-Seidel')
ax1.legend()
ax1.set_title(f'Residuo tras {I} iteraciones')
ax1.set_xlabel('Iteración')
ax1.set_ylabel('R')

plt.show()  # Añañido para mostrar las gráficas al ejecutar el script