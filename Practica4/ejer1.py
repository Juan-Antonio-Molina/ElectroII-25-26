"""
***
Created on Fri Oct 8 17:42:47 2021
@author: gregzonc

Adaptado de García Olmedo 2006 (Ap'endice B: Secci'on B1.3.2)
***
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def Pot(V0, V1, V2, V3, V4, R):
    """
    Actualiza el potencial en un nodo seg'un el m'etodo de
    diferencias finitas
    """
    Vout = (V1 + V2 + V3 + V4)/4
    Rout = R + np.abs(V0 - Vout)
    return Vout, Rout


# Parámetros del problema
L = 1.00 # Espacio a modelar
N = 30 # Red: N x N
delta = L/N # Lado de celda unidad
Nint = 1000 # Numero de iteraciones
m = 5 # Altura del electrodo (nodos)
l = 5 # tamaño del electrodo (nodos)
V1 = 1 # Potencial del electrodo

# Inicializaci'on
V = np.zeros([N + 1, N + 1])
V[0:l+1, m] = V1 # Aplicar condici'on de contorno en el electrodo
V0 = V.copy() # Guardar copia inicial para comparaci'on
x = np.linspace(0, L, N + 1, endpoint=True)
y = np.linspace(0, L, N + 1, endpoint=True)

# Vector para almacenar residuos
R = np.zeros(Nint)

# Bucle principal de iteraci'on
for i in np.arange(Nint):
    # Lado izquierdo (entre los electrodos)
    for jt in np.arange(1, m):
        Vold = V[0, jt]
        V[0, jt], R[i] = 1/4 * (V[0, jt - 1] + V[0, jt + 1] + 2 * V[1, jt]), R[i] + np.abs(Vold - V[0, jt])

    # Lado izquierdo (arriba del electrodo)
    for jt in np.arange(m + 1, N):
        Vold = V[0, jt]
        V[0, jt], R[i] = 1/4 * (V[0, jt - 1] + V[0, jt + 1] + 2 * V[1, jt]), R[i] + np.abs(Vold - V[0, jt])

    # Esquina superior izquierda
    Vold = V[0, N]
    V[0, N] = (V[0, N - 1] + V[1, N])/2
    R[i] = R[i] + np.abs(Vold - V[0, N])

    # Lado superior
    for it in np.arange(1, N):
        Vold = V[it, N]
        V[it, N] = 1/4 * (2 * V[it, N - 1] + V[it - 1, N] + V[it + 1, N])
        R[i] = R[i] + np.abs(Vold - V[it, N])

    # Esquina superior derecha
    Vold = V[N, N]
    V[N, N] = (V[N, N - 1] + V[N - 1, N])/2
    R[i] = R[i] + np.abs(Vold - V[N, N])

    # Lado derecho
    for jt in np.arange(1, N):
        Vold = V[N, jt]
        V[N, jt] = 1/4 * (2 * V[N - 1, jt] + V[N, jt - 1] + V[N, jt + 1])
        R[i] = R[i] + np.abs(Vold - V[N, jt])

    # Secci'on interior x=(1, ..., N - 1); y < m
    for it in np.arange(1, N):
        for jt in np.arange(1, m):
            V[it, jt], R[i] = Pot(V[it, jt], V[it + 1, jt], V[it - 1, jt], V[it, jt + 1], V[it, jt - 1], R[i])

    # Secci'on interior x=(l+1, ..., N - 1); y = m
    for it in np.arange(l+1, N):
        V[it, m], R[i] = Pot(V[it, m], V[it + 1, m], V[it - 1, m], V[it, m + 1], V[it, m - 1], R[i])

    # Secci'on interior x=(1, ..., N - 1); y=(m + 1, ..., N - 1)
    for it in np.arange(1, N):
        for jt in np.arange(m + 1, N):
            V[it, jt], R[i] = Pot(V[it, jt], V[it + 1, jt], V[it - 1, jt], V[it, jt + 1], V[it, jt - 1], R[i])

# Calcular el campo
Ex = np.zeros_like(V)
Ex[1:N,:] = -(V[2:N+1,:] - V[0:N-1,:])/(2*delta)
Ex[0,:] = -(V[1,:] - V[0,:])/delta
Ex[N,:] = -(V[N,:] - V[N-1,:])/delta

Ey = np.zeros_like(V)
Ey[:,1:N] = -(V[:,2:N+1] - V[:,0:N-1])/(2*delta)
Ey[:,0] = -(V[:,1] - V[:,0])/delta
Ey[:,N] = -(V[:,N] - V[:,N-1])/delta

# Visualización del potencial
X, Y = np.meshgrid(x, y)
fig, ax1 = plt.subplots(num='Potencial')
barra = ax1.contourf(X, Y, np.transpose(V), levels=50, cmap='viridis')
fig.colorbar(barra, ax=ax1, label='Potencial (V)')
ax1.contour(X, Y, np.transpose(V), colors='k', linewidths=0.5, levels=15) # Algunas líneas de contorno

# Dibujamos el electrodo
xelec = np.array([0, delta * l])
yelec = delta * m * np.array([1,1])
ax1.plot(xelec, yelec, color='r', linewidth=3, solid_capstyle='butt', label='Electrodo')
ax1.legend() # Muestra la leyenda del electrodo

ax1.set_title('Distribución de Potencial')
ax1.set_xlabel('Posición x (m)')
ax1.set_ylabel('Posición y (m)')
ax1.set_aspect('equal')
plt.show()

# Visualización del campo
fig, ax1 = plt.subplots(num='Potencial')
barra = ax1.contourf(X, Y, np.transpose(V), levels=50, cmap='viridis')
fig.colorbar(barra, ax=ax1, label='Potencial (V)')
ax1.streamplot(X, Y, np.transpose(Ex), np.transpose(Ey),
               density=1.2, linewidth=0.8, color='k', arrowsize=0.8)

# Dibujamos el electrodo
xelec = np.array([0, delta * l])
yelec = delta * m * np.array([1,1])
ax1.plot(xelec, yelec, color='r', linewidth=3, solid_capstyle='butt', label='Electrodo')
ax1.legend()

ax1.set_title('Distribución de Potencial y Campo Eléctrico')
ax1.set_xlabel('Posición x (m)')
ax1.set_ylabel('Posición y (m)')
ax1.set_aspect('equal')
plt.show()

# Gr'afico del residuo
fig, p2 = plt.subplots(num='Residuo', figsize=(8, 4))
p2.semilogy(R, label='Residuo Acumulado', color='r', linewidth=2)
p2.set_title('Convergencia del Método de Diferencias Finitas')
p2.set_xlabel('Número de Iteración')
p2.set_ylabel('Residuo Acumulado (Escala logarítmica)')
p2.grid(True, which="both", ls="--", alpha=0.6) # Añadir rejilla
p2.legend()
plt.show()

# EJERCICIO 2
N_ajuste = np.arange(200,1000)
R_ajuste = np.log(R[200:])
res = stats.linregress(N_ajuste, R_ajuste) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")
