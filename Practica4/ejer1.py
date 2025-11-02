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

# EJERCICIO 3
### Dibujamos el potencial
fig, ax3 = plt.subplots(figsize=(8, 6))

# Líneas equipotenciales
levels = [0, 0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
contours = ax3.contour(X, Y, np.transpose(V), levels=levels, colors='darkcyan',
                     alpha=0.7)
ax3.clabel(contours, inline=True, fontsize=9, fmt='%1.2f')
# El campo eléctrico:
ax3.streamplot(X, Y, np.transpose(Ex), np.transpose(Ey),
               density=1.2, linewidth=0.8, color='k', arrowsize=0.8)

# Dibujamos el electrodo
xelec = np.array([0, delta * l])
yelec = delta * m * np.array([1,1])
ax3.plot(xelec, yelec, color='r', linewidth=3, solid_capstyle='butt', label='Electrodo')
ax3.legend()

# Configuración del gráfico
plt.title('Líneas equipotenciales del sistema y campo eléctrico')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.gca().set_aspect('equal', adjustable='box')
ax3.grid(True, linestyle='--')
plt.legend(loc='lower right')
plt.show()

## Comprobamos las condiciones de contorno
# Dirichlet1:
Velec = V[0:l+1, m] # Es el potencial en el electrodo
Xelec = x[0:l+1]
fig, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(Xelec, Velec, label='Puntos de datos')
ax4.plot(Xelec, Velec, color='darkcyan')

ax4.set_xlabel(r'X [m]')
ax4.set_ylabel(r'V [V]')
ax4.set_title('Potencial en el electrodo')
ax4.legend()
ax4.grid(True, linestyle='--')
plt.show()

err_d1 =np.abs(Velec-1)
max_err_d1 = err_d1.max()
print(f"El máximo de los errores en el electrodo es Err1= {max_err_d1}")

# Dirichlet2:
Vinf = V[0:N+1, 0] # Es el potencial en el contorno inferior
fig, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(x, Vinf, label='Puntos de datos')
ax5.plot(x, Vinf, color='darkcyan')

ax5.set_xlabel(r'X [m]')
ax5.set_ylabel(r'V [V]')
ax5.set_title('Potencial en el contorno inferior')
ax5.legend()
ax5.grid(True, linestyle='--')
plt.show()

err_d2 =np.abs(Vinf)
max_err_d2 = err_d2.max()
print(f"El máximo de los errores en el contorno inferior es Err2= {max_err_d2}")

# Neumann1:
# Vamos a calcular la máxima diferencia entre V1,j y V0,j para todo j:
V0j = V[0, 1:N+1]
V1j = V[1, 1:N+1]
difs1 = np.abs((V0j - V1j)/V0j)
max_difs1 = difs1.max()
print(f"La máxima de las diferencias en el contorno izquierdo es Difs1= {max_difs1}")

# Neumann2:
# Vamos a calcular la máxima diferencia entre Vi,N-1 y Vi,N para todo i:
ViN_1 = V[0:N+1, N-1]
ViN = V[0:N+1, N]
difs2 = np.abs((ViN_1 - ViN)/ViN)
max_difs2 = difs2.max()
print(f"La máxima de las diferencias en el contorno superior es Difs2= {max_difs2}")

# Neumann3:
# Vamos a calcular la máxima diferencia entre VN-1,j y VN,j para todo j:
VNj = V[N, 1:N+1]
VN_1j = V[N-1, 1:N+1]
difs3 = np.abs((VNj - VN_1j)/VNj)
max_difs3 = difs3.max()
print(f"La máxima de las diferencias en el contorno derecho es Difs3= {max_difs3}")

# Comparación con condensador infinito:

### Dibujamos el potencial
fig, ax6 = plt.subplots(figsize=(8, 6))

x_cerca = x[0:l+1]
y_cerca = y[0:m+1]
Xcerca, Ycerca = np.meshgrid(x_cerca, y_cerca)
V_cerca = V[0:l+1,0:m+1]
Ex_cerca = Ex[0:l+1,0:m+1]
Ey_cerca = Ey[0:l+1,0:m+1]

# Líneas equipotenciales
levels = [0, 0.05, 0.1, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
contours = ax6.contour(Xcerca, Ycerca, np.transpose(V_cerca), levels=levels, colors='darkcyan',
                     alpha=0.7)
ax6.clabel(contours, inline=True, fontsize=9, fmt='%1.2f')
# El campo eléctrico:
ax6.streamplot(Xcerca, Ycerca, np.transpose(Ex_cerca), np.transpose(Ey_cerca),
               density=1.2, linewidth=0.8, color='k', arrowsize=0.8)

# Dibujamos el electrodo
xelec = np.array([0, delta * l])
yelec = delta * m * np.array([1,1])
ax6.plot(xelec, yelec, color='r', linewidth=3, solid_capstyle='butt', label='Electrodo')
ax6.legend()

# Configuración del gráfico
plt.title('Líneas equipotenciales del sistema y campo eléctrico')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.gca().set_aspect('equal', adjustable='box')
ax6.grid(True, linestyle='--')
plt.legend(loc='lower right')
plt.show()


