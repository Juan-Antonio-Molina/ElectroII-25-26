"""
ejer2.py - Ejercicio 2: Validación de las condiciones de contorno

Objetivos:
-

Created on Sun Sept 21 14:07:23 2025
@autors: juanan, manuelpi
"""
from em2um import SemiCajaPotencial
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Datos iniciales
L = 2
V0 = 2
Nx = 500
NumModos = 188
V_teo = V0*np.ones(Nx)

# V(0,y)=V(L,y)=0 cuando y>0:
y = np.linspace(0.05 * L, L, Nx)
x1 = np.zeros_like(y)
x2 = L*np.ones(Nx)

tolerancia = V0 * 10 ** (-3)
V1 = SemiCajaPotencial(NumModos, V0, L, x1, y)
if np.max(np.abs(V1)) < tolerancia:
    print(" La condición de V(0,y) = 0 se cumple para el número de modos dado")
else:
    print(" La condición de V(0,y) = 0 no se cumple para el número de modos dado")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y, V1, color="blue", label=f'Puntos de datos')
ax.plot(y, np.zeros_like(V1), color='red',label=r'V = 0')

ax.set_xlabel('y [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $x= 0$')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

V2 = SemiCajaPotencial(NumModos, V0, L, x2, y)
if np.max(np.abs(V2)) < tolerancia:
    print(" La condición de V(L,y) = 0 se cumple para el número de modos dado")
else:
    print(" La condición de V(L,y) = 0 no se cumple para el número de modos dado")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y, V2, color="blue", label=f'Puntos de datos')
ax.plot(y, np.zeros_like(V2), color='red',label=r'V = 0')

ax.set_xlabel('y [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $x= L$')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()


# V(x,0)=V0 cuando 0x está entre 0 y L:
x3 = np.linspace(0.05 * L, 0.95 * L, Nx)
y3 = np.zeros_like(x3)
V3 = SemiCajaPotencial(NumModos, V0, L, x3, y3)
V3_2 = SemiCajaPotencial(50, V0, L, x3, y3)
error = np.abs(V3/V_teo - 1)

if np.max(np.abs(V3/V_teo - 1)) <= 0.01:
    print(" La condición de V(x,0) = V0 se cumple para el número de modos dado")
else:
    print(" La condición de V(x,0) = V0 no se cumple para el número de modos dado")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x3, V3, color="blue", label=f'Puntos de datos')
ax.plot(x3, V0*np.ones(len(V3)), color='red',label=r'$V = V_0$')

ax.set_xlabel('x [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $y = 0$')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x3, V3_2, color="blue", label=f'Puntos de datos')
ax.plot(x3, V0*np.ones(len(V3)), color='red',label=r'$V = V_0$')

ax.set_xlabel('x [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $y = 0$')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# V tiende a 0 cuando "y" tiende a infinito
y4 = np.linspace(L, 10*L, Nx)
x4 = L/2*np.ones(Nx)
V4 = SemiCajaPotencial(NumModos, V0, L, x4, y4) / V0
y_exp = np.exp((-1)*y4)
y_exp2 = np.exp(2*(-1)*y4)

fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(y4, V4, label='Puntos de datos')
ax_ajuste.plot(y4, y_exp, color='red', label='pendiente -1')
ax_ajuste.plot(y4, y_exp2, color='indigo', label='pendiente -2')

ax_ajuste.set_xlabel('y [m]')
ax_ajuste.set_ylabel(r'$log(V/V_0)$')
ax_ajuste.set_yscale('log')
ax_ajuste.set_title(r'Comportamiento de $\log(V/V_0)$ frente a la distancia')
ax_ajuste.legend()
ax_ajuste.grid(True, linestyle='--')
plt.show()

# Desviaciones
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y4, V4, color="blue", label=f'Puntos de datos')
ax.plot(y4, np.zeros_like(V4), color='red',label=r'V = 0')

ax.set_xlabel('y [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $x= L/2$')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()