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
Nx = 200
NumModos = 60
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

V2 = SemiCajaPotencial(NumModos, V0, L, x2, y)

if np.max(np.abs(V1)) < tolerancia:
    print(" La condición de V(0,y) = 0 se cumple para el número de modos dado")
else:
    print(" La condición de V(0,y) = 0 no se cumple para el número de modos dado")


# V(x,0)=V0 cuando 0x está entre 0 y L:
x3 = np.linspace(0.05 * L, 0.95 * L, Nx)
y3 = np.zeros_like(x3)
V3 = SemiCajaPotencial(NumModos, V0, L, x3, y3)

if np.max(np.abs(V3/V_teo - 1)) <= 0.01:
    print(" La condición de V(x,0) = V0 se cumple para el número de modos dado")
else:
    print(" La condición de V(x,0) = V0 no se cumple para el número de modos dado")

# V tiende a 0 cuando "y" tiende a infinito
y4 = np.linspace(3*L, 10*L, Nx)
x4 = L/2*np.ones(Nx)
V4 = SemiCajaPotencial(NumModos, V0, L, x4, y4) / V0

fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(y4, V4, label='Puntos de datos')
ax_ajuste.plot(y4, V4, color='red', label='funcion')

ax_ajuste.set_xlabel('y [m]')
ax_ajuste.set_ylabel(r'$V/V_0$')
ax_ajuste.set_title(r'Ajuste lineal de $V/V_0$ frente a la distancia')
ax_ajuste.legend()
ax_ajuste.grid(True, linestyle='--')
plt.show()

# Comprobemos que la caida exponencial
y_exp = np.exp((-1)*y4)
res = stats.linregress(y_exp, V4) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")

# Dibujar los puntos y el ajuste en un nuevo subplot
y_ajuste = res.slope * y_exp + res.intercept
fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(y_exp, V4, label='Puntos de datos')
ax_ajuste.plot(y_exp, y_ajuste, color='red', label='Ajuste Lineal')

ax_ajuste.set_xlabel(r'$\exp(x)$')
ax_ajuste.set_ylabel(r'$V/V_0$')
ax_ajuste.set_title(r'Ajuste lineal de $V/V_0$ frente a la distancia')
ax_ajuste.legend()
ax_ajuste.grid(True, linestyle='--')
plt.show()

# Cuantificar. ¿Cómo almaceno el error de la condición de infinito?...
