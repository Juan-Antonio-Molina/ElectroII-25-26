"""
ejer3.py - Ejercicio 3: Calcular el potencial en una línea

Objetivos:
-

Created on Sun Sept 21 14:07:23 2025
@autors: juanan, manuelpi
"""

from em2um import SemiCajaPotencial
import numpy as np
import matplotlib.pyplot as plt


V0 = 2 # Por ejemplo
L = 2 # Por ejemplo
Ny = 50 # Numero de puntos de la particion

NumModos = np.arange(1, Ny/2, 2) # M < N/2
x = L/2*np.ones(Ny)
y = np.linspace(0,2*L,Ny)

# Graficamos los potenciales a lo largo de la recta x = L/2
cmap = plt.cm.get_cmap('viridis') # paleta de colores jeje
colores_norm = np.linspace(0, 1, len(NumModos))

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(NumModos)):
    V_i = SemiCajaPotencial(NumModos[i], V0, L, x, y)
    ax.plot(y, V_i, color=cmap(colores_norm[i]), label=f'Modo {NumModos[i]}') # label dinamico

#Graficar la Función Exponencial e^(-y)
ax.plot(y, V0*np.exp((-1)*y), color='red',label=r'$V_0 \cdot \exp(-y)$')
ax.set_xlabel('y [m]')
ax.set_ylabel('V [V]')
ax.set_title(r'Potencial a lo largo de la recta $x= L/2$ para diferentes modos')
ax.legend(title='Número de Modo', loc='best')  # Mejorar la leyenda
ax.grid(True, linestyle='--')
plt.show()

# Determine la distancia a la que el potencial se vuelve despreciable
# Consideramos potencial despreciable cuando se cumple la condiciton de "Conductor a tierra"
tolerancia = V0 * 10**(-2) # PONGO 2 PORQUE CON 3 NO LLEGA KNFKJNDQKJBDQKJNWDKJWFQKJDNQKJD
for i in range(Ny):
    if V_i[i] < tolerancia:
        print("La distancia a la que el potencial se vuelve despreciable es ", y[i])
        break


# Explicar el comportamiento: patata



