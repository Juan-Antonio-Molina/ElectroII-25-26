"""
ejer2.py - Ejercicio 2: Carga puntual y esfera conductora aislada

@autors: juanan, manuelpi
"""
import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga
from ejer1 import comp_tang, comp_normal

# Considerando la misma situación
# que en el ejercicio 1:
#Datos iniciales
a = 1.0
d = 3.0
q1 = 1.0e-9; r1 = [d, 0.0]

# Esta vez, la esfera no estará a tierra,
# pero al ser conductora, su superficie
# será una equipotencial.
# También debemos tener en cuenta que, en el
# interior de la esfera, el campo eléctrico es
# idénticamente nulo.
# Por tanto, necesitamos dos cargas imágenes:
# una negativa para hacer la superficie equipotencial,
# de manera análoga al caso anterior,
# y otra positiva en el centro geométrico de la
# esfera para mantener la condición de Q_TOT=0
# y de V=cte en la superficie de la esfera.

q3 = -q2; r3 = [0.0, 0.0]

# Las almacenamos
vec_cargas = [q1, q2, q3]
vec_posiciones = [r1, r2, r3]

# Graficamos el potencial en todo el espacio,
# y en particular lo calculamos para la superficie
# de la esfera.

L = 4.0 # Límite del eje x/y para el gráfico
N = 100 # Número de puntos PONER PARES PORFAVOR GRACIAS

x = np.linspace(-L+1, L+1, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

V = np.zeros_like(X)
Ex = np.zeros_like(X)
Ey = np.zeros_like(X)

for i in range(len(vec_cargas)):
  Ex_i, Ey_i = campo_carga(vec_cargas[i], vec_posiciones[i], X, Y)
  Ex += Ex_i
  Ey += Ey_i
  V += potencial_carga(vec_cargas[i], vec_posiciones[i], X, Y)

# Para calcular el potencial en la superficie:
theta = np.linspace(0, 2*np.pi, 100) ## Vector de ángulos
## Vectores de posiciones, de potencial y de campo:
x2 = a*np.cos(theta)
y2 = a*np.sin(theta)

V_super = np.zeros_like(x2)
Ex_super = np.zeros_like(x2)
Ey_super = np.zeros_like(x2)

for i in range(len(vec_cargas)):
  Ex_i, Ey_i = campo_carga(vec_cargas[i], vec_posiciones[i], x2, y2)
  Ex_super += Ex_i
  Ey_super += Ey_i
  V_super += potencial_carga(vec_cargas[i], vec_posiciones[i], x2, y2)

# Calculamos y visualizamos también el campo eléctrico,
# verificamos que es perpendicular a la superficie
# de la esfera, y calculamos su valor en r=[a/2,a/2]

# Para verificar la perpendicularidad, procedemos
# como en la primera práctica:
E_super = [Ex_super, Ey_super]
E_normal_super = comp_normal(E_super, x2, y2)
E_tan_super = comp_tang(E_super, x2, y2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(theta, E_normal_super, label = 'Normal')
ax.plot(theta, E_tan_super, label = 'Tangencial')

ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel(r'E [N/C]')
ax.set_title('Componentes del campo eléctrico en la superficie de la esfera')
ax.legend()
ax.grid(True, linestyle='--')
plt.show()

# Calculamos la densidad de carga superficial en función
# del ángulo y la comparamos con la obtenida
# en el ejercicio1.

# Finalmente, calculamos la carga total inducida
# en la esfera mediante la integración de la densidad
# de carga a lo largo de su superficie.
