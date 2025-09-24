"""
ejer2.py - Ejercicio 2: Componentes del campo eléctrico

Objetivos:
- Para una carga puntual centrada en el origen, calcular las componentes normal y
tangencial del campo eléctrico en puntos de un círculo de radio unidad.
- Representar gráficamente estas componentes en función del ángulo.
- Calcular y analizar el error respecto a los valores teóricos.
- Explicar la relación con el teorema de Gauss.

Created on Sun Sept 21 14:07:23 2025
@autors: juanan, manuelpi
"""
import numpy as np
import matplotlib.pyplot as plt
from em2um import potencial_carga, campo_carga, k_0
from scipy import stats

# Establecemos el valor de la carga y su posición:
q = 1.6e-19; r = [0, 0]

def comp_normal(campo, x, y):
    """
    Calcula la componente normal de un campo vectorial
    cualquiera en un punto (x,y) del plano.

    - campo: campo que queremos manipular
    - x: coordenada x del punto
    - y: coordenada y del punto
    """
    # Realizaremos el producto escalar del vector campo
    # con el vector normal en el punto (x,y):
    dist = np.sqrt(x**2 + y**2)
    campo_n = campo[0] * x/dist + campo[1] * y/dist
    return campo_n

def comp_tang(campo, x, y):
    """
    Calcula la componente tangencial de un campo vectorial
    cualquiera en un punto (x,y) del plano.

    - campo: campo que queremos manipular
    - x: coordenada x del punto
    - y: coordenada y del punto
    """
    # Realizaremos el producto escalar del vector campo
    # con el vector normal en el punto (x,y):
    dist = np.sqrt(x**2 + y**2)
    campo_tg = campo[0] * (-y)/dist + campo[1] * x/dist
    return campo_tg

# Para representar estas componentes en función del ángulo,
# tendremos en cuenta que las posiciones en el círculo unidad
# vienen dadas por r = (cos(theta), sin(theta))
pi = np.pi
angles = np.linspace(0, 2*pi, 50)
E_normal = np.zeros_like(angles)
E_tangencial = np.zeros_like(angles)
radio = 1

# Ahora calcularemos estas componentes para la carga escogida:
for i in range(len(angles)):
    x = radio * np.cos(angles[i])
    y = radio * np.sin(angles[i])
    E = campo_carga(q, r, x, y)
    E_normal[i] = comp_normal(E, x, y)
    E_tangencial[i] = comp_tang(E, x, y)

plt.plot(angles, E_normal, label = 'Normal')
plt.plot(angles, E_tangencial, label = 'Tangencial')
plt.legend(loc='best')
plt.xlabel('radianes')
plt.ylabel('N/C')
plt.title('Componentes del campo eléctrico de una carga puntual')
plt.show()

# Para hallar los errores, compararemos los resultados numéricos
# obtenidos con En = kq, Etg = 0:
e_abs_n = np.abs(E_normal - k_0 * q)
e_abs_tg = np.abs(E_tangencial - 0)
e_rel_n = e_abs_n / np.abs(k_0 * q)

err_promedio_n = sum(e_rel_n)/len(e_rel_n)
err_promedio_n_abs = sum(e_abs_n)/len(e_abs_n)
err_promedio_tg = sum(e_abs_tg)/len(e_abs_tg)
print('El error promedio (relativo) en la componente normal es: ', err_promedio_n)
print('El error promedio (absoluto) en la componente normal es: ', err_promedio_n_abs)
print('El error promedio (absoluto) en la componente tangencial es: ', err_promedio_tg)

plt.plot(angles, e_abs_n, label = 'Error Absoluto (Normal)')
plt.plot(angles, e_rel_n, label = 'Error Relativo (Normal)')
plt.plot(angles, e_abs_tg, label = 'Error Absoluto (Tangencial)')
plt.legend(loc='best')
plt.xlabel('radianes')
plt.title('Errores en función del ángulo')
plt.show()

print('El épsilon de la máquina es: ', np.finfo(float).eps)



