# Solución directa
import numpy as np
np.set_printoptions(precision=5)

# Matriz del vector de potenciales conocidos
delta = 1e-2 #m
densidad = 50e-9 #C/m^3
e_0 = 8.8541878188e-12  # permitividad vacio
d = - delta**2 * densidad/e_0
B = np.array([[d], [d], [d], [d]])

# Matriz A del sistema lineal Ax = B para los 6 nodos internos (Va...Vf)
# Obtenida de la "molécula de 5 puntos" aplicada a cada nodo
A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-2, 0, 4, -1],
    [0, -2, -1, 4]
])

print("Determinante de A ({:.4f})".format(np.linalg.det(A)))

# Inversión de la matriz A
A_inv = np.linalg.inv(A)

# Cálculo del vector de potenciales X = A_inv * B
X = np.dot(A_inv, B)
print("X*={}".format(np.transpose(X)))