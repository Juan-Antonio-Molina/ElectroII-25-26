# Solución directa
import numpy as np
np.set_printoptions(precision=5)

# Matriz del vector de potenciales conocidos
B = np.array([[100], [80], [40], [20]])

# Matriz A del sistema lineal Ax = B para los 6 nodos internos (Va...Vf)
# Obtenida de la "molécula de 5 puntos" aplicada a cada nodo
A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
])

print("Determinante de A ({:.4f})".format(np.linalg.det(A)))

# Inversión de la matriz A
A_inv = np.linalg.inv(A)

# Cálculo del vector de potenciales X = A_inv * B
X = np.dot(A_inv, B)
print("X*={}".format(np.transpose(X)))