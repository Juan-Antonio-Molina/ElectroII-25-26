from sympy import symbols, Matrix, sin, diag, Rational, Integer
from tensores_importantes import christoffel_symbols, riemann_tensor, ricci_tensor, ricci_scalar, pretty_print_tensors

# Ejemplo 21.10 del Hartle
t, x, y, z = symbols('t x y z')
coords = [t, x, y, z]
g = symbols('g')
f = 1 - g*x
schwarz = Matrix([[-f, 0,   0,         0],
                  [0,  1, 0,         0],
                  [0,  0,   1,      0],
                  [0,  0,   0,   1]])

Gamma_s = christoffel_symbols(schwarz, coords)
Riem_s = riemann_tensor(Gamma_s, coords)
Ricci_s = ricci_tensor(Riem_s)
R_s = ricci_scalar(Ricci_s, schwarz.inv())

print("\n\n=== MÉTRICA DE SCHWARZSCHILD (componentes simplificadas) ===")
pretty_print_tensors(coords, schwarz, Gamma_s, Riem_s, Ricci_s, R_s, max_display=20)
