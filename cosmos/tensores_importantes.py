"""
gr_tensors.py

Funciones para calcular:
 - Símbolos de Christoffel (tipo 1/2): Gamma^rho_{mu nu}
 - Tensor de Riemann: R^rho_{ sigma mu nu}
 - Tensor de Ricci: R_{sigma nu}
 - Ricci escalar: R

Uso: ver el bloque if __name__ == "__main__" al final para ejemplos.
"""

from sympy import symbols, Matrix, diff, simplify, zeros, pprint, sin
import numpy as np

def christoffel_symbols(metric, coords):
    """
    metric: sympy Matrix 4x4 (g_{mu nu})
    coords: list/tuple of 4 sympy symbols [x0, x1, x2, x3]
    Returns: Gamma (4-index object) as a 3D list Gamma[r][m][n] = Gamma^r_{mn}
    """
    dim = 4
    g = metric
    g_inv = simplify(g.inv())
    Gamma = [[[0 for _ in range(dim)] for __ in range(dim)] for ___ in range(dim)]

    for r in range(dim):
        for m in range(dim):
            for n in range(dim):
                expr = 0
                for s in range(dim):
                    expr += g_inv[r, s] * (diff(g[s, m], coords[n]) +
                                           diff(g[s, n], coords[m]) -
                                           diff(g[m, n], coords[s]))
                expr = simplify(expr/2)
                Gamma[r][m][n] = expr
    return Gamma

def riemann_tensor(Gamma, coords):
    """
    Gamma: output from christoffel_symbols
    coords: list of coordinate symbols
    Returns: Riemann R^rho_{ sigma mu nu} as a 4D list R[rho][sigma][mu][nu]
    """
    dim = 4
    R = [[[[0 for _ in range(dim)] for __ in range(dim)] for ___ in range(dim)] for ____ in range(dim)]

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    term = diff(Gamma[rho][nu][sigma], coords[mu]) - diff(Gamma[rho][mu][sigma], coords[nu])
                    # + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma} - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}
                    add = 0
                    for lam in range(dim):
                        add += Gamma[rho][mu][lam]*Gamma[lam][nu][sigma] - Gamma[rho][nu][lam]*Gamma[lam][mu][sigma]
                    R[rho][sigma][mu][nu] = simplify(term + add)
    return R

def ricci_tensor(riemann):
    """
    riemann: R^rho_{ sigma mu nu}
    Returns: Ricci R_{sigma nu} = R^rho_{ sigma rho nu}
    """
    dim = 4
    Ric = [[0 for _ in range(dim)] for __ in range(dim)]
    for sigma in range(dim):
        for nu in range(dim):
            s = 0
            for rho in range(dim):
                s += riemann[rho][sigma][rho][nu]
            Ric[sigma][nu] = simplify(s)
    return Ric

def ricci_scalar(ricci, metric_inv):
    """
    ricci: R_{sigma nu}
    metric_inv: g^{mu nu} (inverse metric sympy Matrix)
    Returns scalar R = g^{sigma nu} R_{sigma nu}
    """
    dim = 4
    s = 0
    for a in range(dim):
        for b in range(dim):
            s += metric_inv[a, b] * ricci[a][b]
    return simplify(s)

def pretty_print_tensors(coords, metric, Gamma, Riemann, Ricci, Rscalar, max_display=10):
    print("Coordenadas:", coords)
    print("\nMétrica g_{μν}:")
    pprint(metric)

    print("\nSímbolos de Christoffel Γ^ρ_{μν} (algunos no nulos):")
    dim = 4
    count = 0
    for r in range(dim):
        for m in range(dim):
            for n in range(dim):
                if Gamma[r][m][n] != 0:
                    print(f"Gamma^{r}_{{{m}{n}}} =")
                    pprint(simplify(Gamma[r][m][n]))
                    count += 1
                    if count >= max_display:
                        print("... (más componentes no nulos omitidos)")
                        break
            if count >= max_display:
                break
        if count >= max_display:
            break

    print("\nComponentes de Ricci R_{μν} (no nulos):")
    for i in range(dim):
        for j in range(dim):
            if Ricci[i][j] != 0:
                print(f"R_{i}{j} =")
                pprint(Ricci[i][j])

    print("\nRicci escalar R =")
    pprint(Rscalar)

if __name__ == "__main__":
    # Ejemplo: coordenadas (t, r, theta, phi)
    t, r, th, ph = symbols('t r th ph')
    coords = [t, r, th, ph]

    # ---------------------------
    # Ejemplo 1: Minkowski (prueba)
    # signature (-,+,+,+)
    eta = Matrix([[-1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    Gamma_eta = christoffel_symbols(eta, coords)
    Riem_eta = riemann_tensor(Gamma_eta, coords)
    Ricci_eta = ricci_tensor(Riem_eta)
    R_eta = ricci_scalar(Ricci_eta, eta.inv())

    print("=== MÉTRICA DE MINKOWSKI (comprobación) ===")
    pretty_print_tensors(coords, eta, Gamma_eta, Riem_eta, Ricci_eta, R_eta)

    # ---------------------------
    # Ejemplo 2: Schwarzschild (en unidades G=c=1)
    # ds^2 = -(1-2M/r) dt^2 + (1-2M/r)^{-1} dr^2 + r^2 dΩ^2
    M = symbols('M')
    f = 1 - 2*M/r
    schwarz = Matrix([[-f, 0,   0,         0],
                      [0,  1/f, 0,         0],
                      [0,  0,   r**2,      0],
                      [0,  0,   0,   r**2*(sin(th))**2]])
    # Necesitamos importar sin para esta parte, o definirlo arriba; usar sympy.sin
    from sympy import sin
    schwarz[3,3] = r**2 * sin(th)**2

    Gamma_s = christoffel_symbols(schwarz, coords)
    Riem_s = riemann_tensor(Gamma_s, coords)
    Ricci_s = ricci_tensor(Riem_s)
    R_s = ricci_scalar(Ricci_s, schwarz.inv())

    print("\n\n=== MÉTRICA DE SCHWARZSCHILD (componentes simplificadas) ===")
    pretty_print_tensors(coords, schwarz, Gamma_s, Riem_s, Ricci_s, R_s, max_display=20)


