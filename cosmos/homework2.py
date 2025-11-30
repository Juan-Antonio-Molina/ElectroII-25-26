import numpy as np
import matplotlib.pyplot as plt
from math import isfinite

# Parámetros del problema
Omega_mat = 0.3
Omega_X   = 0.7
Omega_rad = 1e-4
Omega_k   = 0.0
H0_inv = 14.0  # H0^-1 ~ 14 Gyr


# Funciones cosmológicas
def hubble_function(a_bar, w):
    """
    Calcula la cantidad H(t)/H_0 de la primera ecuación de
    Friedmann en función de las omegas.
    - a_bar: float, factor de escala adimensional dado por a(t)/a(t_0)
    - w: float, ecuación de estado p = w * rho * c^2
    """
    return np.sqrt(Omega_mat*a_bar**(-3) + Omega_rad*a_bar**(-4) + Omega_k*a_bar**(-2) + Omega_X*a_bar**(-3*(1+w)))

def da_dt(a_bar, w):
    """
    Sabemos entonces que como H(t)/H_0 = a_bar(t)'/a_bar(t) * 1/H_0 = hubble_function().
    Esta funcion devuelve a_bar(t)'/H_0 = hubble_function() a_bar(t)
    - a_bar: float, factor de escala adimensional dado por a(t)/a(t_0)
    - w: float, ecuación de estado p = w * rho * c^2
    """
    return a_bar * hubble_function(a_bar, w)

# Aproximacion numérica
def integrate_a(a0, t0, t_final, w, dt):
    """
    Se invoca la función da_dt() = a_bar(t)'/H_0 = da_bar/dt * 1/H_0.
    Nos interesa saber dt = da_bar/H_0, por lo que integramos con respecto al tiempo
    Empleamos runge-kutta de 4 pasos
    - a0: float, factor de escala adimensional inicial
    - t0: float, tiempo inicial
    - t_final: float, tiempo final
    - w: float, ecuación de estado p = w * rho * c^2
    - dt: float, separación de la discretización
    """
    t = t0
    a = a0
    ts = [t]
    asol = [a]

    max_steps = int(abs((t_final - t0)/dt)) + 5
    sign = np.sign(t_final - t0) if t_final != 0 else 1

    for step in range(max_steps):
        if (t - t_final)*sign >= 0:
            break

        h = dt
        k1 = da_dt(a, w)
        k2 = da_dt(a + 0.5*h*k1, w)
        k3 = da_dt(a + 0.5*h*k2, w)
        k4 = da_dt(a + h*k3, w)
        a = a + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        ts.append(t)
        asol.append(a)

        if not isfinite(a) or a <= 0 or a > 1e8:
            break

    return np.array(ts), np.array(asol)


# Condiciones iniciales
t0 = 0.0
a0 = 1.0     # en el presente a(t_0)/a_0 = 1

# Paso
dt_forward  = 1e-3
dt_backward = -1e-3

# Caso 1: w = -1
w1 = -1.0
t_past1, a_past1 = integrate_a(a0, t0, -5.0, w1, dt_backward)
t_future1, a_future1 = integrate_a(a0, t0, 5.0, w1, dt_forward)

plt.figure(figsize=(7, 4))
plt.plot(np.concatenate([t_past1, t_future1]),
         np.concatenate([a_past1, a_future1]))
#plt.yscale("log")
plt.xlabel(r"t [1/H_0]")
plt.ylabel(r"a(t)/a_0")
plt.title(r"Evolución temporal de a(t)/a_0 para  w = -1")
plt.grid(True)
plt.tight_layout()
plt.show()

# Caso 2: w = -3/2
w2 = -1.5
t_past2, a_past2 = integrate_a(a0, t0, -2.0, w2, dt_backward)
t_future2, a_future2 = integrate_a(a0, t0, 2.0, w2, dt_forward)

plt.figure(figsize=(7, 4))
plt.plot(np.concatenate([t_past2, t_future2]),
         np.concatenate([a_past2, a_future2]))
plt.yscale("log")
plt.xlabel(r"t [1/H_0]")
plt.ylabel("a(t)/a_0 (escala logarítmica)")
plt.title(r"Evolución temporal de a(t)/a_0 para w = -3/2 (phantom)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Cálculo del Big Rip
big_rip_time = None
for tval, aval in zip(t_future2, a_future2):
    if aval > 1e6:
        big_rip_time = tval
        break

if big_rip_time is not None:
    print("\n=== BIG RIP DETECTADO ===")
    print("Tiempo hasta Big Rip: {:.3f} H0^-1".format(big_rip_time))
    print("Equivalente: {:.2f} Gyr".format(big_rip_time * H0_inv))
else:
    print("No ocurre Big Rip en el intervalo simulado.")
