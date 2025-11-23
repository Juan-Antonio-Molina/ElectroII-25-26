import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


def Lorentz(u, t, q, m, B0):
    x, y, z, vx, vy, vz = u
    v = np.array([vx, vy, vz])
    B = np.array([0,0,B0])
    F = q * np.cross(v, B)
    dvx_dt = F[0] / m
    dvy_dt = F[1] / m
    dvz_dt = F[2] / m
    dx_dt = vx
    dy_dt = vy
    dz_dt = vz
    return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]

# Condiciones iniciales
pi = np.pi
q, m, B0 = 1, 1, 1
x0, y0, z0 = 0, 2, 0
vx0, vy0, vz0 = 2, 0, 2


# Distribución temporal
T = 2 * pi * m / (q * B0) # periodo del ciclotron
u0 = [x0, y0, z0, vx0, vy0, vz0]
dt = 0.01;
t = np.arange(0, 2 * T, dt)

#Resolvemos numéricamente
sol = odeint(Lorentz, u0, t, args=(q, m, B0))

# Dibujamos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sol[0, 0], sol[0, 1], sol[0, 2], marker='o')
ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], marker='o', color='r')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label='Numérica')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'Visualización de la trayectoria con B = {B0}z')

# Solución analítica
w = 2*pi/T
Rg = 2
phi_0 = 0
x_0, y_0, z_0 = 0, 0, 0
x = x_0 + Rg*np.sin(w*t + phi_0)
y = y_0 + Rg*np.cos(w*t + phi_0)
z = z_0 + vz0*t
ax.plot(x, y, z, ':r', label='Analítica')
ax.legend(loc='best')
plt.show()

# Veamoslo tambien en el plano XY
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sol[0, 0], sol[0, 1], marker='o')
ax.scatter(sol[-1, 0], sol[-1, 1], marker='o', color='r')
ax.plot(sol[:, 0], sol[:, 1], color='darkcyan', label='Numérica')
ax.plot(x, y, ':r', label='Analítica')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Visualización de la trayectoria en el plano XY con B = {B0}z')
ax.legend()
ax.grid(True, linestyle='--')
ax.set_aspect('equal', adjustable='box')
plt.show()


# Verifiación numérica
r = sol[:, 0:3]
v = sol[:, 3:6]
dt = t[1] - t[0]

k_para = np.zeros(len(v))
k_perp = np.zeros_like(k_para)
R_1 = np.zeros_like(k_para)
u_1 = np.zeros_like(k_para)


for it in range(len(v)):
    # Preparativos
    B = [0,0,B0] # Campo magnetico
    mod_B = np.sqrt(np.dot(B, B)) # Modulo del campo magnetico
    mod_v = np.sqrt(np.dot(v[it, :], v[it, :])) # Modulo de la velocidad

    u_para = B / mod_B # Vector unitario paralelo a B
    v_para = np.dot(v[it, :], u_para) * u_para # Proyeccion de la velocidad en direccion de B
    v_perp = v[it, :] - v_para # Proyeccion de la velocidad perpendiular de B
    mod_v_perp = np.sqrt(np.dot(v_perp, v_perp)) # Modulo de la v perpendicular

    # Calculo de cantidades
    k_para[it] = 0.5 * m * np.dot(v_para, v_para) # K paralela
    k_perp[it] = 0.5 * m * np.dot(v_perp, v_perp) # K perpendicular
    u_1[it] = k_perp[it] / mod_B # Momento magnetico
    R_1[it] = m * mod_v_perp / (q * mod_B) # Radio de giro

#Analisis del error
err_Rg = np.max(np.abs(R_1 - R_1[0]))
err_kpara = np.max(np.abs(k_para - k_para[0]))
err_kperp = np.max(np.abs(k_perp - k_perp[0]))
err_u = np.max(np.abs(u_1 - u_1[0]))
print(f"La máxima desviación encontrada para Rg es {err_Rg}, siendo la inicial Rg = {R_1[0]}")
print(f"La máxima desviación encontrada para K perpendicular es {err_kperp}, siendo la inicial K_perp = {k_perp[0]}")
print(f"La máxima desviación encontrada para K paralela es {err_kpara}, siendo la inicial K_para = {k_para[0]}")
print(f"La máxima desviación encontrada para u es {err_u}, siendo la inicial u = {u_1[0]}")
