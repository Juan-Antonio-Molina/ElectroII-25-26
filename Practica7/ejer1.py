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


# %%
pi = np.pi
q, m, B0 = 1, 1, 1
x0, y0, z0 = 0, 2, 0
vx0, vy0, vz0 = 2, 0, 2
T = 2 * pi * m / (q * B0)

u0 = [x0, y0, z0, vx0, vy0, vz0]
dt = 0.01;
t = np.arange(0, 2 * T, dt)
sol = odeint(Lorentz, u0, t, args=(q, m, B0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sol[0, 0], sol[0, 1], sol[0, 2], marker='o')
ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], marker='o', color='r')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])

# Analítica
w = 2*pi/T
Rg = 2
phi_0 = 0
x = 0 + Rg*np.sin(w*t + phi_0)
y = 0 + Rg*np.cos(w*t + phi_0)
z = 0 + vz0*t

#Dibujamos la analitica
ax.plot(x, y, z, ':r')
plt.show()

# %%
r = sol[:, 0:3]
v = sol[:, 3:6]
dt = t[1] - t[0]
a = np.gradient(v, t, axis=0)  # usa diferencias centradas en el interior y forward/backward en los extremos.

k_para = np.zeros(len(v))
k_perp = np.zeros_like(k_para)
R_1 = np.zeros_like(k_para)
u_1 = np.zeros_like(k_para)


for it in range(len(v)):
    B = [0,0,B0]
    mod_B = np.sqrt(np.dot(B, B))
    mod_v = np.sqrt(np.dot(v[it, :], v[it, :]))
    u_para = B / mod_B
    v_para = np.dot(v[it, :], u_para) * u_para
    v_perp = v[it, :] - v_para
    mod_v_perp = np.sqrt(np.dot(v_perp, v_perp))

    k_para[it] = 0.5 * m * np.dot(v_para, v_para)
    k_perp[it] = 0.5 * m * np.dot(v_perp, v_perp)

    u_tang = v[it, :] / mod_v
    a_tang = np.dot(a[it, :], u_tang) * u_tang
    a_norm = a[it, :] - a_tang
    mod_a_norm = np.sqrt(np.dot(a_norm, a_norm))
    R_1[it] = m * mod_v_perp / (q * mod_B)
    u_1[it] = k_perp[it] / mod_B

#Analisis del error
err_Rg = np.max(np.abs(R_1 - R_1[0]))
err_kpara = np.max(np.abs(k_para - k_para[0]))
err_kperp = np.max(np.abs(k_perp - k_perp[0]))
err_u = np.max(np.abs(u_1 - u_1[0]))
print(f"La máxima desviación encontrada para Rg es {err_Rg}, siendo la inicial Rg = {R_1[0]}")
print(f"La máxima desviación encontrada para K perpendicular es {err_kperp}, siendo la inicial K_perp = {k_perp[0]}")
print(f"La máxima desviación encontrada para K paralela es {err_kpara}, siendo la inicial K_para = {k_para[0]}")
print(f"La máxima desviación encontrada para u es {err_u}, siendo la inicial u = {u_1[0]}")
