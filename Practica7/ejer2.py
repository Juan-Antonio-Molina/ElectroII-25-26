import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


def Botella(x, y, z):
    return [-x * z / 3, -y * z / 3, (24 + z * z) / 3]


def Lorentz(u, t, q, m, B0):
    x, y, z, vx, vy, vz = u
    v = np.array([vx, vy, vz])
    B = Botella(x, y, z)
    F = q * np.cross(v, B)
    dvx_dt = F[0] / m
    dvy_dt = F[1] / m
    dvz_dt = F[2] / m
    dx_dt = vx
    dy_dt = vy
    dz_dt = vz
    return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]


# Ejercicio 2
pi = np.pi
q, m, B0 = 1, 1, 1
x0, y0, z0 = 0, 1, 0
vx0, vy0, vz0 = 8, 0, 0
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
plt.show()

# Ejercicio 3
pi = np.pi
q, m, B0 = 1, 1, 1
x0, y0, z0 = 0, 1, 0
vx0, vy0, vz0 = 8, 0, 1
T = 2 * pi * m / (q * B0)

u0 = [x0, y0, z0, vx0, vy0, vz0]
dt = 0.01;
t = np.arange(0, T, dt)
sol = odeint(Lorentz, u0, t, args=(q, m, B0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sol[0, 0], sol[0, 1], sol[0, 2], marker='o')
ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], marker='o', color='r')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
plt.show()

# Cálculo parámetros
r = sol[:, 0:3]
v = sol[:, 3:6]
dt = t[1] - t[0]
a = np.gradient(v, t, axis=0)  # usa diferencias centradas en el interior y forward/backward en los extremos.


# Parametros constantes
k_para = np.zeros(len(v))
k_perp = np.zeros_like(k_para)
R_1 = np.zeros_like(k_para)
R_2 = np.zeros_like(k_para)
u_1 = np.zeros_like(k_para)

# Campo magnético

lista_modB = np.zeros_like(k_para)
inverse_B = np.zeros_like(k_para)

for it in range(len(v)):
    B = Botella(r[it, 0], r[it, 1], r[it, 2])
    mod_B = np.sqrt(np.dot(B, B))
    lista_modB[it] = mod_B
    inverse_B[it] = 1 / mod_B
    mod_v = np.sqrt(np.dot(v[it, :], v[it, :]))
    u_para = B / mod_B
    v_para = np.dot(v[it, :], u_para) * u_para
    v_perp = v[it, :] - v_para
    mod_v_perp = np.sqrt(np.dot(v_perp, v_perp))

    k_para[it] = 0.5 * m * np.dot(v_para, v_para)
    k_perp[it] = 0.5 * m * np.dot(v_perp, v_perp)

    u_tang = v[it, :] / mod_v # Vectores de velocidad tangentes unitarios
    a_tang = np.dot(a[it, :], u_tang) * u_tang # Proyeccion aceleracion en v tangente
    a_norm = a[it, :] - a_tang # Proyeccion aceleracion en v perpendicular
    mod_a_norm = np.sqrt(np.dot(a_norm, a_norm)) # Modulo de la aceleracion normal

    R_1[it] = mod_v * mod_v / mod_a_norm
    R_2[it] = m * mod_v_perp / (q * mod_B)
    u_1[it] = k_perp[it] / mod_B

    #if np.abs(phi[it] - phi[0]) < tol_phi:
        #paso.append(it)

#Análisis del error
err_Rg = np.max(np.abs(R_1 - R_1[0]))
err_kpara = np.max(np.abs(k_para - k_para[0]))
err_kperp = np.max(np.abs(k_perp - k_perp[0]))
err_u = np.max(np.abs(u_1 - u_1[0]))
print(f"La máxima desviación encontrada para Rg es {err_Rg}, siendo la inicial Rg = {R_1[0]}")
print(f"La máxima desviación encontrada para K perpendicular es {err_kperp}, siendo la inicial K_perp = {k_perp[0]}")
print(f"La máxima desviación encontrada para K paralela es {err_kpara}, siendo la inicial K_para = {k_para[0]}")
print(f"La máxima desviación encontrada para u es {err_u}, siendo la inicial u = {u_1[0]}")


# Vemos el paso con la componente x de la posición,
# pues esta es inicialmente nula.
paso = []
tol_x = 3e-2
pos_x = r[:,0]

# Voy a escribir los índices para que al hacer debug
# pueda comprobar que son vueltas distintas y que no
# esté cogiendo puntos cercanos:
indices = []
# Así también podemos ir ajustando la tolerancia

for jt in range(len(pos_x)):
    if np.abs(pos_x[jt]) < tol_x:
        paso.append(jt)
        indices.append(jt)

# Guardamos las posiciones z correspondientes a cada
# vez que se anula la componente x:
for it in range(len(paso)):
    paso[it] = r[it,2]

pasos = []
for it in range(len(paso)-1):
    pasos.append(paso[it+1]-paso[it])

fig_paso, ax_paso = plt.subplots(figsize=(8, 6))
ax_paso.plot(pasos, color='purple', label='Pasos')

ax_paso.set_xlabel('Número de vueltas')
ax_paso.set_ylabel('Paso [uds]')
ax_paso.legend()
ax_paso.set_title('Pasos a lo largo de la trayectoria')
ax_paso.legend()
plt.show()



# Relación entre la intensidad de B y el radio de giro:

res = stats.linregress(inverse_B, R_1) # Ajuste lineal
print(f"Pendiente = {res.slope:.3f} err: {res.stderr:.3f}")
print(f"Interseccion = {res.intercept:.3f} err: {res.
      intercept_stderr:.3f}")
print(f"Coef. correlación Pearson r = {res.rvalue:.3f}")
print(f"R^2 (calidad ajuste) = {res.rvalue**2:.3f}")

# Dibujar los puntos y el ajuste en un nuevo subplot
y_ajuste = res.slope * inverse_B + res.intercept
fig_ajuste, ax_ajuste = plt.subplots(figsize=(8, 6))
ax_ajuste.scatter(inverse_B, R_1, label='Puntos de datos')
ax_ajuste.plot(inverse_B, y_ajuste, color='red', label='Ajuste Lineal')

ax_ajuste.set_xlabel(r'$1/B$ [uds]')
ax_ajuste.set_ylabel(r'$R_g$ [uds]')
ax_ajuste.legend()
ax_ajuste.set_title(r'Radio de giro respecto $1/B$')
ax_ajuste.legend()
plt.show()

# Ahora vamos a analizar el fenómeno de reflexión magnética.
# Para ello, tomaremos la componente z (paralela al eje de rotación) de la
# velocidad de nuestra partícula y veremos cuándo cambia de signo:
v_z = v[:,2]
fig_refl, ax_refl = plt.subplots(figsize=(8, 6))
ax_refl.plot(t, v_z, color='red', label='Velocidad en la dirección z')

ax_refl.set_xlabel(r't [uds]')
ax_refl.set_ylabel(r'$v_z$ [uds]')
ax_refl.legend()
ax_refl.set_title('Fenómeno de reflexión magnética')
ax_refl.legend()
plt.show()


