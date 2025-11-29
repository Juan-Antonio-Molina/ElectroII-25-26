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
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label='Solución numérica')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(r'Visualización de la trayectoria con $\vec{B}=\vec{B}_{botella}$')
plt.show()

# Ejercicio 3
pi = np.pi
q, m, B0 = 1, 1, 1
x0, y0, z0 = 1, 1, 1
vx0, vy0, vz0 = 8, 2, 0
T = 2 * pi * m / (q * B0)

u0 = [x0, y0, z0, vx0, vy0, vz0]
dt = 0.01;
t = np.arange(0, T, dt)
sol = odeint(Lorentz, u0, t, args=(q, m, B0))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sol[0, 0], sol[0, 1], sol[0, 2], marker='o')
ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], marker='o', color='r')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label='Solución numérica')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(r'Visualización de la trayectoria con $\vec{B}=\vec{B}_{botella}$')
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
slope = []

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

    slope.append(m*mod_v_perp/q)

    #if np.abs(phi[it] - phi[0]) < tol_phi:
        #paso.append(it)



# Hacemos lo análogo para el radio de giro:
fig_radio, ax_r1 = plt.subplots(figsize=(8, 6))
ax_r1.plot(t,R_1, color='cyan', label=r'$R_g^{(1)}$')

ax_r1.set_xlabel('Tiempo [s]')
ax_r1.set_ylabel('Radio de giro [m]')
ax_r1.legend()
ax_r1.set_title('Variación del radio de giro')
ax_r1.set_ylim(R_1[0]-0.1, R_1[0]+0.1)
ax_r1.legend()
plt.show()

# Hacemos lo análogo para el radio de giro:
fig_radio, ax_r1 = plt.subplots(figsize=(8, 6))
ax_r1.plot(t,R_2, color='cyan', label=r'$R_g^{(2)}$')

ax_r1.set_xlabel('Tiempo [s]')
ax_r1.set_ylabel('Radio de giro [m]')
ax_r1.legend()
ax_r1.set_title('Variación del radio de giro')
ax_r1.set_ylim(R_2[0]-0.1, R_2[0]+0.1)
ax_r1.legend()
plt.show()

# Vamos a observar la variación del módulo del
# campo magnético a lo largo de la trayectoria:
fig_campo, ax_b = plt.subplots(figsize=(8, 6))
ax_b.plot(r[:,2],lista_modB, color='orange', label='Módulo B')

ax_b.set_xlabel('Posición z [m]')
ax_b.set_ylabel('Campo Magnético B [T]')
ax_b.legend()
ax_b.set_title('Variación del campo magnético')
ax_b.legend()
plt.show()

# Hacemos lo análogo para el radio de giro:
fig_radio, ax_r1 = plt.subplots(figsize=(8, 6))
ax_r1.plot(r[:,2],R_1, color='pink', label='Radio de giro')

ax_r1.set_xlabel('Posición z [m]')
ax_r1.set_ylabel('Radio de giro [m]')
ax_r1.legend()
ax_r1.set_title('Variación del radio de giro')
ax_r1.legend()
plt.show()


#Análisis del error
err_Rg = np.max(np.abs(R_1 - R_1[0]))
err_kpara = np.max(np.abs(k_para - k_para[0]))
err_kperp = np.max(np.abs(k_perp - k_perp[0]))
err_u = np.max(np.abs(u_1 - u_1[0]))
print(f"La máxima desviación encontrada para Rg es {err_Rg}, siendo la inicial Rg = {R_1[0]}")
print(f"La máxima desviación encontrada para K perpendicular es {err_kperp}, siendo la inicial K_perp = {k_perp[0]}")
print(f"La máxima desviación encontrada para K paralela es {err_kpara}, siendo la inicial K_para = {k_para[0]}")
print(f"La máxima desviación encontrada para u es {err_u}, siendo la inicial u = {u_1[0]}")

# Para comprobar la conservación de estas magnitudes también podemos,
# simplemente, graficarlas a lo largo de la trayectoria.
# En concreto, las graficaremos frente a la componente z
# de la posición de la partícula:

# RADIO DE GIRO:
fig_rg, ax_rg = plt.subplots(figsize=(8, 6))
ax_rg.plot(r[:,2],R_1, color='green', label='Radio de giro')

ax_rg.set_xlabel('Posición z [m]')
ax_rg.set_ylabel('Radio de giro [m]')
ax_rg.legend()
ax_rg.set_title('Conservación del radio de giro')
ax_rg.legend()
plt.show()

# K PARALELA
fig_kpa, ax_kpa = plt.subplots(figsize=(8, 6))
ax_kpa.plot(r[:,2],k_para, color='green', label='K paralela')

ax_kpa.set_xlabel('Posición z [m]')
ax_kpa.set_ylabel(r'$K_{\parallel}$ [J]')
ax_kpa.legend()
ax_kpa.set_title('Conservación de la energía cinética paralela')
ax_kpa.legend()
plt.show()

# K PErPENDICULAR
fig_kpe, ax_kpe = plt.subplots(figsize=(8, 6))
ax_kpe.plot(r[:,2],k_perp, color='green', label='K perpendicular')

ax_kpe.set_xlabel('Posición z [m]')
ax_kpe.set_ylabel(r'$K_{\perp}$ [J]')
ax_kpe.legend()
ax_kpe.set_title('Conservación de la energía cinética perpendicular')
ax_kpe.legend()
plt.show()

# MOMENTO MAGNÉTICO
fig_mu, ax_mu = plt.subplots(figsize=(8, 6))
ax_mu.plot(r[:,2],u_1, color='green', label='Momento magnético')

ax_mu.set_xlabel('Posición z [m]')
ax_mu.set_ylabel(r'$\mu$ [J/T]')
ax_mu.legend()
ax_mu.set_title('Conservación del momento magnético')
ax_mu.set_ylim(u_1[0]-0.1, u_1[0]+0.1)
ax_mu.legend()
plt.show()

# K TOTAL
fig_k, ax_k = plt.subplots(figsize=(8, 6))
ax_k.plot(r[:,2],k_perp+k_para, color='green', label='K total')

ax_k.set_xlabel('Posición z [m]')
ax_k.set_ylabel(r'$K_{total}$ [J]')
ax_k.legend()
ax_k.set_title('Conservación de la energía cinética (total)')
ax_k.set_ylim(k_perp[0]+k_para[0]-0.1, k_perp[0]+k_para[0]+0.1)
ax_k.legend()
plt.show()


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

for jt in range(len(pos_x)-1):
    if np.abs(pos_x[jt+1]-pos_x[0]) < tol_x:
        paso.append(jt+1)
        indices.append(jt+1)

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
ax_paso.set_ylabel('Paso [m]')
ax_paso.legend()
ax_paso.set_title('Pasos a lo largo de la trayectoria')
ax_paso.legend()
plt.show()

# También podríamos haber visto cuándo la componente x
# de la posición cambia de signo
paso_new = []
for it in range(len(v)-1):
    if r[it,0]*r[it+1,0] < 0:
        paso_new.append(r[it+1,2])

pasos_new = []
for it in range(len(paso_new)-1):
    pasos_new.append(np.abs(paso_new[it+1]-paso_new[it]))

fig_paso, ax_paso = plt.subplots(figsize=(8, 6))
ax_paso.plot(pasos_new, color='purple', label='Pasos')

ax_paso.set_xlabel('Número de vueltas')
ax_paso.set_ylabel('Paso [m]')
ax_paso.legend()
ax_paso.set_title('Pasos a lo largo de la trayectoria')
ax_paso.legend()
plt.show()

# Para observar el paso también podemos, simplemente,
# graficar la posición x y la z a lo largo de la trayectoria:
fig_pa, ax_pa = plt.subplots(figsize=(8, 6))
ax_pa.plot(r[:,2],r[:,0], color='green', label='Componente x')

ax_pa.set_xlabel('Posición z [m]')
ax_pa.set_ylabel('Posición x [m]')
ax_pa.legend()
ax_pa.set_title('Variación del paso')
ax_pa.legend()
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

ax_ajuste.set_xlabel(r'$1/B$ [$T^{-1}$]')
ax_ajuste.set_ylabel(r'$R_g$ [m]')
ax_ajuste.legend()
ax_ajuste.set_title(r'Radio de giro respecto $1/B$')
ax_ajuste.legend()
plt.show()

# Ahora vamos a analizar el fenómeno de reflexión magnética.
# Para ello, tomaremos la componente z (paralela al eje de rotación) de la
# velocidad de nuestra partícula y veremos cuándo cambia de signo:
v_z = v[:,2]
fig_refl, ax_refl = plt.subplots(figsize=(8, 6))
ax_refl.plot(t, v_z, color='brown', label='Velocidad en la dirección z')

ax_refl.set_xlabel(r't [s]')
ax_refl.set_ylabel(r'$v_z$ [m/s]')
ax_refl.legend()
ax_refl.set_title('Fenómeno de reflexión magnética')
ax_refl.legend()
plt.show()


