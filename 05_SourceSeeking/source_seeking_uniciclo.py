import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GaussianField:
    def __init__(self, x0=3.0, y0=4.0, A=15.0, sigma_x=5.0, sigma_y=5.0):
        self.x0 = x0
        self.y0 = y0
        self.A = A
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def evaluate(self, r):
        r = np.array(r)
        if r.ndim == 1:
            x, y = r[0], r[1]
        else:
            x, y = r[..., 0], r[..., 1]
        exponent = -(((x - self.x0)**2) / (2 * self.sigma_x**2) + 
                     ((y - self.y0)**2) / (2 * self.sigma_y**2))
        return self.A * np.exp(exponent)
    def gradient(self, r):
        """
        Calcula el vector gradiente analítico en la posición r = [x, y].
        Apuntará siempre en la dirección de máximo crecimiento (hacia la fuente).
        """
        r = np.array(r)
        if r.ndim == 1:
            x, y = r[0], r[1]
        else:
            x, y = r[..., 0], r[..., 1]
            
        sigma_val = self.evaluate(r)
        
        # Derivadas parciales analíticas
        grad_x = -((x - self.x0) / (self.sigma_x**2)) * sigma_val
        grad_y = -((y - self.y0) / (self.sigma_y**2)) * sigma_val
        
        if r.ndim == 1:
            return np.array([grad_x, grad_y])
        else:
            return np.stack([grad_x, grad_y], axis=-1)
# =====================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN (MODELO UNICICLO - SECCIÓN III-B)
# =====================================================================

campo = GaussianField(x0=3.0, y0=4.0, A=15.0, sigma_x=5.0, sigma_y=5.0)

N = 3
# Posiciones iniciales de los robots (formando un triángulo rígido inicial)
posiciones = np.array([
    [-3.0, -4.0],
    [-5.0, -4.0],
    [-6.0, -4.5]
])

# Orientaciones iniciales aleatorias para los 3 unisiclos (en radianes)
np.random.seed(42) # Para repetibilidad
thetas = np.random.uniform(-np.pi, np.pi, N)

# Parámetros físicos y de control del paper
u_r = 1.5         # Velocidad lineal CONSTANTE de los robots
k_gamma = 5.0     # Ganancia de control angular (debe ser suficientemente grande)
dt = 0.02         # Paso de tiempo pequeño para la estabilidad de la rotación
pasos = 600

# Inicialización de la variable virtual delta_i,* para cada robot (Ecuación 9)
# Inicialmente vale lo mismo que el error de ángulo real empaquetado en (-pi, pi]
r_c = np.mean(posiciones, axis=0)
x_i = posiciones - r_c
D = np.max(np.linalg.norm(x_i, axis=1))
lineas_campo = np.array([campo.evaluate(p) for p in posiciones])
L_hat = np.sum([lineas_campo[i] * x_i[i] for i in range(N)], axis=0) / (N * (D**2))

theta_d_inicial = np.arctan2(L_hat[1], L_hat[0])
# Error inicial real: delta_i = theta_i - theta_d
delta_real_inicial = thetas - theta_d_inicial
# Normalizar en (-pi, pi]
delta_virtual = np.arctan2(np.sin(delta_real_inicial), np.cos(delta_real_inicial))

# Guardar historial para la animación
historial_posiciones = np.zeros((pasos, N, 2))
historial_thetas = np.zeros((pasos, N))

theta_d_anterior = theta_d_inicial
errores_angulares_historial=[]
det_M_historial=[]

# =====================================================================
# BUCLE DE SIMULACIÓN (DINÁMICA DE UNISICLO CON VARIABLE VIRTUAL)
# =====================================================================

for t in range(pasos):
    historial_posiciones[t] = posiciones.copy()
    historial_thetas[t] = thetas.copy()
    
    # 1. Calcular Centroide y Dirección de Ascenso Estimada L_hat
    r_c = np.mean(posiciones, axis=0)
    x_i = posiciones - r_c
    D = np.max(np.linalg.norm(x_i, axis=1))
    lineas_campo = np.array([campo.evaluate(posiciones[i]) for i in range(N)])
    
    L_hat = np.zeros(2)
    for i in range(N):
        L_hat += lineas_campo[i] * x_i[i]
    L_hat /= (N * (D**2))
    
    # 1. Calcular el gradiente REAL en la posición del centroide r_c
    grad_real = campo.gradient(r_c)

    # 2. Calcular el error angular entre el estimado (L_hat) y el real (grad_real)
    norm_L = np.linalg.norm(L_hat)
    norm_grad = np.linalg.norm(grad_real)

    if norm_L > 1e-6 and norm_grad > 1e-6:
        # Producto escalar para obtener el coseno del ángulo entre ambos vectores
        cos_theta = np.dot(L_hat, grad_real) / (norm_L * norm_grad)
        # Asegurar límites por errores numéricos de precisión
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        error_angular = np.arccos(cos_theta)  # En radianes
        error_angular_grados = np.degrees(error_angular)
    else:
        error_angular_grados = 0.0

    # 3. Calcular la matriz de momentos M(X) para ver la degeneración (Sección II-B)
    M_matrix = np.zeros((2, 2))
    for i in range(N):
        # x_i es un vector fila (1, 2), lo convertimos a columna para el producto exterior
        x_col = x_i[i].reshape(2, 1)
        M_matrix += np.dot(x_col, x_col.T)

    # El determinante nos dice cómo de "abierto" está el triángulo
    # Si det_M -> 0, los robots están en línea recta (degenerados)
    det_M = np.linalg.det(M_matrix)

    # (Opcional) Puedes ir guardando estos datos en listas para graficarlos después
    errores_angulares_historial.append(error_angular_grados)
    det_M_historial.append(det_M)
    
    # NORMALIZACIÓN: Obtener el vector unitario del campo guía m_d
    norm_L = np.linalg.norm(L_hat)
    if norm_L > 1e-6:
        m_d = L_hat / norm_L
    else:
        m_d = np.array([1.0, 0.0]) # Evitar división por cero en el centro exacto
        
    theta_d = np.arctan2(m_d[1], m_d[0])
    
    # 3. Calcular el error angular real empaquetado en (-pi, pi]
    #    y actualizar la variable virtual delta_i,*
    for i in range(N):
        error_real = thetas[i] - theta_d
        # Mapear a (-pi, pi]
        delta_virtual[i] = np.arctan2(np.sin(error_real), np.cos(error_real))
    
    # 4. Ley de control angular (Acción rápida proporcional)
    omega_i = -k_gamma * delta_virtual
    
    # 5. Integración de la cinemática del Uniciclo
    thetas += omega_i * dt
    posiciones[:, 0] += u_r * np.cos(thetas) * dt  
    posiciones[:, 1] += u_r * np.sin(thetas) * dt

# =====================================================================
# ANIMACIÓN DE LOS RESULTADOS
# =====================================================================

fig, ax = plt.subplots(figsize=(8, 7))

# Fondo del campo
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = campo.evaluate(np.stack([X, Y], axis=-1))
contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
fig.colorbar(contour, label='Intensidad del campo $')

ax.scatter(campo.x0, campo.y0, color='red', marker='*', s=200, label='Fuente', zorder=5)

lineas_trayectoria = [ax.plot([], [], '--', lw=1.5, label=f'Robot {i}')[0] for i in range(N)]
# Usamos quiver para pintar los robots como flechas que muestran hacia dónde miran (\theta_i)
# CORRECCIÓN AQUÍ: Inicializamos quiver con el tamaño y datos del frame 0
quiver_robots = ax.quiver(
    historial_posiciones[0, :, 0], historial_posiciones[0, :, 1], 
    np.cos(historial_thetas[0]), np.sin(historial_thetas[0]), 
    color='black', scale=15, zorder=4
)
centro_masa_dot = ax.scatter([], [], color='white', marker='x', s=100, label='Centroide ($r_c$)', zorder=4)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('Posición X')
ax.set_ylabel('Posición Y')
ax.set_title('Simulación Source-Seeking: Modelo Uniciclo (Ecuación 9)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

def init():
    for linea in lineas_trayectoria:
        linea.set_data([], [])
    centro_masa_dot.set_offsets(np.empty((0, 2)))
    return lineas_trayectoria + [centro_masa_dot]

def animate(frame):
    pos_actuales = historial_posiciones[frame]
    thetas_actuales = historial_thetas[frame]
    
    # Trayectorias
    for i in range(N):
        lineas_trayectoria[i].set_data(historial_posiciones[:frame+1, i, 0], historial_posiciones[:frame+1, i, 1])
        
    # Actualizar flechas de los robots (posición y dirección de orientación)
    U = np.cos(thetas_actuales)
    V = np.sin(thetas_actuales)
    quiver_robots.set_offsets(pos_actuales)
    quiver_robots.set_UVC(U, V)
    
    # Centroide
    r_c_actual = np.mean(pos_actuales, axis=0)
    centro_masa_dot.set_offsets([r_c_actual])
    
    return lineas_trayectoria + [quiver_robots, centro_masa_dot]

ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=pasos, interval=25, blit=True, repeat=False
)

# =====================================================================
# GUARDAR COMO GIF
# =====================================================================
print("Guardando la animación como GIF... Por favor, espera.")

# fps = 1000 / interval. Como interval=25ms, 1000/25 = 40 frames por segundo.
# Puedes bajar los fps a 20 o 30 si quieres que el archivo pese menos.
ani.save(
    'source_seeking_uniciclo3.gif', 
    writer='pillow', 
    fps=40,
    dpi=100  # Ajusta la resolución/calidad. 100 es ideal para compartir por chat/web.
)

print("¡GIF guardado exitosamente como 'source_seeking_uniciclo.gif'!")
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(errores_angulares_historial, label='Error Angular (grados)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Error Angular (°)')
plt.title('Evolución del Error Angular')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(det_M_historial, label='Error Lineal (m)')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Determinante(m)')
plt.title('Evolución del Determinante')
plt.grid()
plt.legend()
plt.show()