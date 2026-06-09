import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GaussianField:
    def __init__(self, x0=0.0, y0=0.0, A=10.0, sigma_x=5.0, sigma_y=5.0):
        """
        Define un campo escalar gaussiano con un único máximo.
        
        Parametros:
        x0, y0 : Coordenadas del máximo (la fuente).
        A      : Amplitud o valor máximo del campo.
        sigma_x, sigma_y : Desviación estándar (dispersión del campo).
        """
        self.x0 = x0
        self.y0 = y0
        self.A = A
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def evaluate(self, r):
        """
        Evalúa la intensidad del campo en una posición r = [x, y].
        Soporta tanto un único punto como un array de puntos (N, 2).
        """
        r = np.array(r)
        if r.ndim ==  1:
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
# Visualización del Campo y sus Gradientes
# =====================================================================
'''
# 1. Instanciar el campo con la fuente en (2, 3)
fuente_x, fuente_y = 2.0, 3.0
campo = GaussianField(x0=fuente_x, y0=fuente_y, A=10.0, sigma_x=4.0, sigma_y=4.0)

# 2. Crear una rejilla de puntos para graficar
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
# Reorganizar la rejilla en formato (N, 2) para evaluar eficientemente
grid_points = np.stack([X, Y], axis=-1)

# 3. Evaluar el campo en toda la rejilla
Z = campo.evaluate(grid_points)

# 4. Calcular los gradientes en puntos espaciados (para que las flechas no saturen el gráfico)
x_sub = np.linspace(-10, 10, 15)
y_sub = np.linspace(-10, 10, 15)
X_sub, Y_sub = np.meshgrid(x_sub, y_sub)
grid_sub = np.stack([X_sub, Y_sub], axis=-1)
gradientes = campo.gradient(grid_sub)

# Separar componentes del gradiente (U = componente x, V = componente y)
U = gradientes[..., 0]
V = gradientes[..., 1]

# 5. Graficar usando Matplotlib
plt.figure(figsize=(8, 6))
# Mapa de calor del campo escalar
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Intensidad del campo ')

# Campo de vectores (Gradiente)
plt.quiver(X_sub, Y_sub, U, V, color='white', scale=15, pivot='mid', alpha=0.8, label='Vectores Gradiente')

# Marcar la posición real de la fuente
plt.scatter(fuente_x, fuente_y, color='red', marker='*', s=200, label='Fuente (Máximo)')

plt.title('Campo Escalar Gaussiano y sus Vectores Gradiente')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.show()
'''
# =====================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# =====================================================================

# 1. Instanciar el entorno
campo = GaussianField(x0=3.0, y0=4.0, A=15.0, sigma_x=5.0, sigma_y=5.0)

# 2. Inicializar los robots (N = 3)
# Colocamos los 3 robots formando un pequeño triángulo lejos de la fuente
N = 3
posiciones = np.array([
    [-6.0, -5.0],  # Robot 0
    [-4.0, -6.0],  # Robot 1
    [-5.0, -3.0]   # Robot 2
])

# Parámetros temporales
dt = 0.05       # Paso de tiempo
pasos = 1000     # Duración de la simulación
k_gain = 0.5    # Ganancia de control (ajusta la velocidad de convergencia)

# Grafo de comunicación: Totalmente conectado (a_ij = 1 para todo i != j)
A_matrix = np.ones((N, N)) - np.eye(N)

# Contenedor para guardar el historial de posiciones (para la animación)
# Dimensiones: (pasos, N, 2)
historial_posiciones = np.zeros((pasos, N, 2))

# =====================================================================
# BUCLE PRINCIPAL DE SIMULACIÓN (Integrador)
# =====================================================================

for t in range(pasos):
    historial_posiciones[t] = posiciones.copy()
    # 1. Calcular el centroide r_c (Sección II-A)
    r_c = np.mean(posiciones, axis=0)
    
    # 2. Calcular vectores de geometría x_i y el radio máximo D
    x_i = posiciones - r_c
    normas_x = np.linalg.norm(x_i, axis=1)
    D = np.max(normas_x)
    
    # 3. Medir la intensidad de campo \sigma(r_i) para cada robot
    lineas_campo = np.array([campo.evaluate(posiciones[i]) for i in range(N)])
    
    # 4. Calcular la dirección de ascenso aproximada \hat{L}_\sigma (Ecuación 4)
    # \hat{L}_\sigma = (1 / (N * D^2)) * \sum (\sigma_i * x_i)
    suma_ponderada = np.zeros(2)
    for i in range(N):
        suma_ponderada += lineas_campo[i] * x_i[i]
        
    L_hat = suma_ponderada / (N * (D**2))
    
    # 5. Aplicar ley de control de dinámica libre (Sección III-A)
    # Todos los robots se mueven en la dirección L_hat.
    # Añadimos una ganancia multiplicativa opcional para regular la velocidad de simulación.
    ganancia = 2.0 
    u_control = ganancia * L_hat
    
    # Integrador simple (Euler)
    posiciones += u_control * dt

# =====================================================================
# ANIMACIÓN Y VISUALIZACIÓN
# =====================================================================

fig, ax = plt.subplots(figsize=(8, 7))

# Dibujar el fondo del campo escalar (mapa de calor)
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = campo.evaluate(np.stack([X, Y], axis=-1))
contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
fig.colorbar(contour, label='Intensidad del campo')

# Marcar la fuente (máximo)
ax.scatter(campo.x0, campo.y0, color='red', marker='*', s=200, label='Fuente', zorder=5)

# Elementos gráficos que se actualizarán en la animación
lineas_trayectoria = [ax.plot([], [], '--', lw=1.5, label=f'Robot {i}')[0] for i in range(N)]
puntos_robots = ax.scatter([], [], s=80, edgecolors='black', zorder=4)
centro_masa_dot = ax.scatter([], [], color='white', marker='x', s=100, label='Centro de Masa', zorder=4)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('Posición X')
ax.set_ylabel('Posición Y')
ax.set_title('Simulación Source-Seeking: Enjambre de Robots (N=3)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

# Función de inicialización para la animación
def init():
    for linea in lineas_trayectoria:
        linea.set_data([], [])
    puntos_robots.set_offsets(np.empty((0, 2)))
    centro_masa_dot.set_offsets(np.empty((0, 2)))
    return lineas_trayectoria + [puntos_robots, centro_masa_dot]

# Función que se ejecuta en cada fotograma
def animate(frame):
    # Obtener posiciones de los robots en este instante de tiempo
    pos_actuales = historial_posiciones[frame]
    
    # 1. Actualizar las líneas de las trayectorias recorridas hasta ahora
    for i in range(N):
        trayectoria_x = historial_posiciones[:frame+1, i, 0]
        trayectoria_y = historial_posiciones[:frame+1, i, 1]
        lineas_trayectoria[i].set_data(trayectoria_x, trayectoria_y)
        
    # 2. Actualizar la posición puntual de los robots
    puntos_robots.set_offsets(pos_actuales)
    
    # 3. Calcular y actualizar el centro de masa del enjambre
    centro_masa = np.mean(pos_actuales, axis=0)
    centro_masa_dot.set_offsets([centro_masa])
    
    return lineas_trayectoria + [puntos_robots, centro_masa_dot]

# Crear la animación
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=pasos, interval=30, blit=True, repeat=False
)

plt.show()