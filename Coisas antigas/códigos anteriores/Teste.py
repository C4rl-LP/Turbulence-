import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Parâmetros
# ======================
omega = 1.0
R2 = 2.0
R3 = 1

phi2 = np.array([np.pi/4 + i*np.pi/2 for i in range(4)])
phi3 = np.array([np.pi/4 + j*np.pi/2 for j in range(4)])

x1 = np.array([0.0, 0.0])

# ======================
# Funções de posição
# ======================
def x2(i, t):
    return x1 + R2*np.sqrt(2)*np.array([
        np.cos(omega*t + phi2[i]),
        np.sin(omega*t + phi2[i])
    ])

def x3(i, j, t):
    return x2(i, t) + R3*np.sqrt(2)*np.array([
        np.cos(omega*2*t  + phi3[j]),
        np.sin(omega*2*t  + phi3[j])
    ])

# ======================
# Setup do gráfico
# ======================
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Sistema hierárquico de órbitas")

# Pontos
p_x1, = ax.plot([], [], 'ro', markersize=8)
p_x2, = ax.plot([], [], 'bo', markersize=6)
p_x3, = ax.plot([], [], 'go', markersize=4)

# ======================
# Animação
# ======================
def update(frame):
    t = frame * 0.05

    # x1
    p_x1.set_data(x1[0], x1[1])

    # x2
    x2_positions = np.array([x2(i, t) for i in range(4)])
    p_x2.set_data(x2_positions[:,0], x2_positions[:,1])

    # x3
    x3_positions = []
    for i in range(4):
        for j in range(4):
            x3_positions.append(x3(i, j, t))
    x3_positions = np.array(x3_positions)

    p_x3.set_data(x3_positions[:,0], x3_positions[:,1])

    return p_x1, p_x2, p_x3

ani = FuncAnimation(fig, update, frames=400, interval=30)
plt.show()
