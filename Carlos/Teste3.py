import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

R_1 = 4 
O_1 = 1
x_1 = np.array([0,0])
def R(n):
    return R_1*2**(1-n)
def O(n):
    return O_1*2**((2/3)* (n-1))

def phi(n,index):
    return np.pi*(1/4 + (index[-1]-1)/2)

def x(n, index,t):
    #index lista da forma [i,j,k,l,...,m] com n-1 letras:
    if len(index) != n-1:
        return "Erro de tamanho de array" 
    if n == 1:
        return x_1
    diff = R(n)*np.sqrt(2)*np.array([np.cos(O(n)*t + phi(n, index)), np.sin(O(n)*t + phi(n, index)) ])
    index.pop()
    return diff + x(n-1, index,t)





fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Sistema hierárquico de órbitas")

p_x1, = ax.plot([], [], 'ro', markersize=8)
p_x2, = ax.plot([], [], 'bo', markersize=6)
p_x3, = ax.plot([], [], 'go', markersize=4)


def update(frame):
    t = frame * 0.05

    # x1
    p_x1.set_data(x(1,[],t)[0], x(1,[],t)[1])

    # x2
    x2_positions = np.array([x(2,[i], t) for i in range(4)])
    p_x2.set_data(x2_positions[:,0], x2_positions[:,1])

    # x3
    x3_positions = []
    for i in range(4):
        for j in range(4):
            x3_positions.append(x(3,[i, j], t))
    x3_positions = np.array(x3_positions)

    p_x3.set_data(x3_positions[:,0], x3_positions[:,1])

    return p_x1, p_x2, p_x3



ani = FuncAnimation(fig, update, frames=4000, interval=1)
plt.show()

