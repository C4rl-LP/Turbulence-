
import os 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import product





R_1 = 1            # Raio característico do nível 1
O_1 = 1            # Frequência angular base
x_1 = np.array([0, 0])  # Centro inicial (nível 1)

# Fator de escala entre frequências dos níveis
lamb = 2**(2/3)



def solve_RK4(func, r0, t0, dt, t_max):
    """
    Resolve o sistema dr/dt = func(t, r) usando RK4.

    Parâmetros:
    - func : função do campo vetorial
    - r0   : condição inicial (array)
    - t0   : tempo inicial
    - dt   : passo de tempo
    - t_max: tempo final
    """
    Nt = int((t_max - t0)/dt) + 1
    r = np.zeros((Nt, *r0.shape))
    t = np.zeros(Nt)

    r[0] = r0
    t[0] = t0

    for i in range(Nt - 1):
        ti = t[i]
        ri = r[i]

        k1 = func(ti, ri)
        k2 = func(ti + dt/2, ri + dt/2*k1)
        k3 = func(ti + dt/2, ri + dt/2*k2)
        k4 = func(ti + dt,   ri + dt*k3)

        r[i+1] = ri + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = ti + dt

    return t, r



from itertools import product

R_1 = 1            # Raio característico do nível 1
O_1 = 1            # Frequência angular base
x_1 = np.array([0, 0])  # Centro inicial (nível 1)

# Fator de escala entre frequências dos níveis
lamb = 2**(2/3)



def indices_nivel(n):
    """
    Gera os índices que identificam cada centro no nível n.
    O número de centros cresce como 4^(n-1).
    """
    if n == 1:
        return [[]]
    return list(product(range(4), repeat=n-1))


def R(n):
    """ Raio característico do nível n """
    return R_1 * 2**(1 - n)


def Omega(n):
    """ Frequência angular do nível n """
    return O_1 * lamb**(n - 1)


def T(n):
    """ Período associado ao nível n """
    return 2 * np.pi / Omega(n)


# ======================================================================
# GEOMETRIA DOS CENTROS
# ======================================================================

def phi(n, index):
    """
    Fase angular associada ao último índice da hierarquia.
    """
    return np.pi * (1/4 + (index[-1] - 1) / 2)


def x_centros(n, index, t):
    """
    Calcula recursivamente a posição do centro associado
    a um índice hierárquico no nível n.
    """
    if n == 1:
        return x_1

    if len(index) != n - 1:
        raise ValueError("Erro de tamanho de array")

    # Deslocamento circular do centro atual
    diff = R(n) * np.sqrt(2) * np.array([
        np.cos(Omega(n) * t + phi(n, index)),
        np.sin(Omega(n) * t + phi(n, index))
    ])

    # Soma com o centro do nível anterior
    return diff + x_centros(n - 1, index[:-1], t)

def centros_nivel(n, t):
    """
    Retorna array (N_centros, 2) com todos os centros do nível n
    """
    centros = np.array([x_1])  # nível 1

    for k in range(2, n+1):

        Rk = R(k) * np.sqrt(2)
        angle_base = Omega(k) * t

        # 4 possíveis fases
        fases = np.array([
            -np.pi/4,
             np.pi/4,
             3*np.pi/4,
             5*np.pi/4
        ])

        deslocamentos = Rk * np.column_stack([
            np.cos(angle_base + fases),
            np.sin(angle_base + fases)
        ])

        # Produto cartesiano vetorizado
        centros = (
            centros[:, None, :] + deslocamentos[None, :, :]
        ).reshape(-1, 2)

    return centros


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# CONFIGURAÇÃO
# ==============================

N_max = 4           # número máximo de níveis
L = 1.5             # tamanho da janela
frames = 400        # número de frames
interval = 30       # ms entre frames

t_vals = np.linspace(0, 2*np.pi / Omega(1), frames)

# ==============================
# FIGURA
# ==============================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect('equal')

# uma coleção de scatters (um por nível)
scatters = []

cores = plt.cm.plasma(np.linspace(0.2, 1, N_max))

for n in range(1, N_max+1):
    sc = ax.scatter([], [], s=30, color=cores[n-1], label=f"Nível {n}")
    scatters.append(sc)

ax.legend(loc="upper right")

# ==============================
# FUNÇÃO DE UPDATE
# ==============================

def update(frame):

    t = t_vals[frame]

    for n in range(1, N_max+1):

        centros = centros_nivel(n, t)

        scatters[n-1].set_offsets(centros)

    return scatters

# ==============================
# ANIMAÇÃO
# ==============================

anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=interval,
    blit=True
)

plt.show()