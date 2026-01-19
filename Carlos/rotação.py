# ======================================================================
# IMPORTAÇÕES
# ======================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import product


# ======================================================================
# PARÂMETROS GLOBAIS DO MODELO
# ======================================================================

R_1 = 1            # Raio característico do nível 1
O_1 = 1            # Frequência angular base
x_1 = np.array([0, 0])  # Centro inicial (nível 1)

# Fator de escala entre frequências dos níveis
lamb = 2**(2/3)


# ======================================================================
# INTEGRADOR NUMÉRICO: RUNGE-KUTTA DE 4ª ORDEM (RK4)
# ======================================================================

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


# ======================================================================
# ESTRUTURA HIERÁRQUICA DOS NÍVEIS
# ======================================================================

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


# ======================================================================
# DEFINIÇÃO DO CAMPO VETORIAL
# ======================================================================

def campo(x, y, t, n, index):
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x_centro_n = x_centros(n, index, t)

    # Diferenças vetoriais (arrays)
    dx = x - x_centro_n[0]
    dy = y - x_centro_n[1]

    # Parâmetro do corte espacial
    alpha = 3
    r2_cut = alpha**2 * R(n)

    # Máscara booleana: partículas suficientemente próximas
    mask = dx**2 + dy**2 <= r2_cut

    # Inicializa o campo como zero
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)

    # Se nenhuma partícula estiver dentro do raio, retorna zero direto
    if not np.any(mask):
        return vx, vy

    # Fator radial (somente onde o campo é relevante)
    factor = -(dx[mask]**2 + dy[mask]**2) * 2 / R(n)

    # Intensidade do campo
    poten = 2 * np.pi * np.exp(factor - 1) / T(n)

    # Campo rotacional
    vx[mask] = -dy[mask] * poten
    vy[mask] =  dx[mask] * poten

    return vx, vy


def campo_total(x, y, t, n_max=3):
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos.
    """
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)

    for n in range(1, n_max + 1):
        for idx in indices_nivel(n):
            vx, vy = campo(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy


# ======================================================================
# SIMULAÇÃO DE PARTÍCULAS
# ======================================================================

np.random.seed(1200)

def simular_particulas(N, N_fields, dimensao_quadrado, r0, dt=0.01, t_max=1):
    """
    Simula N partículas em um pequeno quadrado
    ao redor da posição inicial r0.
    """

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t, N_fields)
        return np.column_stack((Vx, Vy))

    L = dimensao_quadrado

    # Distribuição inicial aleatória das partículas
    particle_t0 = r0 + np.random.uniform(-L/2, L/2, size=(N, 2))
    print(particle_t0)

    t, r = solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r


# ======================================================================
# SIMULAÇÕES AUXILIARES: CAMPO CONGELADO E CAMPO MÓVEL
# ======================================================================

def simular_particula_campo_congelado(
    r0, 
    N_fields, 
    t_freeze=0.0, 
    dt=0.01, 
    t_max=10
):
    """
    Simula uma partícula em um campo congelado
    em um instante fixo t = t_freeze.
    """
    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t_freeze, N_fields)
        return np.column_stack((Vx, Vy))

    t, r = solve_RK4(funcao, r0, 0.0, dt, t_max)
    return t, r


def simular_particula_campo_movel(
    r0, 
    N_fields, 
    dt=0.01, 
    t_max=10
):
    """
    Simula uma partícula em um campo dependente do tempo.
    """
    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t, N_fields)
        return np.column_stack((Vx, Vy))

    t, r = solve_RK4(funcao, r0, 0.0, dt, t_max)
    return t, r


# ======================================================================
# EXECUÇÃO DAS SIMULAÇÕES
# ======================================================================

r0 = np.array([[np.pi/6, np.sqrt(2)/4]])
N_fields = 3

# Campo dependente do tempo
t1, r_movel = simular_particula_campo_movel(
    r0,
    N_fields=N_fields,
    dt=0.01,
    t_max=10
)

# Campo congelado
t2, r_congelado = simular_particula_campo_congelado(
    r0,
    N_fields=N_fields,
    t_freeze=0.0,
    dt=0.01,
    t_max=10
)

# Nuvem de partículas
ts, rs = simular_particulas(
    N=3,
    N_fields=N_fields,
    dimensao_quadrado=0.05,
    r0=r0,
    dt=0.01,
    t_max=20
)


# ======================================================================
# EXTRAÇÃO DOS DADOS PARA PLOT
# ======================================================================

x_m, y_m = r_movel[:, 0, 0], r_movel[:, 0, 1]
x_c, y_c = r_congelado[:, 0, 0], r_congelado[:, 0, 1]
x_s, y_s = rs[:, :, 0], rs[:, :, 1]


# ======================================================================
# VISUALIZAÇÃO
# ======================================================================

plt.figure(figsize=(6, 6))

plt.plot(x_m, y_m, label='Campo dependente do tempo')
plt.plot(x_c, y_c, '--', label='Campo congelado (t = 0)')
plt.plot(x_s, y_s)

plt.scatter(x_m[0], y_m[0], color='green', s=50, label='Início')
plt.scatter(x_m[-1], y_m[-1], color='red', s=50, label='Fim')
plt.scatter(x_c[-1], y_c[-1], color='purple', s=50, label='Fim do congelado')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparação: campo móvel vs campo congelado')
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.show()
