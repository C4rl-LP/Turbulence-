import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
from tqdm import tqdm
import integradores as it
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
    idx = index[-1]  # agora 0,1,2,3

    return np.pi/4 + idx * (np.pi/2)


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


def quadrante(dx, dy):
    q = np.zeros_like(dx, dtype=int)
    q[(dx < 0) & (dy >= 0)] = 1
    q[(dx < 0) & (dy < 0)] = 2
    q[(dx >= 0) & (dy < 0)] = 3
    return q


def vizinhos(q):
    return np.stack([
        q,
        (q + 1) % 4,
        (q - 1) % 4
    ], axis=1)  # (Np, 3)


def atualiza_centro(cx, cy, t, n, idx):
    """
    cx, cy: (Np,)
    idx: (Np,)
    """
    Rn = R(n)
    On = Omega(n)

    phi_val = phi(n, idx)  # precisa aceitar array!

    shift_x = np.sqrt(2)*Rn * np.cos(On*t + phi_val)
    shift_y = np.sqrt(2)*Rn * np.sin(On*t + phi_val)

    return cx + shift_x, cy + shift_y


def campo_2(x, y, x_c, y_c,t, n, c= 0.8):
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Diferenças vetoriais (arrays)
    dx = x - x_c
    dy = y - y_c
    r2 = dx**2 + dy**2
    R2_coef = np.sqrt(2)*R(n)**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2_coef
    
    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(c / ((r2[mask] / R2_coef) - 1))
            / T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy


def campo_total_otimo(x, y, t, n_max):

    x = np.asarray(x)
    y = np.asarray(y)

    cx, cy = 0.0, 0.0  
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    V_sum_x, V_sum_y  = campo_2(x,y, cx, cy, 0, 1)
    vx += V_sum_x
    vy += V_sum_y


    for n in range(2, n_max + 1):
        
        dx = x - cx
        dy = y - cy
        # DEfinir dx para encontrar o quadrante
        q = quadrante(dx, dy)
        Rn = R(n)

        
        phi_val = phi(n, np.array([q]))
        cx_main = cx + np.sqrt(2)*Rn*np.cos(Omega(n)*t + phi_val)
        cy_main = cy + np.sqrt(2)*Rn*np.sin(Omega(n)*t + phi_val)


        # 🔹 vizinhos geométricos
        cx_all, cy_all = vizinhos_geometricos(cx_main, cy_main, Rn)
        mask = dentro_do_dominio(cx_all, cy_all, R(1))
        cx_all = cx_all[mask]
        cy_all = cy_all[mask]
        for cxi, cyi in zip(cx_all, cy_all):
            V_sum_x, V_sum_y = campo_2(x,y, cxi, cyi, 0, n)
            vx += V_sum_x
            vy += V_sum_y
        cx, cy = cx_main, cy_main
        

    return vx, vy 

def campo_2_barra(x, y, t, n, index):
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x_centro_n = x_centros(n, index, t)

    # Diferenças vetoriais (arrays)
    dx = x - x_centro_n[0]
    dy = y - x_centro_n[1]
    r2 = dx**2 + dy**2
    R2 = np.sqrt(2)*R(n)**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2

    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(0.3 / ((r2[mask] / R2) - 1))
            / T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy


def campo_total_2(x, y, t, n_max=3):
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos.
    """
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)
    
    for n in range(1, n_max + 1):
        for idx in indices_nivel(n):

            vx, vy = campo_2_barra(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy



# Essa função posteriormente deve depender de t, pois os vizinhos mgemétricos mudam no tempo
def vizinhos_geometricos(cx, cy, Rn):
    """
    Retorna:
    - centro principal
    - 4 vizinhos geométricos (cima, baixo, esquerda, direita)
    """

    d = 2 * Rn

    cx_all = np.array([
        cx,        # centro
        cx + d,    # direita
        cx - d,    # esquerda
        cx,        # cima
        cx         # baixo
    ])

    cy_all = np.array([
        cy,        # centro
        cy,        # direita
        cy,        # esquerda
        cy + d,    # cima
        cy - d     # baixo
    ])

    return cx_all, cy_all
def dentro_do_dominio(cx, cy, R1):
    return (np.abs(cx) <= R1) & (np.abs(cy) <= R1)




def centros_por_nivel(x, y, t, n_max):

    centros_hist = []


    cx, cy = 0.0, 0.0  # ou x_1
    centros_hist.append((np.array([cx]), np.array([cy])))

    for n in range(2, n_max + 1):

        dx = x - cx
        dy = y - cy

        q = quadrante(dx, dy)
        Rn = R(n)

        # 🔹 centro principal (hierárquico)
        phi_val = phi(n, np.array([q]))
        cx_main = cx + np.sqrt(2)*Rn*np.cos(Omega(n)*t + phi_val)
        cy_main = cy + np.sqrt(2)*Rn*np.sin(Omega(n)*t + phi_val)

        # 🔹 vizinhos geométricos
        cx_all, cy_all = vizinhos_geometricos(cx_main, cy_main, Rn)

        # 🔹 filtra domínio
        mask = dentro_do_dominio(cx_all, cy_all, R(1))
        cx_all = cx_all[mask]
        cy_all = cy_all[mask]

        centros_hist.append((cx_all, cy_all))

        # 🔹 segue apenas o centro principal
        cx, cy = cx_main, cy_main
    print(centros_hist)
    return centros_hist

def plot_centros(x, y, t, n_max=5):

    centros = centros_por_nivel(x, y, t, n_max)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c='red', s=100, label='ponto')

    for i, (cx, cy) in enumerate(centros, 1):
        plt.scatter(cx, cy, label=f'n={i}')

    # domínio
    R1 = R(1)
    plt.xlim(-R1, R1)
    plt.ylim(-R1, R1)

    plt.grid()
    plt.legend()
    plt.axis('equal')
    plt.show()



def simular_particulas_2_static(N, N_fields, dimensao_quadrado, r0, dt=0.01, t_max=1):
    """
    Simula N partículas em um pequeno quadrado
    ao redor da posição inicial r0.
    """

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total_otimo(x, y, 0, N_fields)
        return np.column_stack((Vx, Vy))

    L = dimensao_quadrado

    # Distribuição inicial uniforme em um círculo de raio L/2
    R0 = L / 2

    theta = np.random.uniform(0, 2*np.pi, size=N-1)
    u = np.random.uniform(0, 1, size=N-1)

    r = R0 * np.sqrt(u)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    particle_t0 = r0 + np.column_stack((dx, dy))

    # adiciona a partícula central exatamente em r0
    particle_t0 = np.vstack((r0,particle_t0))

    t, r = it.solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r
def testar_estabilidade_2_static(
    nivel_max,
    
    N_particulas,
    r0,
    pasta_saida="estabilidade_2_static",
    nivel_min = 1
):
    """
    Testa a estabilidade do ponto r0 para diferentes números de níveis do campo.
    Para cada nível n:
      - simula uma nuvem de partículas ao redor de r0
      - plota as trajetórias
      - salva a figura em disco
    """

    os.makedirs(pasta_saida, exist_ok=True)

    for n in tqdm(
        range(nivel_min, nivel_max + 1),
        desc="Testando níveis",
        unit="nível"
    ):


        # Escalas naturais do nível
        L = R(n)/16         # tamanho da nuvem inicial
        dt = 0.08 * R(n)             # passo temporal
        t_max = 10         # tempo total (alguns períodos)

        print(f"Testando estabilidade para n = {n}")

        ts, rs = simular_particulas_2_static(
            N=N_particulas,
            N_fields=n,
            dimensao_quadrado=L,
            r0=r0,
            dt=dt,
            t_max=t_max
        )

        # rs tem shape (Nt, N, 2)
        Nt = rs.shape[0]

        plt.figure(figsize=(6, 6))


        cores = plt.cm.viridis(np.linspace(0, 1, N_particulas))

        for i in range(N_particulas):
            # Trajetória
            plt.plot(
                rs[:, i, 0],
                rs[:, i, 1],
                lw=1,
                alpha=0.4,
                color=cores[i]
            )

            # Ponto inicial (vazado)
            plt.scatter(
                rs[0, i, 0],
                rs[0, i, 1],
                facecolors="none",
                edgecolors=cores[i],
                s=30,
                linewidths=1.5
            )

            # Ponto final (bem visível)
            plt.scatter(
                rs[-1, i, 0],
                rs[-1, i, 1],
                color=cores[i],
                s=30,
                zorder=3
            )

        # Ponto central
        plt.scatter(
            r0[0, 0],
            r0[0, 1],
            c="red",
            s=100,
            marker="x",
            linewidths=2,
            label="ponto r0"
        )

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Estabilidade em torno de r0 — Níveis até n = {n}")
        plt.legend()
        plt.grid(alpha=0.3)


        nome_arquivo = os.path.join(
            pasta_saida,
            f"estabilidade_nivel_{n}_em_x{r0[0,0]:.3f}_y{r0[0,1]:.2f}em.png"
        )
        plt.savefig(nome_arquivo, dpi=200)
        plt.close()

        print(f"Imagem salva em: {nome_arquivo}")

r0 = np.array([[.2, -.0]])

testar_estabilidade_2_static(
    nivel_max=5,
    N_particulas=50,
    r0=r0,
    nivel_min = 1
)

import numpy as np
import matplotlib.pyplot as plt

n_max = 2      # níveis do campo
t0 = 0.0       # instante para congelar o campo

# malha
N = 300
x = np.linspace(-R(1), R(1), N)
y = np.linspace(-R(1), R(1), N)

X, Y = np.meshgrid(x,y)

# campo na malha
Vx, Vy = campo_total_2(X, Y, t0, n_max)
Vx2, Vy2 = campo_total_otimo(X,Y, t0, n_max)
# plot
plt.figure(figsize=(8,8))

plt.streamplot(
    X, Y,
    Vx-Vx2, Vy-Vy2,
    density=3.0,
    linewidth=1
)

plt.xlim(-R(1),R(1))
plt.ylim(-R(1),R(1))
plt.axis('equal')
plt.grid()
plt.show()