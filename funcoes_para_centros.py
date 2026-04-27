import numpy as np
import matplotlib.pyplot as plt
from itertools import product

R_1 = 1            # Raio característico do nível 1
O_1 = 1            # Frequência angular base
x_1 = np.array([0, 0])  # Centro inicial (nível 1)
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


def centros_por_nivel_vetorizado(x,y,t,n_max):

    x=np.atleast_1d(x)
    y=np.atleast_1d(y)

    Np=len(x)

    centros_hist=[]

    cx=np.zeros(Np)
    cy=np.zeros(Np)

    # nível 1
    centros_hist.append(
        [
         (np.array([cx[i]]),
          np.array([cy[i]]))
         for i in range(Np)
        ]
    )

    for n in range(2,n_max+1):

        dx=x-cx
        dy=y-cy

        q=quadrante(dx,dy)

        Rn=R(n)

        phi_val=np.pi/4 + q*np.pi/2

        cx_main = cx + np.sqrt(2)*Rn*np.cos(
            Omega(n)*t + phi_val
        )

        cy_main = cy + np.sqrt(2)*Rn*np.sin(
            Omega(n)*t + phi_val
        )

        d=2*Rn

        cx_all=np.column_stack([
            cx_main,
            cx_main+d,
            cx_main-d,
            cx_main,
            cx_main
        ])

        cy_all=np.column_stack([
            cy_main,
            cy_main,
            cy_main,
            cy_main+d,
            cy_main-d
        ])

        mask=dentro_do_dominio(cx_all, cy_all, R(1))
        nivel_centros = [
            (
              cx_all[i,mask[i]],
              cy_all[i,mask[i]]
            )
            for i in range(Np)
        ]

        centros_hist.append(nivel_centros)

        cx=cx_main
        cy=cy_main

    return centros_hist


if __name__ == "__main__":

    a =centros_por_nivel_vetorizado(np.array([.33, -.33, 0.1]), np.array([.33, -.33, 0.1]), 0, 3)
    print(a[2][1])
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

