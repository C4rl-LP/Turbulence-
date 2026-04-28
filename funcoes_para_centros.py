import numpy as np
import matplotlib.pyplot as plt
from itertools import product

R_1 = 1            # Raio característico do nível 1
O_1 = 2*np.pi            # Frequência angular base
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

def T(n):
    """ Período associado ao nível n """
    return 2*np.pi/O_1 * lamb**(-n +1 ) 

def Omega(n):
    """ Frequência angular do nível n """
    return O_1 * lamb**(n - 1)   





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

def subcentro(n,cx_0, cy_0,  t):
    sub_cx = np.zeros(4)
    sub_cy = np.zeros(4)
    for i in range(4):
        phi_i = np.pi/4 + np.pi/2 * i
        sub_cx[i] =  cx_0 + R(n+1) *np.sqrt(2)*np.cos(Omega(n+1)*t + phi_i )
        sub_cy[i] =  cy_0 + R(n+1) *np.sqrt(2)*np.sin(Omega(n+1)*t + phi_i )
    return sub_cx, sub_cy       

def sondar(n, xp, yp, cx_0, cy_0, t):
    sub_cx, sub_cy = subcentro(n, cx_0, cy_0, t)
    cx_valid = []
    cy_valid = []
    for s_cx_i, s_cy_i in zip(sub_cx, sub_cy):
        dx = xp - s_cx_i
        dy = yp - s_cy_i 
        print(s_cx_i, s_cy_i)
        print((dx**2 + dy**2))
        if dx**2 + dy**2 < 2*R(n+1)**2:
            cx_valid.append(s_cx_i)
            cy_valid.append(s_cy_i)
    return cx_valid, cy_valid

def verificar(N, xp, yp, t, x_0, y_0): # x_0, y_0 é  a coordenada do primeiro centro n =1
    centros = [np.array([x_0, y_0])]
    centros_vistos =  [np.array([x_0, y_0])]
    for j in range(1, N):
        sub = []
        for r_0k in centros_vistos:
            valid_x, valid_y = sondar(j, xp,yp, r_0k[0],r_0k[1],t)
            for cxi,cyi in zip(
                valid_x,
                valid_y
            ):
                sub.append((cxi,cyi))
        centros_vistos = sub

        centros.append(sub)
    return centros

def sondar_vet(n,xp,yp,cx0,cy0,t):

    xp=np.atleast_1d(xp)
    yp=np.atleast_1d(yp)

    sub_cx,sub_cy = subcentro(
        n,cx0,cy0,t
    )

    # (Np,4)
    dx = xp[:,None]-sub_cx
    dy = yp[:,None]-sub_cy

    mask = (
        dx**2+dy**2
        < 2*R(n+1)**2
    )

    return sub_cx,sub_cy,mask

def verificar_vet(N,xp,yp,t,x0,y0):

    xp=np.atleast_1d(xp)
    yp=np.atleast_1d(yp)

    Np=len(xp)

    # para cada partícula:
    ativos=[
        [(x0,y0)]
        for _ in range(Np)
    ]

    hist=[ativos.copy()]


    for n in range(1,N):

        novos=[]

        for p in range(Np):
            
            ativos_p=[]

            for cx,cy in ativos[p]:

                sub_cx,sub_cy,mask = sondar_vet(
                    n,
                    np.array([xp[p]]),
                    np.array([yp[p]]),
                    cx,
                    cy,
                    t
                )

                filhos=list(
                    zip(
                      sub_cx[mask[0]],
                      sub_cy[mask[0]]
                    )
                )

                ativos_p.extend(filhos)

            novos.append(ativos_p)

        ativos=novos
        hist.append(ativos.copy())

    return hist






if __name__ == "__main__":


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
    xp=np.array([
    0.10,
    0.25,
    -0.30,
    0.40
    ])

    yp=np.array([
        0.50,
        0.20,
        0.10,
    -0.20
    ])

    hist=verificar_vet(
    4,
    xp,yp,0,
    0,0
    )
    print(hist[0])

    print(1/(lamb*5))

    plt.figure(figsize=(8,8))

    # partículas
    plt.scatter(
        xp,yp,
        c='red',
        s=70,
        label='pontos'
    )
    cores=['k','b','g','orange','purple','brown']
    for nivel in range(len(hist)):

        for p in range(len(xp)):

            centros = hist[nivel][p]

            if len(centros)==0:
                continue

            cx=[c[0] for c in centros]
            cy=[c[1] for c in centros]

            plt.scatter(
                cx,
                cy,
                s=30,
                label=None
            )

            # liga ponto aos centros ativos
            for cxi,cyi in centros:
                plt.plot(
                    [xp[p],cxi],
                    [yp[p],cyi],
                    alpha=.25, color=cores[nivel%len(cores)]
                )

    plt.axis('equal')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.grid()
    plt.show()
