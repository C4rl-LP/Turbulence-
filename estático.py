import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import integradores as it
import funcoes_para_centros as fc 
from funcoes_para_centros import R_1, O_1, x_1, lamb
import os





# Campo base como função dos centros, sem usar o index
def campo_2(x, y, x_c, y_c,t, n, c= 0.4):
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
    R2_coef = np.sqrt(2)*fc.R(n)**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2_coef
    
    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(c / ((r2[mask] / R2_coef) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy
# Campo base que utiliza os index (CUIDADO COM O VALOR 'c" que está definifo dentro da função)
def campo_2_com_index(x, y, t, n, index):
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x_centro_n = fc.x_centros(n, index, t)

    # Diferenças vetoriais (arrays)
    dx = x - x_centro_n[0]
    dy = y - x_centro_n[1]
    r2 = dx**2 + dy**2
    R2 = np.sqrt(2)*fc.R(n)**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2

    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(0.8 / ((r2[mask] / R2) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy

# Campo total de forma ótima, neste caso ele é vetorizado mas os centros ótimos são baseados num ponto central
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
        
  
        # DEfinir dx para encontrar o quadrante
        q = fc.quadrante(
            np.array([x[0]-cx]),
            np.array([y[0]-cy])
        )[0]
        Rn = fc.R(n)

        
        phi_val = fc.phi(n, np.array([q]))
        cx_main = cx + np.sqrt(2)*Rn*np.cos(fc.Omega(n)*t + phi_val)
        cy_main = cy + np.sqrt(2)*Rn*np.sin(fc.Omega(n)*t + phi_val)


        # 🔹 vizinhos geométricos
        cx_all, cy_all = fc.vizinhos_geometricos(cx_main, cy_main, Rn)
        mask = fc.dentro_do_dominio(cx_all, cy_all, fc.R(1))
        cx_all = cx_all[mask]
        cy_all = cy_all[mask]
        for cxi, cyi in zip(cx_all, cy_all):
            V_sum_x, V_sum_y = campo_2(x,y, cxi, cyi, 0, n)
            vx += V_sum_x
            vy += V_sum_y
        cx, cy = cx_main, cy_main
        

    return vx, vy 

# Aqui faz-se a conta com centros ótimos para todos os pontos (Intera sobre Np!!!!)
def campo_total_otimo_vet(x,y,t,n_max):

    x=np.atleast_1d(x)
    y=np.atleast_1d(y)

    vx=np.zeros_like(x,dtype=float)
    vy=np.zeros_like(y,dtype=float)

    cx=0.0
    cy=0.0

    # nível 1
    Vx,Vy = campo_2(x,y,cx,cy,t,1)

    vx += Vx
    vy += Vy


    for n in range(2,n_max+1):

        # usa partícula referência para árvore
        dx=x-cx
        dy=y-cy

        q=fc.quadrante(dx,dy)

        Rn=fc.R(n)

        phi_val=np.pi/4 + q*np.pi/2

        cx_main = (
            cx
            + np.sqrt(2)*Rn*
            np.cos(fc.Omega(n)*t+phi_val)
        )

        cy_main = (
            cy
            + np.sqrt(2)*Rn*
            np.sin(fc.Omega(n)*t+phi_val)
        )


        cx_all,cy_all = fc.vizinhos_geometricos(
            cx_main,
            cy_main,
            Rn
        )

        mask=fc.dentro_do_dominio(
            cx_all,cy_all,fc.R(1)
        )

        cx_all=cx_all[mask]
        cy_all=cy_all[mask]


        # -------- vetoriza soma dos centros --------

        # shape (Np, Nc)
        DX = x[:,None] - cx_all[None,:]
        DY = y[:,None] - cy_all[None,:]

        r2 = DX**2 + DY**2

        R2coef=np.sqrt(2)*Rn**2

        pot=np.zeros_like(r2)

        m = r2 < R2coef

        pot[m]=(
            2*np.pi
            *np.exp(
                0.8/((r2[m]/R2coef)-1)
            )
            /fc.T(n)
        )

        # soma sobre centros
        vx += np.sum(-DY*pot,axis=1)
        vy += np.sum( DX*pot,axis=1)

        # ramo principal
        cx=cx_main
        cy=cy_main

    return vx,vy

# Campo total usando todos os index e não ótimo
def campo_total_2_com_index(x, y, t, n_max=3):
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos.
    """
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)
    
    for n in range(1, n_max + 1):
        for idx in fc.indices_nivel(n):

            vx, vy = campo_2_com_index(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy

# Função para simular as partículas.



def testar_estabilidade_2(
    nivel_max,
    
    N_particulas,
    r0,
    pasta_saida="estabilidade_com_todos_os_pontos",
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

    for n in range(nivel_min, nivel_max + 1):


        # Escalas naturais do nível
        L = fc.R(n)/16         # tamanho da nuvem inicial
        dt = 0.08 * fc.R(n)             # passo temporal
        t_max = 5         # tempo total (alguns períodos)

        print(f"Testando estabilidade para n = {n}")

        ts, rs = it.simular_particulas_2_static(
            N=N_particulas,
            N_fields=n,
            dimensao_quadrado=L,
            r0=r0,
            dt=dt,
            t_max=t_max, func= campo_total_otimo_vet
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

r0 = np.array([[-.2, .1]])

testar_estabilidade_2(
    nivel_max=8,
    N_particulas=50,
    r0=r0,
    nivel_min = 8
)


