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
            * np.exp(0.4 / ((r2[mask] / R2) - 1))
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
def campo_total_correto(x, y, t, n_max=3):
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
def campo_total_podado(xp, yp, t, n_max):
    """
    Soma contribuições apenas dos centros
    encontrados por verificar().
    """
    xp=np.atleast_1d(xp)
    yp=np.atleast_1d(yp)
    # centro raiz n=1
    x0,y0 = 0.0,0.0
    Np = len(xp)
    # árvore podada
    centros = fc.verificar_vet(
        n_max,
        xp,yp,
        t,
        x0,y0
    )

    vx_total=0.0
    vy_total=0.0


    # nível 1 (raiz)
    vx,vy = campo_2(
        xp,yp,
        x0,y0,
        t,
        1
    )

    vx_total += vx
    vy_total += vy

    for n in range(2, n_max +1):
        for j in range(Np):
            for cx, cy in centros[n-1][j]:
                vx, vy = campo_2(xp[j], yp[j], cx, cy, t, n)
                vx_total[j] += vx 
                vy_total[j] += vy 
    return vx_total, vy_total

x = np.array([-.2])
y = np.array([.1])


n = 20
# Função para simular as partículas.
x = np.array([-.2, -.201, .1])
y = np.array([.1, .1, .013])
t = 0

a = campo_total_otimo(x, y, t, n)
b = campo_total_otimo_vet(x, y, t, n)
d = campo_total_podado(x,y, t, n)
c = campo_total_correto(x, y , t, n)


print(f'campo total otimo:{a}')
print(f'campo total otimo vetorizado certo:{b}')
print(f'campo total correto:{c}')
print(f'campo total podado:{d}')

'''
r0 = np.array([[-.2, .1]])
a= 'estabilidade_2'
b ='estabilidade_com_todos_os_pontos'
it.testar_estabilidade_2(
    nivel_max=7,
    N_particulas=50,
    r0=r0,
    nivel_min = 5,
    pasta_saida= b,
    funcao=campo_total_correto
)
'''