import numpy as np 
import os
from matplotlib import pyplot as plt
import funcoes_para_centros as fc
from funcoes_para_centros import R_1, O_1, lamb, x_1, T_1, alpha_padrao, c_padrao
import time 

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

def simular_particulas(N, N_fields, dimensao_quadrado, r0, dt=0.01, t_max=1, func = None, temporal = False):
    """
    Simula N partículas em um pequeno quadrado
    ao redor da posição inicial r0.
    """

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        if temporal:
            Vx, Vy = func(x, y, t, N_fields)
        else:
            Vx, Vy = func(x, y, 0, N_fields)
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

    t, r = solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r

def testar_estabilidade_estatico(
    nivel_max,
    
    N_particulas,
    r0,
    pasta_saida="estabilidade_2",
    nivel_min = 1,
    funcao = None
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
        t0 = time.perf_counter()

        # Escalas naturais do nível
        L = fc.R(n)/8         # tamanho da nuvem inicial
        dt = 0.03 * fc.R(n)             # passo temporal
        t_max = lamb/2        # tempo total (alguns períodos)

        print(f"Testando estabilidade para n = {n}")

        ts, rs = simular_particulas(
            N=N_particulas,
            N_fields=n,
            dimensao_quadrado=L,
            r0=r0,
            dt=dt,
            t_max=t_max, func= funcao
        )

        # rs tem shape (Nt, N, 2)
        Nt = rs.shape[0]
        t1 = time.perf_counter()
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
        print(f"Tempo demorado: {t1 - t0}")
        print(f"Imagem salva em: {nome_arquivo}")

def testar_estabilidade_temporal(
    nivel_max,
    
    N_particulas,
    r0,
    pasta_saida="estabilidade_2",
    pasta_saida_info = "estabilidade_podadda_info",
    nivel_min = 1,
    funcao = None,
    temporal = True
):
    """
    Testa a estabilidade do ponto r0 para diferentes números de níveis do campo.
    Para cada nível n:
      - simula uma nuvem de partículas ao redor de r0
      - plota as trajetórias
      - salva a figura em disco
    """
    
    

    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_saida_info, exist_ok=True)
    nome_do_txt_info = os.path.join(pasta_saida_info, "Informações.txt")
    with open(nome_do_txt_info, "a") as f:
        f.write('\n')
        f.write("Carateristicas \n")
        f.write('\n')
        f.write(f"R_1 = {R_1},       T_1 = {T_1},     lambda = {lamb} \n")
        f.write(f"alpha de corte = {alpha_padrao}, Fator c = {c_padrao}\n")
        f.write(f"r_0 = {r0}, Número de partículas: {N_particulas}, Temporal: {temporal}\n")
        f.write("Estatísticas: \n")
        f.write('\n')
        

    nivel_info  = []
    for n in range(nivel_min, nivel_max + 1):
        t0 = time.perf_counter()

        # Escalas naturais do nível
        L = fc.R(n)/8         # tamanho da nuvem inicial
        dt = 0.03 * fc.R(n)             # passo temporal
        t_max = lamb/2        # tempo total (alguns períodos)

        print(f"Testando estabilidade para n = {n}")

        ts, rs = simular_particulas(
            N=N_particulas,
            N_fields=n,
            dimensao_quadrado=L,
            r0=r0,
            dt=dt,
            t_max=t_max, func= funcao, temporal = temporal
        )
        rs_final = rs[-1, :, :]
        mean = np.mean(rs_final, axis=0)
        var = np.var(rs_final, axis=0, ddof =1 )
        matriz_covarianca = np.cov(rs_final.T, ddof=1)
        sigma2_total = np.mean(np.sum((rs_final - mean)**2, axis=1))
        dic_info = {"Media": mean, "Varianca": var, "Matriz de covariança": matriz_covarianca, "sigma total": sigma2_total, "L": L}
        nivel_info.append(dic_info)
        # rs tem shape (Nt, N, 2)
        Nt = rs.shape[0]
        t1 = time.perf_counter()
        plt.figure(figsize=(6, 6))

        plt.hist2d
        cores = plt.cm.viridis(np.linspace(0, 1, N_particulas))
        
        with open(nome_do_txt_info, 'a') as f:
            
            
            dic = dic_info
            f.write(f"Nível: {n}, Tamanho da núvem incial = {dic['L']}\n")
            f.write(f"Média: {dic['Media']}, Variança: {dic['Varianca']}\n")
            cov = dic["Matriz de covariança"]
            f.write(f"Sigma^2 raio  = {dic['sigma total']}")
            f.write('\n')
            f.write("Matriz de covariância:\n")
            f.write(
                f"[[{cov[0,0]:.4e}, {cov[0,1]:.4e}],\n"
                f" [{cov[1,0]:.4e}, {cov[1,1]:.4e}]]\n"
            )
            
            f.write('\n')
            f.write('=-='*10)
            f.write('\n')


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
        print(f"Tempo demorado: {t1 - t0}")
        print(f"Imagem salva em: {nome_arquivo}")
        plt.figure(figsize=(6,6))

        x = rs_final[:,0]
        y = rs_final[:,1]

        hist = plt.hist2d(
            x,
            y,
            bins=50,
            density=True
        )

        plt.colorbar(label="densidade")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Histograma 2D — nível {n}")

        plt.axis("equal")
        plt.legend()

        nome_hist = os.path.join(
            pasta_saida_info,
            f"hist2d_nivel_{n}_x{r0[0,0]:.3f}_y{r0[0,1]:.2f}.png"
        )

        plt.savefig(nome_hist, dpi=200)
        plt.close()

        print(f"Histograma salvo em: {nome_hist}")

    

    


