from matplotlib import pyplot as plt
import numpy as np
import campos 
import funcoes_para_centros as fc



def plot_campo(t=0.0, N_fields=3, L=5,x_0=  0, y_0= 0, resolucao=200, func=None):

    x = np.linspace(-L+x_0, L+x_0, resolucao)
    y = np.linspace(-L+y_0, L +y_0, resolucao)

    X, Y = np.meshgrid(x, y)

    # achata a malha em lista de pontos
    xp = X.ravel()
    yp = Y.ravel()

    # sua função recebe vetores 1D como antes
    Vx, Vy = func(xp, yp, t, N_fields)

    # remonta para grade 2D
    Vx = Vx.reshape(X.shape)
    Vy = Vy.reshape(Y.shape)

    V = np.sqrt(Vx**2 + Vy**2)

    plt.figure(figsize=(8,8))

    plt.streamplot(
        X, Y,
        Vx, Vy,
        color=V,
        cmap="viridis",
        density=2.5
    )

    plt.colorbar(label='|V|')
    plt.axis('equal')
    plt.show()
def plot_campo_e_intensidade(N, t=0.0, L=1.5, x_0=0.0, y_0=0.0, resolucao=200, func=campos.campo_total_podado, caminho_salvar=None):
    """
    Plota o campo de velocidade de nível N com seus centros correspondentes
    e, ao lado, plota a intensidade do campo |V|^2.
    """
    x = np.linspace(-L+x_0, L+x_0, resolucao)
    y = np.linspace(-L+y_0, L+y_0, resolucao)

    X, Y = np.meshgrid(x, y)

    xp = X.ravel()
    yp = Y.ravel()

    # Calcula o campo de velocidade
    Vx, Vy = func(xp, yp, t, N)

    # Remonta para grade 2D
    Vx = Vx.reshape(X.shape)
    Vy = Vy.reshape(Y.shape)

    V = np.sqrt(Vx**2 + Vy**2)
    V2 = Vx**2 + Vy**2  # Intensidade |V|^2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Campo de Velocidade e Centros ---
    stream = ax1.streamplot(
        X, Y,
        Vx, Vy,
        color=V,
        cmap="viridis",
        density=2.0
    )
    fig.colorbar(stream.lines, ax=ax1, label='|V|')
    ax1.set_title(f"Campo de Velocidade (Nível {N})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect('equal')

    # --- Plot 2: Intensidade |V|^2 e Centros ---
    im = ax2.pcolormesh(X, Y, V2, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax2, label=r'$|V|^2$')
    ax2.set_title(f"Intensidade $|V|^2$ (Nível {N})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal')

    # --- Plotar os Centros ---
    if N <= 8:
        # Plotar os centros de todos os níveis de 1 a N
        for n in range(1, N + 1):
            centros = fc.centros_nivel_todos(n, t)
            # Filtra apenas os centros que estão dentro dos limites de visualização
            mask_x = (centros[:, 0] >= -L + x_0) & (centros[:, 0] <= L + x_0)
            mask_y = (centros[:, 1] >= -L + y_0) & (centros[:, 1] <= L + y_0)
            centros_visiveis = centros[mask_x & mask_y]

            if len(centros_visiveis) > 0:
                # O tamanho diminui para níveis maiores
                size = max(10, 80 - 10 * n)
                
                # Plot dos centros no campo de velocidade
                ax1.scatter(
                    centros_visiveis[:, 0], centros_visiveis[:, 1],
                    label=f"Centros Nível {n}" if (N <= 4 or n in [1, N]) else None,
                    s=size, edgecolors='black', linewidths=0.5, zorder=10
                )
                
                # Plot dos centros na intensidade
                ax2.scatter(
                    centros_visiveis[:, 0], centros_visiveis[:, 1],
                    color="cyan" if n == N else "white",
                    marker="x" if n == N else "o",
                    s=size, edgecolors=None if n == N else 'black', linewidths=1, zorder=10
                )
        ax1.legend(loc="upper right")
    else:
        print(f"Aviso: N={N} é muito grande. O plot dos centros foi omitido para evitar lentidão.")

    plt.tight_layout()
    if caminho_salvar:
        import os
        from config import obter_caminho_config
        
        if isinstance(caminho_salvar, bool) and caminho_salvar:
            pasta_dest = obter_caminho_config(subpasta="campos")
            func_name = func.__name__ if func is not None else "desconhecido"
            nome_arquivo = f"campo_e_intensidade_nivel_{N:02d}_t{t:.2f}_L{L:.2f}_x0_{x_0:.2f}_y0_{y_0:.2f}_{func_name}.png"
            caminho_final = os.path.join(pasta_dest, nome_arquivo)
        elif os.path.isdir(caminho_salvar):
            func_name = func.__name__ if func is not None else "desconhecido"
            nome_arquivo = f"campo_e_intensidade_nivel_{N:02d}_t{t:.2f}_L{L:.2f}_x0_{x_0:.2f}_y0_{y_0:.2f}_{func_name}.png"
            caminho_final = os.path.join(caminho_salvar, nome_arquivo)
        elif not caminho_salvar.endswith('.png'):
            os.makedirs(caminho_salvar, exist_ok=True)
            func_name = func.__name__ if func is not None else "desconhecido"
            nome_arquivo = f"campo_e_intensidade_nivel_{N:02d}_t{t:.2f}_L{L:.2f}_x0_{x_0:.2f}_y0_{y_0:.2f}_{func_name}.png"
            caminho_final = os.path.join(caminho_salvar, nome_arquivo)
        else:
            caminho_final = caminho_salvar
            
        os.makedirs(os.path.dirname(caminho_final), exist_ok=True)
        plt.savefig(caminho_final, dpi=300)
        print(f"Gráfico salvo em: {caminho_final}")
    else:
        plt.show()


def salvar_por_nivel(
    Nmax,
    t=0.0,
    L=.25,
    x_0=.3,
    y_0=.3,
    resolucao=200,
    func=None,
    pasta=None,
    N_min = 1
):
    from config import obter_caminho_config
    if pasta is None:
        pasta = obter_caminho_config(subpasta="campos")

    import os
    os.makedirs(pasta, exist_ok=True)

    for N in range(N_min , Nmax+1):

        x = np.linspace(-L+x_0, L+x_0, resolucao)
        y = np.linspace(-L+y_0, L+y_0, resolucao)

        X, Y = np.meshgrid(x,y)

        xp = X.ravel()
        yp = Y.ravel()

        Vx, Vy = func(xp, yp, t, N)

        Vx = Vx.reshape(X.shape)
        Vy = Vy.reshape(Y.shape)

        V = np.sqrt(Vx**2 + Vy**2)

        fig, ax = plt.subplots(figsize=(8,8))

        stream = ax.streamplot(
            X,Y,
            Vx,Vy,
            color=V,
            cmap="viridis",
            density=2.5
        )

        fig.colorbar(stream.lines,label="|V|")

        ax.set_title(f"Nível {N}")
        ax.set_aspect('equal')

        func_name = func.__name__ if func is not None else "desconhecido"
        nome = os.path.join(pasta, f"campo_nivel_{N:02d}_t{t:.2f}_L{L:.2f}_x0_{x_0:.2f}_y0_{y_0:.2f}_res{resolucao}_{func_name}.png")

        plt.tight_layout()
        plt.savefig(nome,dpi=300)
        plt.close(fig)

        print(f"salvo: {nome}")


if __name__ == "__main__":
    t = 0.0
    N = 8
    x0 = 0
    y0 = 0

    # plot_campo_e_intensidade(N=3, t=t, L=1.5, x_0=x0, y_0=y0, func=campos.campo_total_podado)

    # plot_campo(t=t, N_fields=N, L=1.4,x_0 = x0, y_0 =y0, resolucao=200, func=campos.campo_total_podado)
    '''
    salvar_por_nivel(
        Nmax=20,
        t=0,
        func=campos.campo_total_podado,
        N_min= 9
    )'''
    plot_campo_e_intensidade(N=3,
        t=0.20,
        func=campos.campo_total_podado,
        resolucao=400, caminho_salvar= True
    )
   
    #plot_campo(t =0, N_fields= 17, L=.25, x_0=0.25, y_0= 0.25, func=campos.campo_total_podado)