from matplotlib import pyplot as plt
import numpy as np
import campos 
from funcoes_para_centros import R_1, alpha_padrao, c_padrao, T_1, lamb

from matplotlib.colors import LogNorm


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
t = 0.0
N = 8
x0 = 0
y0 = 0


#plot_campo(t=t, N_fields=N, L=1.4,x_0 = x0, y_0 =y0, resolucao=200, func=campos.campo_total_podado)
#plot_campo(t=t, N_fields=N, L=1.4,x_0 = x0, y_0 =y0,resolucao=200, func=campos.campo_total_podado)

def salvar_por_nivel(
    Nmax,
    t=0.0,
    L=.25,
    x_0=0,
    y_0=0,
    resolucao=1000,
    func=None,
    pasta_name="campos",
    N_min = 1
):
    
    import os
    pasta = pasta_name + f'_{c_padrao}_{alpha_padrao:.3f}'
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
            X, Y,
            Vx, Vy,
            color=V,
            cmap="viridis",
            density=2.0,
            norm=LogNorm(
                vmin=max(V[V>0].min(),1e-12),
                vmax=V.max()
            )
        )

        parametros = (
        rf"$R_1 = {R_1}$" "\n"
        rf"$T_1 = {T_1}$" "\n"
        rf"$\lambda = {lamb:.3f}$" "\n"
        rf"$\alpha = {alpha_padrao:.3f}$" "\n"
        rf"$c = {c_padrao}$" "\n"
        rf"$t = {t}$"
        )
        ax.text(
            0.02, 0.98,                 # posição relativa
            parametros,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.85
            )
        )
        fig.colorbar(stream.lines,label="|V|")

        ax.set_title(f"Campo vetorial — N = {N}")
        ax.set_aspect('equal')
        
        nome = f"{pasta}/campo_nivel_{N:02d}.png"

        plt.tight_layout(rect=[0,0,0.85,1])
        plt.savefig(nome,dpi=300)
        plt.close(fig)

        print(f"salvo: {nome}")

def salvar_por_nivel(
    Nmax,
    t=0.0,
    L=.25,
    x_0=0,
    y_0=0,
    resolucao=200,
    func=None,
    pasta_name="campos",
    N_min = 1
):
    
    import os
    pasta = pasta_name + f'_{c_padrao}_{alpha_padrao:.3f}'
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
        norma = np.sqrt(Vx**2 + Vy**2)

        Vx_plot = Vx/(norma + 1e-12)
        Vy_plot = Vy/(norma + 1e-12)
        fig, ax = plt.subplots(figsize=(8,8))

        stream = ax.streamplot(
            X, Y,
            Vx_plot, Vy_plot,
            color=np.log10(norma + 1e-12),
            cmap="viridis",
            density=3
        )
        parametros = (
        rf"$R_1 = {R_1}$" "\n"
        rf"$T_1 = {T_1}$" "\n"
        rf"$\lambda = {lamb:.3f}$" "\n"
        rf"$\alpha = {alpha_padrao:.3f}$" "\n"
        rf"$c = {c_padrao}$" "\n"
        rf"$t = {t}$"
        )
        ax.text(
            0.02, 0.98,                 # posição relativa
            parametros,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.85
            )
        )
        fig.colorbar(stream.lines,label="|V|")

        ax.set_title(f"Campo vetorial — N = {N}")
        ax.set_aspect('equal')
        
        nome = f"{pasta}/campo_nivel_{N:02d}.png"

        plt.tight_layout(rect=[0,0,0.85,1])
        plt.savefig(nome,dpi=300)
        plt.close(fig)

        print(f"salvo: {nome}")

salvar_por_nivel(
    Nmax=10,
    t=0,
    func=campos.campo_total_podado,
    N_min= 1

)