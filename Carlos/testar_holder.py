import sla
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
import os
from sla import to_cpu  # importar a função auxiliar
import gc


def holder_mod_fixo_y(campo, x0, y0, xs):
    """
    xs: array de posições x (cupy)
    retorna vals (cupy) mesmo shape
    """
    # Garantir que xs é cupy array
    xs_cp = cp.asarray(xs)

    # Avalia vetor de uma só chamada (vetorizado no campo)
    V0x, V0y, *_ = campo(x0, y0)  # escalar (será 0-dim ou shape ())
    Vx, Vy, *_ = campo(xs_cp, cp.full_like(xs_cp, y0))

    dx = V0x - Vx
    dy = V0y - Vy

    vals = cp.sqrt(dx**2 + dy**2)
    return vals
def polyfit_gpu(x, y, deg=1):
    """Versão GPU do np.polyfit (usando mínimos quadrados)."""
    # Monta matriz de Vandermonde
    X = cp.vander(x, deg + 1)
    # Resolve (X^T X) a = X^T y
    A = X.T @ X
    b = X.T @ y
    coef = cp.linalg.solve(A, b)
    return coef  # do maior para o menor grau


def polyval_gpu(coef, x):
    """Avalia polinômio na GPU."""
    y = cp.zeros_like(x)
    for c in coef:
        y = y * x + c
    return y


def estima_holder(campo, x0, y0, R, npts=200, plot_testar=False, outdir="plots", N=None):
    rs = cp.logspace(cp.log10(2**(-N)), cp.log10(R), npts)
    
    xs = x0 + rs
    ys = holder_mod_fixo_y(campo, x0, y0, xs)

    mask = (rs > 0) & (ys > 0)
    if cp.sum(mask) < 5:
        return None

    logr = cp.log2(rs[mask])
    logd = cp.log2(ys[mask])

    coef = polyfit_gpu(logr, logd, 1)
    h_est = coef[0]


    if plot_testar:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # transferir dados para CPU antes do plot
        logr_cpu = cp.asnumpy(logr)
        logd_cpu = cp.asnumpy(logd)
        coef_cpu = cp.asnumpy(coef)

        plt.figure()
        plt.scatter(logr_cpu, logd_cpu, s=10, label="dados")
        plt.plot(logr_cpu, np.polyval(coef_cpu, logr_cpu), 'r', label=f"ajuste: h≈{float(h_est):.3f}")
        
        xm, ym = np.median(logr_cpu), np.median(logd_cpu)
        plt.plot(logr_cpu, ym + (1/3)*(logr_cpu - xm), '--', label="h = 1/3 ref")
        plt.xlabel("log₂|x-x₀|")
        plt.ylabel("log₂||ΔV||")
        plt.legend()
        fname = f"{outdir}/holder_N{N}_x{x0:.2f}_y{y0:.2f}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    return float(h_est)

# ---- Loop principal ----
Ns = range(1, 14)
cp.random.seed(1001)
def pontos_bola_2d(n_pontos=5, raio=1.0):
    """
    Gera pontos uniformemente distribuídos dentro de uma bola (círculo) 2D de raio 'raio'.
    Retorna uma lista de tuplas (x, y) no formato CuPy.
    """
    pts = []
    for _ in range(n_pontos):
        r = raio * cp.sqrt(cp.random.rand())         # distribuição uniforme na área
        theta = 2 * cp.pi * cp.random.rand()         # ângulo uniforme
        x = r * cp.cos(theta)
        y = r * cp.sin(theta)
        pts.append((x, y))
    return pts


pontos = pontos_bola_2d(n_pontos=15, raio=1.0)
plot_testar = True

with open("holder_results.txt", "w") as saida:
    saida.write("N, x0, y0, h_est\n")
    for N in Ns:
        print(f"N={N}: iniciando")
        try:
            campo_t = sla.Vector_plot_quarter_division(N, 1, 1, 2**(1 - 1/3))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"🚫 Falta de memória em N={N}")
            break
        print('fodasi')

        def campo_0(x, y):
            # Força tudo para GPU
            x_gpu = cp.asarray(x)
            y_gpu = cp.asarray(y)
            return campo_t.campos(x_gpu, y_gpu, 0, 0)

        for (x0, y0) in pontos:
            h = estima_holder(campo_0, x0, y0, R=campo_t.R1, npts=200, plot_testar=plot_testar, N=N)
            if h is not None:
                print(f"N={N}, ponto=({float(x0):.2f},{float(y0):.2f}), h≈{h:.3f}")
                saida.write(f"{N}, {float(x0):.5f}, {float(y0):.5f}, {h:.6f}\n")
            else:
                print(f"N={N}, ponto=({float(x0):.2f},{float(y0):.2f}) -- não pôde estimar")


        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        print(f"→ Memória da GPU liberada após N={N}\n")
print("Resultados salvos em holder_results.txt")


