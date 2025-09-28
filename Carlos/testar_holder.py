import sla
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
import os


def cp_polyfit(x, y, deg):
    # Matriz de Vandermonde no GPU
    X = cp.vander(x, deg + 1)  
    # Resolve mínimos quadrados (coef ~ np.polyfit)
    coef, *_ = cp.linalg.lstsq(X, y, rcond=None)
    return coef


def holder_mod_fixo_y(campo, x0, y0, xs):
    V0x, V0y, *_ = campo(cp.array(x0), cp.array(y0))
    Vx, Vy, *_ = campo(xs, cp.full_like(xs, y0))
    vals = cp.sqrt((V0x - Vx)**2 + (V0y - Vy)**2)
    return vals

def estima_holder(campo, x0, y0, R, npts=200, plot_testar=False, outdir="plots", N=None):
    rs = cp.logspace(cp.log10(2**(-N)), cp.log10(R), npts)
    xs = x0 + rs
    ys = holder_mod_fixo_y(campo, x0, y0, xs)

    mask = (rs > 0) & (ys > 0)
    if int(cp.sum(mask).get()) < 5:
        return None

    logr = cp.log2(rs[mask])
    logd = cp.log2(ys[mask])

    # ajuste linear no GPU (grau 1)
    A = cp.vstack([logr, cp.ones_like(logr)]).T
    coef, _, _, _ = cp.linalg.lstsq(A, logd, rcond=None)
    h_est = coef[0]


    if plot_testar:
        logr_cpu = cp.asnumpy(logr)
        logd_cpu = cp.asnumpy(logd)
        coef_cpu = cp.asnumpy(coef)
        fit_vals = np.polyval(coef_cpu, logr_cpu)

        plt.figure()
        plt.scatter(logr_cpu, logd_cpu, s=10, label="dados")
        plt.plot(logr_cpu, fit_vals, 'r', label=f"ajuste: h≈{h_est:.3f}")
        xm = np.median(logr_cpu)
        ym = np.median(logd_cpu)
        plt.plot(logr_cpu, ym + (1/3)*(logr_cpu - xm), '--', label="h = 1/3 ref")
        plt.xlabel("log_2|x-x0|")
        plt.ylabel("log_2||ΔV||")
        plt.legend()
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = f"{outdir}/holder_N{N}_x{x0:.2f}_y{y0:.2f}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    return h_est

# parâmetros da varredura
Ns = range(1, 13)
pontos = [(cp.sqrt(2)/4, cp.pi/5), (cp.sqrt(2), cp.pi), (cp.pi/10, cp.e/10)]
plot_testar = False

saida = open("holder_results.txt", "w")
saida.write("N, x0, y0, h_est\n")

for N in Ns:
    campo_t = sla.Vector_plot_quarter_division(N, 1, 1, 2**(1-1/3))
    def campo_0(x, y):
        return campo_t.campos(x, y, 0, 0)

    for (x0, y0) in pontos:
        h = estima_holder(campo_0, x0, y0, R=campo_t.R1, npts=200, plot_testar=plot_testar, N=N)
        if h is not None:
            print(f"N={N}, ponto=({x0:.2f},{y0:.2f}), h≈{h:.3f}")
            saida.write(f"{N}, {x0:.5f}, {y0:.5f}, {h:.6f}\n")
        else:
            print(f"N={N}, ponto=({x0:.2f},{y0:.2f}) -- não pôde estimar")

saida.close()
print("Resultados salvos em holder_results.txt")
