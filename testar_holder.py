import sla
import numpy as np
from matplotlib import pyplot as plt
import os

def holder_mod_fixo_y(campo, x0, y0, xs):
    V0x, V0y, _, _, _ = campo(x0, y0)
    vals = []
    for x in xs:
        Vx, Vy, _, _, _ = campo(x, y0)
        vals.append(np.sqrt((V0x - Vx)**2 + (V0y - Vy)**2))
    return np.array(vals)

def estima_holder(campo, x0, y0, R, npts=250, plot_testar=False, outdir="plots", N=None):
    xs = np.linspace(x0, x0 + R, npts)
    ys = holder_mod_fixo_y(campo, x0, y0, xs)
    rs = np.abs(xs - x0)

    mask = (rs > 0) & (ys > 0)
    if np.sum(mask) < 5:  # poucos pontos válidos
        return None

    logr = np.log(rs[mask])
    logd = np.log(ys[mask])
    coef = np.polyfit(logr, logd, 1)
    h_est = coef[0]

    # se quiser salvar o gráfico
    if plot_testar:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        plt.figure()
        plt.scatter(logr, logd, s=10, label="dados")
        plt.plot(logr, np.polyval(coef, logr), 'r', label=f"ajuste: h≈{h_est:.3f}")
        
        xm, ym = np.median(logr), np.median(logd)
        plt.plot(logr, ym + (1/3)*(logr - xm), '--', label="h = 1/3 ref")
        plt.xlabel("log |x-x0|")
        plt.ylabel("log ||ΔV||")
        plt.legend()
        fname = f"{outdir}/holder_N{N}_x{x0:.2f}_y{y0:.2f}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

    return h_est

# parâmetros da varredura
Ns = range(1, 12)   # valores de N a testar
pontos = [(np.sqrt(2)/4, np.pi/5), (np.sqrt(2), np.pi)]  # pontos de teste
plot_testar = True  # <<< só muda aqui para ativar/desativar os plots

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

