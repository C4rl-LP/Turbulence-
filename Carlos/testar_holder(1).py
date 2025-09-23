import sla 
import numpy as np
from matplotlib import pyplot as plt

def holder_mod_fixo_y(campo, x0, y0,xs):
    V0x, V0y,_,_,_ = campo(x0, y0)
    vals = []
    for x in xs:
        Vx, Vy, _, _, _ = campo(x, y0)
        vals.append(np.sqrt((V0x - Vx)**2 + (V0y - Vy)**2))
    return np.array(vals)

N = int(input('N= '))
campo_t = sla.Vector_plot_quarter_division(N, 1, 1, 2**(1- 1/3))
def campo_0(x,y):
    return campo_t.campos(x, y, 0, 0)

x0_h, y0_h = (np.sqrt(2), np.sqrt(2))

xs = np.linspace(x0_h, campo_t.R1 + x0_h, 200)

# diferenças
ys = holder_mod_fixo_y(campo_0, x0_h, y0_h, xs)


rs = np.abs(xs - x0_h)


mask = (rs > 0) & (ys > 0)
logr = np.log(rs[mask])
logd = np.log(ys[mask])


coef = np.polyfit(logr, logd, 1)
h_est = coef[0]

plt.figure()
plt.scatter(logr, logd, s=10, label="dados")
plt.plot(logr, np.polyval(coef, logr), 'r', label=f"ajuste: h≈{h_est:.3f}")
# reta de referência h=1/3 passando pela mediana
xm, ym = np.median(logr), np.median(logd)
plt.plot(logr, ym + (1/3)*(logr - xm), '--', label="h = 1/3 ref")
plt.xlabel("log |x-x0|")
plt.ylabel("log ||ΔV||")
plt.legend()
plt.show()

print("Estimativa de h =", h_est)
