from matplotlib import pyplot as plt
import numpy as cp 
from itertools import product

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================
class solve_RK4():
    """
    O objetivo dessa classe é resolver uma edo da forma. 
    r' = V(t, r)
    r(0) = r0
    """

    def __init__(self, func, r0, t0, dt, t_max):
        self.func = func
        self.r = [r0]
        self.t = [t0]
        self.dt = dt
        self.t_max = t_max
    def proximo_passo(self, i):
        ri = self.r[i]
        ti = self.t[i]
        dt = self.dt
        k1 = self.func(ti, ri) 
        k2 = self.func(ti + dt/2, ri + dt/2 *k1)
        k3 = self.func(ti + dt/2, ri + dt/2 * k2)
        k4 = self.func(ti + dt, ri + dt*k3)
        return ri + dt/6 * (k1 + 2*k2 +  2*k3 + k4)
    def fazer(self):
        dt = self.dt
        i= 0
        while self.t[-1] < self.t_max:
            self.r.append(self.proximo_passo(i))
            self.t.append(self.t[i] + dt)
            i += 1 
        return cp.array(self.t), cp.vstack(self.r)

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================
def gerar_tuplas(x, y, n, R):
    if n < 1:
        return [(x, y)]
    deslocs = [[R/(2**i), -R/(2**i)] for i in range(1, n+1)]
    combos = list(product(*deslocs))  
    return [(x + sum(dx), y + sum(dy)) for dx in combos for dy in combos]

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================
class Vector_plot_quarter_division():
    def __init__(self, N_fields, R1, T1, lamb):
        self.N_fields = N_fields
        self.R1 = R1
        self.T1 = T1
        self.lamb = lamb

    def phis_i(self, x,y,i, x0, y0):
        Ri = self.R1 * 2**(-i + 1)
        Ti = self.T1 * self.lamb**(-i + 1)
        r_norm = cp.sqrt((x - x0)**2 + (y - y0)**2)
        return (2*cp.pi*Ri**2/Ti) * cp.exp(-r_norm**2/(2*Ri**2) + 1/2)

    def laplaciano_phis(self, x,y,i, x0, y0):
        Ri = self.R1 * 2**(-i + 1)
        Ti = self.T1 * self.lamb**(-i + 1)
        r2 = (x - x0)**2 + (y - y0)**2
        phi = (2*cp.pi*Ri**2/Ti) * cp.exp(-r2/(2*Ri**2) + 0.5)
        lap = phi * (r2/Ri**4 - 2/Ri**2)
        return lap

    def campos(self, x, y, x0, y0):
        Vx_total = cp.zeros_like(x, dtype=float)
        Vy_total = cp.zeros_like(y, dtype=float)
        phis_total = cp.zeros_like(x, dtype=float)
        lap_total = cp.zeros_like(x, dtype=float)
        pontos = []
        for i in range(1, self.N_fields+1):
            pts = gerar_tuplas(x0, y0, i - 1, self.R1)
            for j, (x0i, y0i) in enumerate(pts):
                ix = j % (2**(i-1))   
                iy = j // (2**(i-1))  
                s = 1 if (ix + iy) % 2 == 0 else -1  
                Ri = self.R1 * 2**(-i + 1)

                phis = self.phis_i(x, y,i, x0i, y0i)
                lap  = self.laplaciano_phis(x, y, i, x0i, y0i)

                phis_total += phis
                lap_total  += lap
                Vx_total += s * (-(y - y0i) / Ri**2) * phis
                Vy_total += s * ((x - x0i) / Ri**2) * phis
            pontos = pontos + pts
        return Vx_total, Vy_total, pontos, phis_total, lap_total

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================

x0, y0 = (0,0)    
N_campos = 1
R1 = 1.0
T1 = 1.0
lamb = 2**(2/3)
campo = Vector_plot_quarter_division(N_fields=N_campos, R1=R1, T1=T1, lamb=lamb)

n = 50
x = cp.linspace(-1.5*campo.R1 + x0, 1.5*campo.R1 + x0, n)
y = cp.linspace(-1.5*campo.R1 + y0, 1.5*campo.R1 + y0, n)
X, Y = cp.meshgrid(x, y)

all_Vx, all_Vy, pts, phis_tot, lap_tot = campo.campos(X, Y, x0, y0)

def func(t, r):
    Vx, Vy, _, _, _ = campo.campos(r[0], r[1], x0, y0)   
    return cp.array([Vx, Vy], dtype=float)

plot = False
resolver_teste = False
plot_centros = False
if resolver_teste:
    ponto_inicial = [-.25, 0]
    t0 = 0
    dt = 0.001
    t_max = 1.5
    resolver = solve_RK4(func, ponto_inicial, t0, dt, t_max)
    t_selved, r_solved = resolver.fazer()

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================

if plot:
    magnitude = cp.hypot(all_Vx, all_Vy)
    xs, ys = zip(*pts)  
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    # 1) Campo vetorial + trajetória RK4
    ax = axes[0]
    if resolver_teste:
        x_sol = r_solved[:, 0]
        y_sol = r_solved[:, 1]
        ax.plot(x_sol, y_sol, "r-", lw=2, label="trajetória RK4")
        ax.plot(x_sol[0], y_sol[0], "go", label="início")
        ax.plot(x_sol[-1], y_sol[-1], "ro", label="fim")
    strm = ax.streamplot(X, Y, all_Vx, all_Vy, color=magnitude, cmap="plasma", density=2, linewidth=1)
    if plot_centros:
        ax.scatter(xs, ys, color="black", s=1, label="centros $\\phi_i$")
    ax.set_title(f"Campo vetorial e trajetória (N = {N_campos})")
    ax.axis("equal")
    ax.legend()
    fig.colorbar(strm.lines, ax=ax).set_label("|V|")

    # 2) Módulo do campo vetorial
    ax = axes[1]
    cont_mag = ax.contourf(X, Y, magnitude, levels=100, cmap="plasma")
    ax.set_title("Módulo do campo vetorial $|V(x, y)|$")
    ax.axis("equal")
    fig.colorbar(cont_mag, ax=ax).set_label("|V|")

    # 3) Campo escalar total
    ax = axes[2]
    cont_phi = ax.contourf(X, Y, phis_tot, levels=100, cmap="viridis")
    ax.set_title("Campo escalar total $\\sum_i \\phi_i(x, y)$")
    ax.axis("equal")
    fig.colorbar(cont_phi, ax=ax).set_label("$\\phi_{\\mathrm{total}}$")

    # 4) Laplaciano do campo escalar total
    ax = axes[3]
    cont_lap = ax.contourf(X, Y, lap_tot, levels=100, cmap="inferno")
    ax.set_title("Laplaciano $\\nabla^2 \\sum_i \\phi_i(x, y)$")
    ax.axis("equal")
    fig.colorbar(cont_lap, ax=ax).set_label("$\\nabla^2 \\phi_{\\mathrm{total}}$")

    plt.suptitle("Visualização do Campo Vetorial, Escalar e Laplaciano", fontsize=16)
    plt.show()


#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================



class teste_estabilidade:
    def __init__ (self,R1= 1.0, T1 = 1.0, lamb = 2**(2/3)):
        
        self.R1 = R1
        self.T1 = T1
        self.lamb = lamb
        
    
    def rodar(self, N_fields, r0, delta, densidade_vizinhos, t_max, dt_base = 0.01, escala_dt= 0.5 ):
        campo = Vector_plot_quarter_division(N_fields, self.R1, self.T1, self.lamb)

        # Ajusta dt conforme N
        dt = dt_base * (escala_dt**N_fields)

        # Função da EDO
        def func(t, r):
            Vx, Vy, _, _, _ = campo.campos(r[0], r[1], 0, 0)
            return cp.array([Vx, Vy], dtype=float)
        print('campo feito')
        # Trajetória central
        rk_central = solve_RK4(func, cp.array(r0, dtype=float), 0, dt, t_max)
        t_central, r_central = rk_central.fazer()
        print('central resolvido')
        trajetorias_vizinhos = []
        x0, y0 = r0
        n_vizinhos = int(densidade_vizinhos *cp.pi*delta**2 )
        for _ in range(n_vizinhos):
            
            theta = cp.random.uniform(0, 2*cp.pi)
            r = delta * cp.sqrt(cp.random.uniform(0, 1)) 
            dx = r * cp.cos(theta)
            dy = r * cp.sin(theta)

            pos_viz = cp.array([x0 + dx, y0 + dy], dtype=float)
            rk_viz = solve_RK4(func, pos_viz, 0, dt, t_max)
            _, r_viz = rk_viz.fazer()
            trajetorias_vizinhos.append(r_viz)
        return {
            "N_fields": N_fields,
            "t": t_central,
            "central": r_central,
            "vizinhos": trajetorias_vizinhos,
            "dt_usado": dt,
            "n_vizinhos": n_vizinhos
        }
tester = teste_estabilidade(R1=1.0, T1=1.0, lamb=2**(2/3))

pos0 = (-0.25, 0)
delta = 0.02
densidade_vizinhos = 10000
t_max = 1.5

resultados = []
for N in [1]:
    res = tester.rodar(N, pos0, delta, densidade_vizinhos, t_max, dt_base=0.01, escala_dt=0.8)
    resultados.append(res)
    print(f"N={N}, dt={res['dt_usado']:.5f}, N vizinhos = {res['n_vizinhos']}")
fig, axes = plt.subplots(1, len(resultados), figsize=(8, 4), sharex=True, sharey=True)

if len(resultados) == 1:
    axes = [axes]

for ax, res in zip(axes, resultados):
    circle = plt.Circle(pos0, delta,color='green', fill=False, lw=1, label=f"Raio δ={delta}")
    ax.add_patch(circle)

    ax.plot(res["central"][:,0], res["central"][:,1], "r-", lw=3, label="central")
    ax.plot(res["central"][0,0], res["central"][0,1], "ro", label="início central", color = 'green', markersize = 4)

    for traj in res["vizinhos"]:
        ax.plot(traj[:,0], traj[:,1], "b-", alpha=0.4)
        ax.plot(traj[0,0], traj[0,1], "go", markersize=3)

    ax.set_title(f"N={res['N_fields']}\n dt={res['dt_usado']:.3e}")
    ax.set_aspect('equal', adjustable='box')  # <- aqui está a correção
    ax.legend()

plt.suptitle("Trajetórias com vizinhos no círculo de raio δ", fontsize=16)
plt.show()
