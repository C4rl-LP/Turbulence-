from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from itertools import product
from joblib import Parallel, delayed

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
class Vector_plot_quarter_division:
    def __init__(self, N_fields, R1, T1, lamb):
        self.N_fields = N_fields
        self.R1 = R1
        self.T1 = T1
        self.lamb = lamb

        # Pré-computa todos os deslocamentos para cada i e também
        # cria lista flatten de todos os centros com suas flags 's' (±1)
        centros = []
        s_flags = []
        idx_offset = 0
        for i in range(1, N_fields+1):
            pts = gerar_tuplas(0, 0, i-1, R1)
            # calcular s para cada ponto j
            npts_i = len(pts)
            for j, (x0i, y0i) in enumerate(pts):
                ix = j % (2**(i-1))
                iy = j // (2**(i-1))
                s = 1 if (ix + iy) % 2 == 0 else -1
                centros.append((x0i, y0i, i, s))
        # converte para arrays de cupy para vetorização rápida
        if len(centros) > 0:
            arr = cp.asarray(centros)  # shape (M,4) dtype=object -> we'll extract columns
            # but safer to build typed arrays:
            self._centers_x = cp.array([c[0] for c in centros], dtype=float)
            self._centers_y = cp.array([c[1] for c in centros], dtype=float)
            self._centers_i = cp.array([int(c[2]) for c in centros], dtype=int)
            self._centers_s = cp.array([int(c[3]) for c in centros], dtype=int)
        else:
            self._centers_x = cp.array([], dtype=float)
            self._centers_y = cp.array([], dtype=float)
            self._centers_i = cp.array([], dtype=int)
            self._centers_s = cp.array([], dtype=int)

        # keep a python list of pontos for legacy plotting (CPU-friendly)
        self.pontos_all = [(float(cx), float(cy)) for (cx, cy, *_ ) in centros] if len(centros) else []

    def phis_i_scalar(self, r2, Ri, Ti):
        # r2 = (x-x0)^2 + (y-y0)^2  (can be array)
        return (2*cp.pi*Ri**2 / Ti) * cp.exp(-r2/(2*Ri**2) + 0.5)

    def laplaciano_phis_scalar(self, r2, Ri, Ti):
        phi = (2*cp.pi*Ri**2 / Ti) * cp.exp(-r2/(2*Ri**2) + 0.5)
        lap = phi * (r2/Ri**4 - 2/Ri**2)
        return lap
    def campos(self, x, y, x0=0.0, y0=0.0, block_size=200_000):
        """
        Vetorizado com chunking por centros (blocks). Compatível com:
        Vx, Vy, pontos, phis_tot, lap_tot = campo.campos(X, Y, x0, y0)

        - x, y podem ser escalares, vetores ou arrays (ex.: meshgrid).
        - retorna phis_tot e lap_tot com a mesma shape que x/y.
        - usa os arrays pré-computados: self._centers_x, self._centers_y,
        self._centers_i, self._centers_s e self.pontos_all.
        """
        # garantir cupy arrays e formas
        x_cp = cp.asarray(x, dtype=cp.float64)
        y_cp = cp.asarray(y, dtype=cp.float64)

        # salvar formato original para reshape no retorno
        orig_shape = x_cp.shape if x_cp.ndim > 0 else ()
        x_flat = x_cp.ravel() if x_cp.size != 1 else cp.array([x_cp.item()])
        y_flat = y_cp.ravel() if y_cp.size != 1 else cp.array([y_cp.item()])
        P = x_flat.size

        # se não há centros definidos, retorna zeros + pontos vazios
        if getattr(self, "_centers_x", None) is None or self._centers_x.size == 0:
            zeros = cp.zeros_like(x_flat, dtype=cp.float64).reshape(orig_shape)
            return zeros, zeros, self.pontos_all, zeros, zeros

        # arrays dos centros e parâmetros por centro
        cx_all = self._centers_x.astype(cp.float64)  # (M,)
        cy_all = self._centers_y.astype(cp.float64)
        ci_all = self._centers_i.astype(cp.int32)
        cs_all = self._centers_s.astype(cp.float64)  # sinais ±1

        # computa Ri e Ti por centro (vetor de tamanho M)
        Ri_all = self.R1 * (2.0 ** (-ci_all + 1))
        Ti_all = self.T1 * (self.lamb ** (-ci_all + 1))

        # iniciais (flatten)
        Vx_acc = cp.zeros(P, dtype=cp.float64)
        Vy_acc = cp.zeros(P, dtype=cp.float64)
        phis_acc = cp.zeros(P, dtype=cp.float64)
        lap_acc  = cp.zeros(P, dtype=cp.float64)

        M = cx_all.size
        # processa centros em blocos para controlar uso de VRAM
        for start in range(0, M, block_size):
            end = min(start + block_size, M)
            cx = cx_all[start:end]       # (m,)
            cy = cy_all[start:end]
            ci = ci_all[start:end]
            cs = cs_all[start:end]
            Ri = Ri_all[start:end]
            Ti = Ti_all[start:end]

            # broadcasting: (P, m)
            dx = x_flat[:, None] - cx[None, :]
            dy = y_flat[:, None] - cy[None, :]
            r2 = dx**2 + dy**2

            # prefatores por centro (1, m)
            pref = (2.0 * cp.pi * (Ri**2) / Ti)[None, :]   # (1,m)
            inv_Ri2 = (1.0 / (Ri**2))[None, :]             # (1,m)
            Ri4 = (Ri**4)[None, :]
            Ri2 = (Ri**2)[None, :]

            # phi e laplaciano (P,m)
            exponent = -r2 / (2.0 * Ri2) + 0.5
            phi_block = pref * cp.exp(exponent)
            lap_block = phi_block * (r2 / Ri4 - 2.0 / Ri2)

            # sinais (1,m)
            s_block = cs[None, :]

            # contribuição para Vx, Vy
            Vx_block = cp.sum(s_block * ( -dy * inv_Ri2 ) * phi_block, axis=1)
            Vy_block = cp.sum(s_block * (  dx * inv_Ri2 ) * phi_block, axis=1)

            # acumular
            Vx_acc += Vx_block
            Vy_acc += Vy_block
            phis_acc += cp.sum(phi_block, axis=1)
            lap_acc  += cp.sum(lap_block, axis=1)

            # liberar memória temporária do bloco
            del dx, dy, r2, pref, inv_Ri2, Ri4, Ri2, exponent, phi_block, lap_block, s_block, Vx_block, Vy_block
            cp.get_default_memory_pool().free_all_blocks()

        # acrescentar offsets x0,y0 se necessário (se você realmente usa x0,y0 para deslocar o campo)
        # aqui x0,y0 foram passados mas não usados nos centros (centros já estão em coords globais).
        # Se quiser aplicar um deslocamento global, descomente:
        # Vx_acc += 0.0
        # Vy_acc += 0.0

        # reshape para forma original
        Vx_out = Vx_acc.reshape(orig_shape)
        Vy_out = Vy_acc.reshape(orig_shape)
        phis_out = phis_acc.reshape(orig_shape)
        lap_out  = lap_acc.reshape(orig_shape)

        return Vx_out, Vy_out, self.pontos_all, phis_out, lap_out




def to_cpu(arr):
    """Converte array GPU → CPU automaticamente se necessário"""
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================
if __name__ == '__main__':
    plot =  True
    resolver_teste = False
    if plot or resolver_teste:
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
            
        
        def rodar(self, N_fields, r0, delta, densidade_vizinhos, t_max, dt_base = 0.01, escala_dt= 0.5,  n_jobs=-1 ):
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
            def simular_vizinho(seed=None):
                if seed is not None:
                    cp.random.seed(seed)  # garante diversidade nos processos paralelos
                theta = cp.random.uniform(0, 2*cp.pi)
                r = delta * cp.sqrt(cp.random.uniform(0, 1)) 
                dx = r * cp.cos(theta)
                dy = r * cp.sin(theta)
                pos_viz = cp.array([x0 + dx, y0 + dy], dtype=float)
                rk_viz = solve_RK4(func, pos_viz, 0, dt, t_max)
                _, r_viz = rk_viz.fazer()
                return r_viz

            # Paralelização com joblib
            trajetorias_vizinhos = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(simular_vizinho)(seed) for seed in range(n_vizinhos)
            )

            return {
                "N_fields": N_fields,
                "t": t_central,
                "central": r_central,
                "vizinhos": trajetorias_vizinhos,
                "dt_usado": dt,
                "n_vizinhos": n_vizinhos
            }
    testar = False
    if testar:
        tester = teste_estabilidade(R1=1.0, T1=1.0, lamb=2**(2/3))

        pos0 = (-0.5, -0.5)
        delta = 0.02
        densidade_vizinhos = 10/(cp.pi*delta**2)
        t_max = 0.5
        n = int(input())
        resultados = []
        for N in [n]:
            res = tester.rodar(N, pos0, delta, densidade_vizinhos, t_max, dt_base=0.005, escala_dt=0.8)
            resultados.append(res)
            print(f"N={N}, dt={res['dt_usado']:.5f}, N vizinhos = {res['n_vizinhos']}")
        fig, axes = plt.subplots(1, len(resultados), figsize=(8, 4), sharex=True, sharey=True)

        if len(resultados) == 1:
            axes = [axes]

        for ax, res in zip(axes, resultados):
            circle = plt.Circle(pos0, delta,color='green', fill=False, lw=1, label=f"Raio δ={delta}")
            ax.add_patch(circle)

            

            for traj in res["vizinhos"]:
                ax.plot(traj[:,0], traj[:,1], "b-", alpha=0.4)
                ax.plot(traj[0,0], traj[0,1], "go", markersize=3, color = 'green')
            ax.plot(res["central"][:,0], res["central"][:,1], "r-", lw=3, label="central")
            ax.plot(res["central"][0,0], res["central"][0,1], "ro", label="início central", color = 'black', markersize = 4)

            ax.set_title(f"N={res['N_fields']}\n Início em: {pos0}")
            ax.set_aspect('equal', adjustable='box')  # <- aqui está a correção
            ax.legend()

        plt.suptitle("Trajetórias com vizinhos no círculo de raio δ", fontsize=16)
        plt.show()

