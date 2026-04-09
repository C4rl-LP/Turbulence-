import sla
import numpy as np
from matplotlib import pyplot as plt
import os
from sla import gerar_tuplas 

import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def gerar_tuplas_hierarquico(x, y, max_n, R):
    """
    Retorna uma lista `levels` onde levels[k] é a lista de nós do nível k (k>=1).
    Cada nó é dict: {'init': (x_init,y_init), 'parent_idx': index_no_nivel_anterior}.
    level 1 contém o ponto base (x,y) com parent_idx = None.
    """
    levels = []
    # nível 1 = ponto base
    levels.append([{'init': (float(x), float(y)), 'parent_idx': None}])
    
    for k in range(2, max_n + 1):  # começa do nível 2
        prev = levels[k - 2]       # nível anterior
        new = []
        delta = float(R) / (2**(k-1))
        for p_idx, p in enumerate(prev):
            px, py = p['init']
            for dx in (delta, -delta):
                for dy in (delta, -delta):
                    child = (px + dx, py + dy)
                    new.append({'init': child, 'parent_idx': p_idx})
        levels.append(new)
    
    return levels
class Vector_plot_quarter_division_dt():
    def __init__(self, N_fields, R1, T1, lamb, omega = 1 ):
        self.N_fields = N_fields
        self.R1 = R1
        self.T1 = T1
        self.lamb = lamb
        self.omega = omega
    def phis_i(self, x,y,i, x0, y0):
        Ri = self.R1 * 2**(-i + 1)
        Ti = self.T1 * self.lamb**(-i + 1)
        r_norm = np.sqrt((x - x0)**2 + (y - y0)**2)
        return (2*np.pi*Ri**2/Ti) * np.exp(-r_norm**2/(2*Ri**2) + 1/2)

    def laplaciano_phis(self, x,y,i, x0, y0):
        Ri = self.R1 * 2**(-i + 1)
        Ti = self.T1 * self.lamb**(-i + 1)
        r2 = (x - x0)**2 + (y - y0)**2
        phi = (2*np.pi*Ri**2/Ti) * np.exp(-r2/(2*Ri**2) + 0.5)
        lap = phi * (r2/Ri**4 - 2/Ri**2)
        return lap
    def mover_r0(x0, y0, omega,t,i):
        if i == 1:
            return x0, y0
        if i > 1:
            return x0
    def campos(self, x, y, t,x0, y0):
        Vx_total = np.zeros_like(x, dtype=float)
        Vy_total = np.zeros_like(y, dtype=float)
        phis_total = np.zeros_like(x, dtype=float)
        lap_total = np.zeros_like(x, dtype=float)
        pontos = []

        level = gerar_tuplas_hierarquico(x0, y0, self.N_fields, self.R1)

        curr_position_prev = [level[0][0]['init']]
        
        for i in range(1, self.N_fields+1):
            k = i -1 
            node = level[k]
            
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


# versão hierárquica que escrevi antes

print(gerar_tuplas_hierarquico(0,0,3,1))
