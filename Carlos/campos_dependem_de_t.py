import sla
import numpy as cp
from matplotlib import pyplot as plt
import os
from sla import gerar_tuplas 

class Vector_plot_quarter_division_dt():
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
    def mover_r0(x0, y0, omega,t,i):
        if i == 1:
            return x0, y0
        if i > 1:
            return x0
    def campos(self, x, y, t,x0, y0):
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