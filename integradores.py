import numpy as np 

def solve_RK4(func, r0, t0, dt, t_max):
    """
    Resolve o sistema dr/dt = func(t, r) usando RK4.

    Parâmetros:
    - func : função do campo vetorial
    - r0   : condição inicial (array)
    - t0   : tempo inicial
    - dt   : passo de tempo
    - t_max: tempo final
    """
    Nt = int((t_max - t0)/dt) + 1
    r = np.zeros((Nt, *r0.shape))
    t = np.zeros(Nt)

    r[0] = r0
    t[0] = t0

    for i in range(Nt - 1):
        ti = t[i]
        ri = r[i]

        k1 = func(ti, ri)
        k2 = func(ti + dt/2, ri + dt/2*k1)
        k3 = func(ti + dt/2, ri + dt/2*k2)
        k4 = func(ti + dt,   ri + dt*k3)

        r[i+1] = ri + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = ti + dt

    return t, r