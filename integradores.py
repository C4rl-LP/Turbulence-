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

def simular_particulas_2_static(N, N_fields, dimensao_quadrado, r0, dt=0.01, t_max=1, func = None):
    """
    Simula N partículas em um pequeno quadrado
    ao redor da posição inicial r0.
    """

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = func(x, y, 0, N_fields)
        return np.column_stack((Vx, Vy))

    L = dimensao_quadrado

    # Distribuição inicial uniforme em um círculo de raio L/2
    R0 = L / 2

    theta = np.random.uniform(0, 2*np.pi, size=N-1)
    u = np.random.uniform(0, 1, size=N-1)

    r = R0 * np.sqrt(u)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    particle_t0 = r0 + np.column_stack((dx, dy))

    # adiciona a partícula central exatamente em r0
    particle_t0 = np.vstack((r0,particle_t0))

    t, r = solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r


