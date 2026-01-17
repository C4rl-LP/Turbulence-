import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import product


R_1 = 1
O_1 = 1
x_1 = np.array([0,0])
lamb = 2**(2/3)

def solve_RK4(func, r0, t0, dt, t_max):
    r = [np.array(r0)]
    t = [t0]
    while t[-1] < t_max:
        ti = t[-1]
        ri = r[-1]

        k1 = func(ti, ri)
        k2 = func(ti + dt/2, ri + dt/2 * k1)
        k3 = func(ti + dt/2, ri + dt/2 * k2)
        k4 = func(ti + dt,   ri + dt * k3)

        r_next = ri + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        r.append(r_next)
        t.append(ti + dt)

    return np.array(t), np.array(r)

def indices_nivel(n):
    if n == 1:
        return [[]]
    return list(product(range(4), repeat=n-1))

def R(n):
    return R_1*2**(1-n)
def Omega(n):
    return O_1*lamb**((n-1))
def T(n):
    return 2*np.pi/Omega(n)


def phi(n,index):
    return np.pi*(1/4 + (index[-1]-1)/2)

def x_centros(n, index, t):
    if n == 1:
        return x_1

    if len(index) != n - 1:
        raise ValueError("Erro de tamanho de array")

    diff = R(n)*np.sqrt(2)*np.array([
        np.cos(Omega(n)*t + phi(n, index)),
        np.sin(Omega(n)*t + phi(n, index))
    ])

    return diff + x_centros(n-1, index[:-1], t)



def campo(x,y,t, n, index):
    x_centro_n = x_centros(n,index,t)
    
    dy = y - x_centro_n[1]
    dx = x - x_centro_n[0]
    factor = -(dx**2 + dy**2)*2/ R(n)
    poten = 2*np.pi*np.exp(factor - 1) / T(n)
    vx = -dy*poten
    vy = dx*poten
    return vx, vy

def campo_total(x, y, t, n_max=3):
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)

    for n in range(1, n_max + 1):
        for idx in indices_nivel(n):
            vx, vy = campo(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy
np.random.seed(1200)

def simular_particulas(N, N_fields,dimensao_quadrado, r0, dt=0.01, t_max=1):

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t, N_fields)
        return np.column_stack((Vx, Vy))

    L = dimensao_quadrado
    particle_t0 = r0 + np.random.uniform(-L/2,L/2,size=(N, 2))
    print(particle_t0)
    t, r = solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++ simular para teste +++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def simular_particula_campo_congelado(
    r0, 
    N_fields, 
    t_freeze=0.0, 
    dt=0.01, 
    t_max=10
):
    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t_freeze, N_fields)
        return np.column_stack((Vx, Vy))

    t, r = solve_RK4(funcao, r0, 0.0, dt, t_max)
    return t, r

def simular_particula_campo_movel(
    r0, 
    N_fields, 
    dt=0.01, 
    t_max=10
):
    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        Vx, Vy = campo_total(x, y, t, N_fields)
        return np.column_stack((Vx, Vy))

    t, r = solve_RK4(funcao, r0, 0.0, dt, t_max)
    return t, r
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
r0 = np.array([[np.pi/6, np.sqrt(2)/4]]) 
N_fields = 3


t1, r_movel = simular_particula_campo_movel(
    r0,
    N_fields=N_fields,
    dt=0.01,
    t_max=10
)

t2, r_congelado = simular_particula_campo_congelado(
    r0,
    N_fields=N_fields,
    t_freeze=0.0,
    dt=0.01,
    t_max=10
)

ts, rs = simular_particulas(N = 3, N_fields=N_fields,dimensao_quadrado = 0.05,r0 = r0,
    dt=0.01,
    t_max=20
)


x_m, y_m = r_movel[:, 0, 0], r_movel[:, 0, 1]
x_c, y_c = r_congelado[:, 0, 0], r_congelado[:, 0, 1]
x_s, y_s = rs[:, :,0], rs[:, :,1]


plt.figure(figsize=(6,6))

plt.plot(x_m, y_m, label='Campo dependente do tempo')
plt.plot(x_c, y_c, '--', label='Campo congelado (t = 0)')
plt.plot(x_s, y_s)

plt.scatter(x_m[0], y_m[0], color='green', s=50, label='Início')
plt.scatter(x_m[-1], y_m[-1], color='red', s=50, label='Fim')
plt.scatter(x_c[-1], y_c[-1], color='purple', s=50, label='Fim do congelado')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparação: campo móvel vs campo congelado')
plt.axis('equal')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.show()
