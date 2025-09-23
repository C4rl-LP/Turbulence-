from matplotlib import pyplot as plt
import numpy as np

class Vector_plot_quarter_division():
    def __init__(self, N_fields, R1, T1, lamb):
        self.N_fields = N_fields
        self.R1 = R1
        self.T1 = T1
        self.lamb = lamb
        

    def definir_coordenadas(self):
        pass
    def phis_i(self, x,y,i, x0, y0):
        Ri = self.R1 * 2**(-i + 1)
        Ti = self.T1 * self.lamb**(-i + 1)
        r_norm = np.sqrt((x - x0)**2 + (y - y0)**2)
        return (Ri**2/Ti**i) * np.exp(-r_norm**2/(2*Ri**2) + 1/2)
    
    def primeirocampo_teste(self, x,y):
        phis = self.phis_i(x, y,1, 0, 0)
        Vx = (-y/self.R1**2) * phis
        Vy = (x/self.R1**2) *phis
        return (Vx, Vy)

campo = Vector_plot_quarter_division(N_fields=1, R1=1.0, T1=1.0, lamb=1.0)

# Criar grid de pontos
n = 20  # resolução
x = np.linspace(-5*campo.R1, 5*campo.R1, n)
y = np.linspace(-5*campo.R1, 5*campo.R1, n)
X, Y = np.meshgrid(x, y)

# Calcular o campo vetorial
Vx, Vy = campo.primeirocampo_teste(X, Y)
print(X, Y)
# Plotar usando quiver
plt.figure(figsize=(6,6))
plt.quiver(X, Y, Vx, Vy, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Campo vetorial (quiver)")
plt.axis("equal")
plt.show()

plt.figure(figsize=(6,6))
plt.streamplot(X, Y, Vx, Vy, color=np.hypot(Vx, Vy), cmap="viridis", density=1.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Campo vetorial (streamplot)")
plt.axis("equal")
plt.show()