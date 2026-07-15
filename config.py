import numpy as np

# =====================================================================
# Parâmetros Físicos e Geométricos do Modelo
# =====================================================================

R_1 = 1.0            # Raio característico do nível 1
T_1 = 1.0            # Período característico do nível 1
O_1 = 2 * np.pi      # Frequência angular base (Omega_1)
x_1 = np.array([0.0, 0.0])  # Centro inicial (nível 1)
lamb = 2**(2/3)      # Parâmetro que determina a razão de decaimento do período (lambda)

# =====================================================================
# Parâmetros de Tweak (alpha, C/c)
# =====================================================================

# alpha: Determina a distância de corte/cutoff para cada nível.
# O raio de cutoff para o nível n é dado por: R_cutoff_n = alpha * R_n
# Note que nas simulações, o padrão é utilizar alpha = np.sqrt(2) ≈ 1.4...
alpha_padrao = np.sqrt(2) 

# c: Parâmetro C da função bump. Controla a suavidade e inclinação 
# do decaimento da velocidade local induzida por cada vórtice/centro.
c_padrao = 0.6
