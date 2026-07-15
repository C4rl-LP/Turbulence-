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


def obter_caminho_config(alpha=None, c=None, subpasta=None):
    """
    Retorna o caminho da pasta de resultados baseado na configuração física ativa.
    Estrutura: resultados/config_R1_{R_1}_T1_{T_1}_lamb_{lamb:.3f}_alpha_{alpha:.3f}_c_{c:.3f}/[subpasta]
    """
    import os
    a = alpha if alpha is not None else alpha_padrao
    cv = c if c is not None else c_padrao
    
    # Nome identificador da configuração
    nome_config = f"config_R1_{R_1:.2f}_T1_{T_1:.2f}_l_{lamb:.3f}_a_{a:.3f}_c_{cv:.3f}"
    
    # Obtém o caminho absoluto do diretório do projeto (onde config.py reside)
    diretorio_projeto = os.path.dirname(os.path.abspath(__file__))
    caminho = os.path.join(diretorio_projeto, "resultados", nome_config)
    
    if subpasta:
        caminho = os.path.join(caminho, subpasta)
    return caminho

