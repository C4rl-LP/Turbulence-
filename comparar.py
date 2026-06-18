import time
import campos 
import os
import numpy as np

'''
# Comparar tempo dos campos 
xp=0.13
yp=0.41

nmax=4
t=0.0

Nrep=100

t0=time.perf_counter()

for _ in range(Nrep):
    campos.campo_total_podado(
        xp,yp,t,nmax
    )

t1=time.perf_counter()

tempo_podado=t1-t0

# -------- campo completo ----------
t0=time.perf_counter()

for _ in range(Nrep):
    campos.campo_total_correto(
        xp,yp,t,nmax
    )

t1=time.perf_counter()

tempo_full=t1-t0


# -------- campo podado ----------



print("Completo :",tempo_full)
print("Podado   :",tempo_podado)

print(
 "speedup =",
 tempo_full/tempo_podado
)

pasta = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta,'Teste_temporal.txt')

with open(arquivo, "a") as f:

    f.write("Teste \n")
    f.write(f"ponto: ({xp},{yp})    N = {nmax}   t = {t}     Repetições = {Nrep} \n")
    f.write("=-"*10)
    f.write("\n")
    f.write(f"Completo : {tempo_full}\n" )
    f.write(f"Podado : {tempo_podado} \n")
    f.write(f"speedup = {tempo_full/tempo_podado}\n")
    f.write("=-"*10)
    f.write("\n")


# ----------------------------
# Parâmetros do teste
# ----------------------------
'''
nmax = 4
t = 0.0

Nx = 50
Ny = 50

xvals = np.linspace(0.01, 0.99, Nx)
yvals = np.linspace(0.01, 0.99, Ny)

erros_abs = []
erros_rel = []

# ----------------------------
# Comparação
# ----------------------------

for x in xvals:
    for y in yvals:

        campo_full = campos.campo_total_correto(
            x, y, t, nmax
        )

        campo_podado = campos.campo_total_podado(
            x, y, t, nmax
        )

        erro_abs = abs(campo_full - campo_podado)

        if abs(campo_full) > 1e-14:
            erro_rel = erro_abs / abs(campo_full)
            erros_rel.append(erro_rel)

        erros_abs.append(erro_abs)

erros_abs = np.array(erros_abs)
erros_rel = np.array(erros_rel)

# ----------------------------
# Estatísticas
# ----------------------------

erro_abs_max = np.max(erros_abs)
erro_abs_med = np.mean(erros_abs)
erro_abs_rms = np.sqrt(np.mean(erros_abs**2))

erro_rel_max = np.max(erros_rel)
erro_rel_med = np.mean(erros_rel)

# ----------------------------
# Impressão
# ----------------------------

print("\nComparação Podado × Completo\n")

print(f"Número de pontos testados = {Nx*Ny}")
print(f"Erro absoluto máximo = {erro_abs_max:.3e}")
print(f"Erro absoluto médio  = {erro_abs_med:.3e}")
print(f"Erro RMS             = {erro_abs_rms:.3e}")

print(f"Erro relativo máximo = {erro_rel_max:.3e}")
print(f"Erro relativo médio  = {erro_rel_med:.3e}")

# ----------------------------
# Salvar em arquivo
# ----------------------------

pasta = os.path.dirname(os.path.abspath(__file__))
arquivo = os.path.join(pasta, "Teste_correcao.txt")

with open(arquivo, "a") as f:

    f.write("Comparacao Podado x Completo\n")
    f.write("-"*50 + "\n")

    f.write(f"nmax = {nmax}\n")
    f.write(f"t = {t}\n")
    f.write(f"grade = {Nx} x {Ny}\n\n")

    f.write(f"Erro absoluto maximo = {erro_abs_max:.6e}\n")
    f.write(f"Erro absoluto medio  = {erro_abs_med:.6e}\n")
    f.write(f"Erro RMS             = {erro_abs_rms:.6e}\n")

    f.write(f"Erro relativo maximo = {erro_rel_max:.6e}\n")
    f.write(f"Erro relativo medio  = {erro_rel_med:.6e}\n")

    f.write("-"*50 + "\n\n")