# Toy model de fluido turbulento 

## Motivação 

Esse projeto foi desenvolvido durante um projeto de IC supervisionado pelo professor Alexei A. Mailybaev.

A motivação é estudar um modelo simples hierárquico de vórtices com simulações numéricas. 

Além disso, o modelo, incialmente estático, foi estendido para um modelo temporal. 

## Objetivos

- Simular advecção de partículas no campo
- Estudar estruturas multiescala do modelo
- Desenvolver otimizações numéricas para o cálculo do campo

## Modelo 

Os centros obedecem a uma estrutura hierárquica:

$$
R_n = R_1 2^{1-n}
$$

e 

$$
T_n = T_1 \lambda ^{1 -n }
$$


onde $\lambda \in (1,2) $ e $R_n$ e $T_n$ são parametros que definem alcance e intensidade do campo, respectivamente. 
