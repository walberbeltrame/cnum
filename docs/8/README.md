# 8 - Primeiro trabalho
Em dupla ou trio, faça de acordo com a especificação do trabalho, usando os conceitos de método da bisseção, métodos de ponto fixo, métodos de Gauss e resolução de sistemas não lineares.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana8
```

Faça o checkout nessa nova branch:

```bash
git checkout semana8
```

Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

<br/>

## Atividade 1
Para economizar energia elétrica, um agricultor implantou um sistema de painel solar na fazenda para alimentar uma bomba d’água, que faz a irrigação das plantações.  
A placa negra utilizada fica exposta ao sol e ao vento. Dessa forma, para gerar eletricidade, a irradiação solar $E$ (em $\text{W/m}^2$) sobre a placa tem que ser maior que a perda de calor por radiação e por condução, dada pela temperatura atmosférica $K$ (em Kelvin).

A equação é dada por:

$$
E = 5,67 \times 10^{-8} \, T^4 + 0,4(T - K)
$$

Determine a temperatura mínima da placa, dada a medição da irradiação e da temperatura do ar, sabendo que o valor médio diário é $E = 500,125$ e $K = 272,975$.

**Sugestão:** usar métodos da bissecção ou ponto fixo.  
**Resposta aproximada:**  

$$
T = 304{,}56801011987010952
$$

<br/>

## Aividade 2
Em um grande hospital, para evitar intermitência no sistema elétrico dos diversos equipamentos médicos, instalou-se três reatores de alta performance em série, com tensão nominal $R$ em função da potência máxima $P$ de operação dos enrolamentos de controle:

$$
17R_1 - 2R_2 - 3R_3 = P_1 = 500
$$  

$$
-5R_1 + 21R_2 - 2R_3 = P_2 = 200
$$  

$$
-5R_1 - 5R_2 + 22R_3 = P_3 = 300
$$  

Calcular as tensões nominais de cada reator.

**Sugestão:** usar o método de Gauss-Seidel.  
**Resposta aproximada:**  

$$
R_1 = 36{,}56081655798128
$$  

$$
R_2 = 20{,}768358378225123
$$  

$$
R_3 = 26{,}66572157641055
$$  

<br />

## Atividade 3
Determine o valor da corrente no resistor **R3**, sabendo que $V_1 = V_2 = 100\,V$ e $R_1 = R_2 = R_3 = 10\,\Omega$.

### Circuito equivalente

            R2           R1
     ┌─────/\/\/\───┬──/\/\/\───┐
     │             │            │
     │             │            │
    V2(+)        /\/\/\      V1(+)
     │            R3            │
     │             │            │
     └─────────────┴────────────┘

$$
\begin{bmatrix}
20 & 10 \\
10 & 20
\end{bmatrix}
\begin{Bmatrix}
I_1 \\
I_2
\end{Bmatrix}
=
\begin{Bmatrix}
100 \\
100
\end{Bmatrix}
$$

**Sugestão:** usar o método de Gauss-Seidel.

**Resposta aproximada:**  

$$
I_{R3} = 6{,}6667\,A
$$

<br />

## Atividade 4

Num sistema de energia solar, um vetor de equilíbrio energético $E$, na placa absorvente e na placa de vidro, segue o seguinte sistema de equações não lineares nas temperaturas absolutas $(K/m^2)$:

$$
(T_1^4 + 0,06823T_1) - (T_2^4 + 0,05848T_2) = E_1
$$    

$$
(T_1^4 + 0,05848T_1) - (2T_2^4 + 0,11696T_2) = E_2
$$  

Calcule o valor aproximado das temperaturas de equilíbrio, sabendo que os valores de equilíbrio energético foram medidos experimentalmente em:  

$E_1 = 0,01753$ e $E_2 = 0,00254$.

**Sugestão:** usar o método de Newton-Raphson.  
**Resposta aproximada:**  

$$
T_1 = 0,30543
$$  

$$ 
T_2 = 0,185261
$$  