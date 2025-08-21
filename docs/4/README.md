# 4 - Método da bisseção
Vamos aprender sobre como usar o método da bisseção para encontrar as raízes de funções reais.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana4
```

Faça o checkout nessa nova branch:

```bash
git checkout semana4
```

<br/>

## Atividade 1
Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

Escreva o método da bisseção dentro do arquivo de algoritmos.

```python
def bissecao(f,     # função que queremos encontrar a raiz
              a,    # a início do intervalo
              b,    # b fim do intervalo
              TOL,   # erro tolerado
              iter=16):  # número máximo de iterações
    c = (a + b) / 2  # ponto médio entre os valores a e b
    if f(a) * f(b) > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")
    else:
        i = 0  # variável contador
        ERRO = abs(f(b) - f(a))  # diferença entre os valores de y

        while ERRO > TOL and i < iter:  # loop iterativo com parada
            c = (a + b) / 2.0
            if f(c) == 0:
                return c, i
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
            i += 1
            ERRO = abs(f(b) - f(a))
        return c, i
```

No arquivo principal, importe e teste a função:  

$$
f(x) = x^3 - x - 2
$$

Em que os valores iniciais a e b são obtidos graficamente.

Gabarito.

```python
import numpy as np
import matplotlib.pyplot as plt

from algoritmos import bissecao

# Definição da função
def f1(x):
    return x**3 - x - 2

def plot(f, number=1):
    # Intervalo para plotar
    x_vals = np.arange(-2, 3, 0.1)
    y_vals = f(x_vals)

    plt.axhline(0, color="black", linewidth=1)  # eixo x
    plt.axvline(0, color="black", linewidth=1)  # eixo y
    plt.plot(x_vals, y_vals)
    plt.grid(True)
    plt.title("Visualização da função f(x)")

    # Salvar gráfico como imagem
    plt.savefig(f"4/bissecao_{number}.png", dpi=120, bbox_inches="tight")
    plt.close()

def main():
    # Atividade 1
    print("-- Atividade 1 --")
    plot(f1, 1)
    r, i = bissecao(f1, 1, 2, 1e-15)
    print(f"raiz = {r} , i = {i}")

if __name__ == "__main__":
    main()
```

<br/>

## Atividade 2
Considere a equação  

$$
\sqrt{x} = \cos(x).
$$  

Use o método da bissecção com intervalo inicial obtido por gráfico para calcular a aproximação $x^{(4)}$ da solução desta equação.

**Resposta:**  
$
x^{(4)} = 0.6875
$ 

<br/>

## Atividade 3
Trace o gráfico e isole as três primeiras raízes positivas da função:  

$$
f(x) = 5\sin(x^2) - \exp\!\left(\frac{x}{10}\right)
$$  

em intervalos de comprimento $(0.1)$. Então, use o método da bissecção para obter aproximações dos zeros com precisão de $10^{-5}$.  

**Resposta:**  

- Intervalo $(0.4, 0.5)$, zero em $(x \approx 0.45931)$.  
- Intervalo $(1.7, 1.8)$, zero em $(x \approx 1.7036)$.  
- Intervalo $(2.5, 2.6)$, zero em $(x \approx 2.5582)$.  

<br/>

## Atividade 4
O desenho abaixo mostra um circuito não linear envolvendo uma fonte de tensão constante, um diodo retificador e um resistor.  

Sabendo que a relação entre a corrente ($I_d$) e a tensão ($v_d$) no diodo é dada pela seguinte expressão:

$ I_d = I_R \left( \exp\!\left(\tfrac{v_d}{v_t}\right) - 1 \right) $

onde $I_R$ é a corrente de condução reversa e $v_t$ a tensão térmica, dada por  

$ v_t = \tfrac{kT}{q} $

com $k$ a constante de Boltzmann, $T$ a temperatura de operação e $q$ a carga do elétron.  

Aqui, $I_R = 1\,pA = 10^{-12}\,A$, $T = 300\,K$.  

Escreva o problema como uma equação na incógnita $v_d$, usando o método da bissecção, e resolva-o com **3 algarismos significativos** para os seguintes casos:

- $V = 30 \, V$ e $R = 1 \, k\Omega$  
- $V = 3 \, V$ e $R = 1 \, k\Omega$  
- $V = 3 \, V$ e $R = 10 \, k\Omega$  
- $V = 300 \, mV$ e $R = 1 \, k\Omega$  
- $V = -300 \, mV$ e $R = 1 \, k\Omega$  
- $V = -30 \, V$ e $R = 1 \, k\Omega$  
- $V = -30 \, V$ e $R = 10 \, k\Omega$  

![Circuito](https://www.ufrgs.br/reamat/CalculoNumerico/livro-py/main4x.png)

**Dica:**  
A equação do circuito é:

$ V = R I_d + v_d $

**Resposta:**  
a) $0{,}623$  
b) $0{,}559$  
c) $0{,}500$  
d) $0{,}300$  
e) $-0{,}300$  
f) $-30$  
g) $-30$

<br/>

## Atividade 5
Calcule o comprimento do cabo ($C$) entre duas torres de transmissão (i.e., a catenária).  

A distância entre as torres é de $d = 500 \, m$.  

A flecha máxima permitida é $f_{max} = 50 \, m$.  

Flecha é a distância vertical entre uma reta que liga os dois pontos de fixação.  

A flecha ($f$) depende do comprimento do vão ($d$) e da tração ($C'$) aplicada ao cabo.  

O seu modelo matemático pode ser:

$ f = C \left[ \cosh\!\left(\tfrac{d}{2C}\right) - 1 \right] \quad (4.8) $

**Resposta:** $633,1621 \, m$

<br/>

## Atividade 6
Um retificador de meia onda a diodo alimenta uma carga indutiva-resistiva ($f = 1 \, kHz$, $L = 100 \, mH$ e $R = 1 \, k\Omega$).  

Encontre o ângulo $\beta$ para o qual a corrente $I_d$ no diodo se anula.  

Considere o seguinte modelo matemático:

$ I_d = \sin(\beta - \phi) + \sin(\phi)e^{-\tfrac{\beta}{\tan(\phi)}} \quad (4.9) $

com  

$ \tan(\phi) = \tfrac{2\pi f L}{R} $

**Resposta:** $\beta = 212,2284^\circ$
