# 7 - Sistemas não lineares
Vamos aprender sobre como usar o método iterativo de Newton-Raphson adaptado aos sistemas de equações, envolvendo aproximações iniciais, cálculo da matriz do Jacobiano e critério de parada, para resolução de sistemas não lineares.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana7
```

Faça o checkout nessa nova branch:

```bash
git checkout semana7
```

<br/>

## Atividade 1
Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

Escreva o método iterativo de Newton-Raphson adaptado dentro do arquivo de algoritmos.

```python
import numpy as np
from scipy.optimize import approx_fprime


def JN(x, F, eps=1e-8):
    x = np.asarray(x, dtype=float)
    n = x.size
    Jnum = np.zeros((n, n), dtype=float)

    def fi(v):
        return F(v)[i]

    for i in range(n):
        Jnum[i, :] = approx_fprime(x, fi, epsilon=eps)

    return Jnum


def G(x, F, J):
    return x - np.linalg.inv(J(x)) @ F(x)


def GN(x, F):
    return x - np.linalg.inv(JN(x, F)) @ F(x)


def fixed_point(a, g, TOL=1e-8, iter=1000):
    x = g(a)
    i = 1
    while np.linalg.norm(x - a) > TOL and i < iter:
        a = x
        x = g(a)
        i += 1
    return x
```

Encontre uma aproximação numérica para o seguinte problema não linear de três equações e três incógnitas:

$$
2x_{1} - x_{2} = \cos(x_{1})
$$

$$
- x_{1} + 2x_{2} - x_{3} = \cos(x_{2})
$$

$$
- x_{2} + x_{3} = \cos(x_{3})
$$

Partindo da seguinte aproximação inicial:

$$
x^{(0)} = (1,\,1,\,1)^{T}
$$

Gabarito

```python
import numpy as np

from algoritmos import (
    G,
    GN,
    fixed_point,
)


def main():
    # Atividade 1
    print("-- Atividade 1 --")

    def F(x):
        x1, x2, x3 = x
        return np.array(
            [
                2.0 * x1 - x2 - np.cos(x1),
                -x1 + 2.0 * x2 - x3 - np.cos(x2),
                -x2 + x3 - np.cos(x3),
            ],
            dtype=float,
        )

    def J(x):
        x1, x2, x3 = x
        return np.array(
            [
                [2.0 + np.sin(x1), -1.0, 0.0],
                [-1.0, 2.0 + np.sin(x2), -1.0],
                [0.0, -1.0, 1.0 + np.sin(x3)],
            ],
            dtype=float,
        )

    x = np.array([1.0, 1.0, 1.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)


if __name__ == "__main__":
    main()
```

<br/>

## Atividade 2
Encontre uma aproximação em cada incógnita para a solução próxima da origem do sistema:

$$
6x - 2y + e^{z} = 2
$$

$$
\sin(x) - y + z = 0
$$

$$
\sin(x) + 2y + 3z = 1
$$

**Resposta esperada:**

$$
x \approx 0.259751, \quad y \approx 0.302736, \quad z \approx 0.045896
$$

<br/>

## Atividade 3
Considere o problema de encontrar os pontos de interseção das curvas descritas:

$$
\frac{x^{2}}{8} + \frac{(y-1)^{2}}{5} = 1
$$

$$
\tan^{-1}(x) + x = y + y^{3}
$$

Com base no gráfico, encontre soluções aproximadas para o problema e use-as para iniciar o método de Newton-Raphson para encontrar as raízes.

**Resposta esperada:**

$$
(-1.2085435,\,-1.0216674) 
\quad \text{e} \quad 
(2.7871115,\,1.3807962)
$$

<br/>

## Atividade 4
Uma indústria consome energia elétrica de três usinas fornecedoras.  

O custo de fornecimento em reais por hora como função da potência consumida em kW é dada pelas seguintes funções:

$$
C_{1}(x) = 10 + 0.3x + 10^{-4}x^{2} + 3.4 \cdot 10^{-9}x^{4} \tag{5.61}
$$

$$
C_{2}(x) = 50 + 0.25x + 2 \cdot 10^{-4}x^{2} + 4.3 \cdot 10^{-7}x^{3}
$$

$$
C_{3}(x) = 500 + 0.19x + 5 \cdot 10^{-4}x^{2} + 1.1 \cdot 10^{-7}x^{4}
$$

Calcule a distribuição de consumo que produz custo mínimo quando a potência total consumida é **1500 kW**.  

Denote por $x_{1}, x_{2}, x_{3}$ as potências consumidas das usinas 1, 2 e 3, respectivamente.  

O custo total será dado por:

$$
C(x_{1},x_{2},x_{3}) = C_{1}(x_{1}) + C_{2}(x_{2}) + C_{3}(x_{3})
$$

Enquanto o consumo total é:

$$
G(x_{1},x_{2},x_{3}) = x_{1} + x_{2} + x_{3} - 1500 = 0
$$

Pelos multiplicadores de Lagrange, temos que resolver o sistema:

$$
\nabla C(x_{1},x_{2},x_{3}) = \lambda \nabla G(x_{1},x_{2},x_{3})
$$

$$
G(x_{1},x_{2},x_{3}) = 0
$$

**Resposta esperada:**

$$
(x_{1},x_{2},x_{3}) \approx (453.62, \; 901.94, \; 144.43)
$$
