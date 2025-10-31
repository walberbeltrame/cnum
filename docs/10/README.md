# 10 - Regressão
Vamos aprender sobre como usar técnicas de ajuste de curvas que buscam modelar a relação entre variáveis, como o método dos mínimos quadrados, para estimar os coeficientes da equação.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana10
```

Faça o checkout nessa nova branch:

```bash
git checkout semana10
```

<br/>

## Atividade 1
Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

Escreva o método de regressão por quadrado mínimo dentro do arquivo de algoritmos.

```python
import numpy as np

def regressao(x, y, v):
    V = v(x)
    Vt = V.T
    A = np.linalg.inv(Vt @ V) @ (Vt @ y)
    return A
```

Seja dado o conjunto de pontos  
$\{(-0.35, 0.2),\ (0.15, -0.5),\ (0.23, 0.54),\ (0.35, 0.7)\}$.

Encontre a função linear $f(x) = a_1 + a_2x$ que melhor se ajusta aos pontos dados no **sentido dos mínimos quadrados**.  
Em seguida, desenhe um gráfico com os pontos e o esboço da função ajustada.

**Resposta esperada:**

$$
f(x) = 0.19 - 0.47x
$$

Gabarito

```python
import numpy as np
import matplotlib.pyplot as plt

from algoritmos import regressao


def plot(x, y, v, num_img=1):
    x_vals = np.linspace(min(x) - 0.25, max(x) + 0.25, 400)
    y_vals = np.zeros_like(x_vals, dtype=float)
    A = regressao(x, y, v)
    for p, ap in enumerate(A):
        y_vals += ap * (x_vals**p)
    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, color="blue", label="Pontos dados")
    plt.plot(x_vals, y_vals, color="red", label="Ajuste linear f(x)")
    plt.title("Ajuste linear por mínimos quadrados")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"10/regressao_{num_img}.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    # Atividade 1
    print("-- Atividade 1 --")

    x = np.array([-0.35, 0.15, 0.23, 0.35], dtype=float)
    y = np.array([0.20, -0.50, 0.54, 0.70], dtype=float)
    v = lambda x: np.column_stack((np.ones(len(x)), x))

    A = regressao(x, y, v)

    print(A)
    plot(x, y, v, 1)

if __name__ == "__main__":
    main()
```

<br/>

## Atividade 2
Seja dado o conjunto de pontos  
$\{(-1.94,\,1.02),\ (-1.44,\,0.59),\ (0.93,\,-0.28),\ (1.39,\,-1.04)\}$.

Encontre a função linear $f(x) = a_1 + a_2x$ que melhor se ajusta aos pontos dados no sentido dos **mínimos quadrados**. Encontre o valor de $f(1)$. Em seguida, desenhe um gráfico com os pontos e o esboço da função ajustada.

**Resposta esperada:**

$$
f(x) = -0.53362743 -0.06891127x
$$

$$
f(1) = -0.6025387
$$

<br/>

## Atividade 3
Encontrar a parábola $y = ax^2 + bx + c$ que melhor aproxima o seguinte conjunto de dados:

| $i$ | 1 | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $x_i$ | 0.01 | 1.02 | 2.04 | 2.95 | 3.55 |
| $y_i$ | 1.99 | 4.55 | 7.20 | 9.51 | 10.82 |

**Resposta esperada:**

$$
y = -0.0407898x^2 + 2.6613293x + 1.9364598
$$

<br/>

## Atividade 4
Dado o seguinte conjunto de dados:

| $x_i$ | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $y_i$ | 31 | 35 | 37 | 33 | 28 | 20 | 16 | 15 | 18 | 23 | 31 |

**a)** Encontre a função do tipo  
$f(x) = a + b \sin(2\pi x) + c \cos(2\pi x)$  
que melhor aproxima os valores dados.

**b)** Encontre a função do tipo  
$f(x) = a + bx + cx^2 + dx^3$  
que melhor aproxima os valores dados.

**Respostas esperadas:**

$$
\text{a)}\quad a = 25.638625,\quad b = 9.8591874,\quad c = 4.9751219
$$

$$
\text{b)}\quad a = 31.475524,\quad b = 65.691531,\quad c = -272.84382,\quad d = 208.23621
$$