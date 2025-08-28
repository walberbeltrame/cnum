# 5 - Métodos de ponto fixo
Vamos aprender sobre como usar os métodos de ponto fixo para encontrar as raízes de funções reais.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana5
```

Faça o checkout nessa nova branch:

```bash
git checkout semana5
```

<br/>

## Atividade 1
Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

Instale a biblioteca SciPy:

```python
pip install scipy
```

Escreva os métodos de ponto fixo dentro do arquivo de algoritmos.

```python
import numpy as np
from scipy.optimize import approx_fprime

def pontofixo(a, g, TOL=1e-8):
    x = g(a)
    while abs(x - a) > TOL:
        a = x
        x = g(a)
    return x

def newton_raphson(a, f, TOL=1e-8, df=None):
    if df is None:
        def dfn(x):
            return approx_fprime(np.array([x]), lambda v: f(v[0]))[0]
    else:
        dfn = df
    g = lambda x: x - f(x) / dfn(x)
    return pontofixo(a, g, TOL)

def secante(a, b, f, TOL=1e-8):
    g = lambda a, b: (a * f(b) - b * f(a)) / (f(b) - f(a))
    x = g(a, b)
    while abs(x - b) > TOL:
        a, b = b, x
        x = g(a, b)
    return x
```

Resolver a equação $e^x = x + 2$ é equivalente a calcular os pontos fixos da função  

$$
g(x) = e^x - 2
$$

Use os métodos do ponto fixo  

$$
x^{(n+1)} = g(x^{(n)})
$$

com $x^{(0)} = -1.8$ para obter uma aproximação de uma das soluções da equação dada com 8 dígitos significativos.  

**Resposta:**  
$
x \approx -1.8414057
$

Gabarito
```python
import numpy as np

from algoritmos import (
    pontofixo,
    newton_raphson,
    secante,
)

f1 = lambda x: np.e**x - x - 2
g1 = lambda x: np.e**x - 2


def main():
    # Atividade 1
    print("-- Atividade 1 --")
    r = pontofixo(-1.8, g1)
    print(f"raiz ponto fixo = {r}")
    r = newton_raphson(-1.8, f1, df=lambda x: np.e**x - 1)
    print(f"raiz newton-raphson df = {r}")
    r = newton_raphson(-1.8, f1)
    print(f"raiz newton-raphson = {r}")
    r = secante(-1.8, -1.7, f1)
    print(f"raiz secante = {r}")


if __name__ == "__main__":
    main()
```

<br/>

## Atividade 2
Encontre a raiz positiva da função  

$$
f(x) = \cos(x) - x^2
$$  

pelos métodos do ponto fixo, inicializando-o com $x^{(0)} = 1$.  

Realize a iteração até obter estabilidade no **quinto dígito significativo**.  

**Resposta:**

$$
x \approx 0.82413 
$$

Processo iterativo:  

$$
x^{(n+1)} = x^{(n)} + \frac{\cos(x) - x^2}{\sin(x) + 2x}
$$

<br/>

## Atividade 3
Aplique os métodos do ponto fixo para resolver a equação:

$$
e^{-x^2} = 2x
$$

**Resposta:**

$$
x \approx 0.4193648
$$

<br/>

## Atividade 4
Resolva os três últimos exercícios da semana anterior usando os métodos do ponto fixo.