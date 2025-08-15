# 3 - Erro com pontos flutuantes
Vamos aprender sobre como usar os erros de ponto flutuante para resolução de problemas numéricos.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana3
```

Faça o checkout nessa nova branch:

```bash
git checkout semana3
```

<br/>

## Atividade 1
A função exponencial natural pode ser definida pelo limite:
$$
e^x=\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n,
$$
mas também é dada pela **série de Maclaurin**:
$$
e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}
=1+\frac{x}{1!}+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots
$$

Implemente em **Python** o cálculo de $e^x$ pela série, interrompendo a soma quando o termo ficar menor que o limite prático de contribuição, usando a precisão de máquina como critério.

Gabarito

```python
import math
import sys

def exp_series(x, atol=0.0):
    """ Aproxima e^x pela série de Maclaurin com critério de parada numérico. """
    eps = sys.float_info.epsilon
    s = 1.0
    term = 1.0
    n = 0
    tol_abs = max(atol, eps)

    while True:
        n += 1
        term *= x / n
        s += term
        if abs(term) < eps * abs(s) or abs(term) < tol_abs:
            break
        if n > 10_000:
            break
    return s, n, term

# Demonstração
for val in [1.0, 5.0, -2.0]:
    approx, nterms, last = exp_series(val)
    print(f"x={val:+g} -> e^x ≈ {approx:.16g} (math.exp={math.exp(val):.16g}, termos={nterms})")
```

<br/>

## Atividade 2

Implemente:
$$
e^x\approx\left(1+\frac{x}{n}\right)^n
$$
com $n$ crescente, e:
1. Explique por que, para $x<0$ e $n$ muito grande, pode ocorrer **cancelamento catastrófico**;
2. Proponha um critério de parada numérico para encerrar o crescimento de $n$ sem perder precisão.

<br/>

## Atividade 3

Para $|x|$ grande, use:
$$
e^x = \left(e^{m\cdot 2^{-k}}\right)^{2^k}, \quad
k = \left\lceil \log_2\!\left(\frac{|x|}{\theta}\right)\right\rceil, \quad m = \frac{x}{2^k}
$$
Calcule $e^{m}$ pela série (Ex. 1) e depois eleve ao quadrado $k$ vezes.

Gabarito

```python
import math

def exp_series_scaling(x, theta=1.0):
    if x == 0.0:
        return 1.0, 0, 0.0, 0
    k = max(0, math.ceil(math.log2(abs(x)/theta))) if abs(x) > theta else 0
    m = x / (2**k)

    em, n_terms, _ = exp_series(m)
    y = em
    for _ in range(k):
        y *= y

    return y, k, n_terms

# Demonstração
for val in [10.0, -20.0]:
    y, k, n = exp_series_scaling(val, theta=1.0)
    print(f"x={val:+g} -> e^x ≈ {y:.6e} (math.exp={math.exp(val):.6e})  [k={k}, termos série(m)={n}]")
```

<br/>

## Atividade 4

Use:
$$
\cos x=\sum_{n=0}^{\infty}(-1)^n\frac{x^{2n}}{(2n)!}
$$
com a recursão:
$$
t_{n+1}=t_n\cdot\frac{-x^2}{(2n+1)(2n+2)}
$$
Defina um critério de parada baseado em `epsilon` e compare o erro relativo para $x\in[-20,20]$ (200 pontos) contra `math.cos(x)`.

<br/>

## Atividade 5

Dado $x$ e uma tolerância $\tau$, encontre o menor $N$ tal que:
$$
R_{N+1}(x)=\sum_{n=N+1}^{\infty}\frac{|x|^n}{n!} < \tau
$$

Gabarito

```python
def min_terms_for_tol(x, tol=1e-12):
    term = 1.0
    n = 0
    while term > tol:
        n += 1
        term *= abs(x) / n
        if n > 100000:
            break
    return n

# Demonstração
for x in [1, 3, 10]:
    n = min_terms_for_tol(x, 1e-12)
    print(f"x={x}: ~{n} termos para atingir tol=1e-12")
```

<br/>

## Atividade 6

Usando `decimal` ou `mpmath`, compute $e^x$ em alta precisão e compare com o resultado de `float64` (Ex. 1) para $x\in\{20, 40, 50\}$.
Analise:
- perda de dígitos significativos;
- quando o `float64` começa a saturar por overflow.

<br/>

Submeta a branch para o servidor:

```bash
git add .
git commit -m "Semana 3"
git push origin semana3
```