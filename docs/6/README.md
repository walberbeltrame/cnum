# 6 - Métodos de Gauss
Vamos aprender sobre como usar os métodos de Gauss para resolução de sistemas lineares quadráticos.

Crie uma nova branch (versão) do repositório:

```bash
git branch semana6
```

Faça o checkout nessa nova branch:

```bash
git checkout semana6
```

<br/>

## Atividade 1
Crie os arquivos da atividade de modo separado.

```txt
├─ /
│   └─ algoritmos.py
│   └─ main.py
```

Consideramos o seguinte sistema linear:

$$
\begin{cases}
x_1 + x_2 + x_3 = 1\\
4x_1 + 4x_2 + 2x_3 = 2\\
2x_1 + x_2 - x_3 = 0
\end{cases}
$$

Na sua forma matricial, este sistema é escrito como

$$
Ax = b \iff
\begin{bmatrix}
1 & 1 & 1 \\
4 & 4 & 2 \\
2 & 1 & -1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
=
\begin{bmatrix}
1 \\
2 \\
0
\end{bmatrix}
$$

Podemos resolver em Python usando a biblioteca NumPy (arquivo main):

```python
import numpy as np

def main():
    # Atividade 1
    print("-- Atividade 1 --")
    A = np.array([
        [1, 1,  1],
        [4, 4,  2],
        [2, 1, -1]
    ], dtype=float)
    B = np.array([1, 2, 0], dtype=float)
    print("Matriz A:")
    print(A)
    print("\nVetor B:")
    print(B)
    print("\nSolução NumPy x:")
    X = np.linalg.solve(A, B)
    print(X)

if __name__ == "__main__":
    main()
```

Agora, escreva os métodos de gauss dentro do arquivo de algoritmos e compare os resultados.

```python
import numpy as np


def lu_pivot(A: np.ndarray):
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.isclose(U[pivot, k], 0.0):
            raise np.linalg.LinAlgError("Matriz singular ou quase singular.")
        if pivot != k:
            U[[k, pivot], k:] = U[[pivot, k], k:]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    np.fill_diagonal(L, 1.0)
    return P, L, U


def lb(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    return Y


def uy(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    X = np.zeros(n)
    for i in reversed(range(n)):
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError(
                "U possui pivô nulo; sistema sem solução única."
            )
        X[i] = (Y[i] - np.dot(U[i, i + 1 :], X[i + 1 :])) / U[i, i]
    return X


def lu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    P, L, U = lu_pivot(A)
    Pb = P @ B
    Y = lb(L, Pb)
    X = uy(U, Y)
    return X


def jacobi(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)
    Xk = np.zeros(n)

    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("A possui elementos diagonais nulos; Jacobi pode falhar.")

    R = A - np.diagflat(D)

    for _ in range(k):
        Xk = (B - R @ X) / D
        if np.linalg.norm(Xk - X, ord=2) < TOL:
            return Xk
        X = Xk.copy()
    return X


def seidel(A: np.ndarray, B: np.ndarray, k: int, TOL: float) -> np.ndarray:
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)

    for _ in range(k):
        Xk = X.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], X[:i])
            s2 = np.dot(A[i, i + 1 :], Xk[i + 1 :])
            X[i] = (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(X - Xk, ord=2) < TOL:
            return X
    return X
```

Gabarito

```python
import numpy as np

from algoritmos import (
    lu,
    jacobi,
    seidel,
)


def main():
    # Atividade 1
    print("-- Atividade 1 --")
    A = np.array([[1, 1, 1], [4, 4, 2], [2, 1, -1]], dtype=float)
    B = np.array([1, 2, 0], dtype=float)
    print("Matriz A:")
    print(A)
    print("\nVetor B:")
    print(B)
    print("\nSolução NumPy x:")
    X = np.linalg.solve(A, B)
    print(X)
    print("\nSolução LU x:")
    X = lu(A, B)
    print(X)
    A = np.array([[2, 1, -1], [4, 4, 2], [1, 1, 1]], dtype=float)
    B = np.array([0, 2, 1], dtype=float)
    print("\nMatriz A:")
    print(A)
    print("\nVetor B:")
    print(B)
    print("\nSolução Jacobi x:")
    X = jacobi(A, B, 100, 1e-8)
    print(X)
    print("\nSolução Seidel x:")
    X = seidel(A, B, 100, 1e-8)
    print(X)


if __name__ == "__main__":
    main()
```

<br/>

## Atividade 2
Resolva o seguinte sistema pelo método de Jacobi e Gauss–Seidel:

$$
\begin{cases}
5x_1 + x_2 + x_3 = 50 \\
-x_1 + 3x_2 - x_3 = 10 \\
x_1 + 2x_2 + 10x_3 = -30
\end{cases}
$$

Use como critério de paragem tolerância inferior a $10^{-3}$  
e inicialize com $x^0 = y^0 = z^0 = 0$.

<br/>

## Atividade 3
Faça uma permutação de linhas no sistema abaixo e resolva pelos métodos de Jacobi e Gauss–Seidel:

$$
\begin{cases}
x_{1} + 10x_{2} + 3x_{3} = 27 \\
4x_{1} + x_{3} = 6 \\
2x_{1} + x_{2} + 4x_{3} = 12
\end{cases}
$$


<br/>

## Atividade 4
O circuito linear da Figura pode ser modelado pelo sistema dado a seguir.  
Escreva esse sistema na forma matricial sendo as tensões $V_1, V_2, V_3, V_4, V_5$ as cinco incógnitas.  
Resolva esse problema quando $V = 127$ e:

- $R_1 = R_2 = R_3 = R_4 = 2$, $R_5 = R_6 = R_7 = 100$ e $R_8 = 50$
- $R_1 = R_2 = R_3 = R_4 = 2$, $R_5 = 50$, $R_6 = R_7 = R_8 = 100$

Sistema de equações:

$$
V_1 = V
$$

$$
\frac{V_1 - V_2}{R_1} + \frac{V_5 - V_2}{R_2} - \frac{V_2}{R_5} = 0
$$

$$
\frac{V_2 - V_3}{R_2} + \frac{V_4 - V_3}{R_3} - \frac{V_3}{R_6} = 0
$$

$$
\frac{V_3 - V_4}{R_3} + \frac{V_5 - V_4}{R_4} - \frac{V_4}{R_7} = 0
$$

$$
\frac{V_4 - V_5}{R_4} - \frac{V_5}{R_8} = 0
$$

![Circuito](https://www.ufrgs.br/reamat/CalculoNumerico/livro-py/main11x.png)

Complete a tabela abaixo representando a solução com **4 algarismos significativos**:

| Caso | $V_1$ | $V_2$ | $V_3$ | $V_4$ | $V_5$ |
|------|-------|-------|-------|-------|-------|
| a    |       |       |       |       |       |
| b    |       |       |       |       |       |

Então, refaça este problema reduzindo o sistema para apenas 4 incógnitas ($V_2, V_3, V_4, V_5$).
