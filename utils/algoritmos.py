import numpy as np
from scipy.optimize import approx_fprime


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


def newton_coef(x, y):
    n = len(x)
    a = np.array(y, dtype=float).copy()
    for j in range(1, n):
        a[j:n] = (a[j:n] - a[j - 1 : n - 1]) / (x[j:n] - x[0 : n - j])
    return a


def newton(x, y):
    a = newton_coef(x, y)
    p = np.array([a[-1]])
    for k in range(len(a) - 2, -1, -1):
        p = np.convolve(p, np.array([1.0, -x[k]]))
        if len(p) > 0:
            p[-1] += a[k]
    return p


def lagrange_add(p, q):
    if len(q) > len(p):
        p, q = q, p
    off = len(p) - len(q)
    res = p[:]
    for i in range(len(q)):
        res[off + i] += q[i]
    return res


def lagrange_mul(p, q):
    deg = (len(p) - 1) + (len(q) - 1)
    res = [0.0] * (deg + 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res


def lagrange_scale(p, s):
    return [s * c for c in p]


def lagrange(x, y):
    n = len(x)
    p = [0.0]
    for i in range(n):
        m = [1.0]
        d = 1.0
        xi = x[i]
        for j in range(n):
            if j == i:
                continue
            m = lagrange_mul(m, [1.0, -x[j]])
            d *= xi - x[j]
        pi = lagrange_scale(m, y[i] / d)
        p = lagrange_add(p, pi)
    return p


def polinomial(x, y):
    n = len(x)
    V = np.vander(x, N=n, increasing=False)
    a = np.linalg.solve(V, y.astype(float))
    return a

def regressao(x, y, v):
    V = v(x)
    Vt = V.T
    A = np.linalg.inv(Vt @ V) @ (Vt @ y)
    return A


def dp(f, x, h):
    return (f(x + h) - f(x)) / h


def dr(f, x, h):
    return (f(x) - f(x - h)) / h


def dc(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def medio(f, a, b):
    h = b - a
    return h * f((a + b) / 2)


def trapezio(f, a, b):
    h = b - a
    return h * (0.5 * f(a) + 0.5 * f(b))


def simpson(f, a, b):
    h = (b - a) / 2
    return h * ((1 / 3) * f(a) + (4 / 3) * f((a + b) / 2) + (1 / 3) * f(b))


def integral(metodo, f, a, b, n=1e-3):
    s = 0.0
    c = a
    d = a + n
    while d <= b:
        s += metodo(f, c, d)
        c = d
        d += n
    return s