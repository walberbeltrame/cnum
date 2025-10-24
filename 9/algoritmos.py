import numpy as np


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

