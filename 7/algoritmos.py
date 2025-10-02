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
