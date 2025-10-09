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
