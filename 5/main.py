import numpy as np

from algoritmos import (
    pontofixo,
    newton_raphson,
    secante,
)

# Função da Atividade1
f1 = lambda x: np.e**x - x - 2
g1 = lambda x: np.e**x - 2
df1 = lambda x: np.e**x - 1

# Função da Atividade2
f2 = lambda x: np.cos(x) - x**2
g2 = lambda x: np.sqrt(np.cos(x))
df2 = lambda x: -np.sin(x) - 2*x

# Função da Atividade3
f3 = lambda x: np.e**(-x**2) - 2*x
g3 = lambda x: np.e**(-x**2)/2
df3 = lambda x: (-2)*np.e**(-x**2) - 2

# Função da Atividade4
def f4(x, V, R):
    IR = 1e-12  # corrente de saturação (A)
    T = 300.0  # temperatura (K)
    k = 1.38064852e-23  # constante de Boltzmann (J/K)
    q = 1.60217662e-19  # carga do elétron (C)
    vt = k * T / q  # tensão térmica (V)
    return R * IR * (np.exp(x / vt) - 1) + x - V

# Função da Atividade5
def f5(x):
    d = 500
    fmax = 50
    return (x*(np.cosh(d/(2*x))-1)) - fmax

# Função da Atividade6
def f6(x):
    F = 1e3
    L = 100e-3
    R = 1e3
    T = 2 * np.pi * F * L / R
    A = np.atan(T)
    return np.sin(x - A) + np.sin(A) * np.exp(-x / T)

def main():
    # Atividade 1
    print("-- Atividade 1 --")
    r = pontofixo(-1.8, g1)
    print(f"raiz ponto fixo = {r}")
    r = newton_raphson(-1.8, f1, df=df1)
    print(f"raiz newton-raphson df = {r}")
    r = newton_raphson(-1.8, f1)
    print(f"raiz newton-raphson = {r}")
    r = secante(-1.8, -1.7, f1)
    print(f"raiz secante = {r}")
    # Atividade 2
    print("-- Atividade 2 --")
    r = pontofixo(1.0, g2, TOL=1e-5)
    print(f"raiz ponto fixo = {r:.5}")
    r = newton_raphson(1.0, f2, TOL=1e-5, df=df2)
    print(f"raiz newton-raphson df = {r:.5}")
    r = newton_raphson(1.0, f2, TOL=1e-5)
    print(f"raiz newton-raphson = {r:.5}")
    r = secante(1.0, 1.1, f2, TOL=1e-5)
    print(f"raiz secante = {r:.5}")
    # Atividade 3
    print("-- Atividade 3 --")
    r = pontofixo(1.0, g3)
    print(f"raiz ponto fixo = {r}")
    r = newton_raphson(1.0, f3, df=df3)
    print(f"raiz newton-raphson df = {r}")
    r = newton_raphson(1.0, f3)
    print(f"raiz newton-raphson = {r}")
    r = secante(1.0, 1.1, f3)
    print(f"raiz secante = {r}")
    # Atividade 4
    print("-- Atividade 4 --")
    VRs = [
        (30, 1e3, 0, 1),
        (3, 1e3, 0, 1),
        (3, 1e4, 0, 1),
        (0.3, 1e3, 0, 0.5),
        (-0.3, 1e3, -1, 0),
        (-30, 1e3, -40, 0),
        (-30, 1e4, -40, 0),
    ]
    for V, R, a, b in VRs:
        try:
            f4_vrs = lambda x: f4(x, V, R)
            r = newton_raphson(b, f4_vrs)
            print(f"V={V} V, R={R/1e3:.0f}kΩ --> vd = {r:.3f} V")
        except ValueError as error:
            print(f"V={V} V, R={R/1e3:.0f}kΩ --> {error}")
    # Atividade 5
    print("-- Atividade 5 --")
    r = newton_raphson(550, f5)
    print(f"raiz = {r:.4}")
    # Atividade 6
    print("-- Atividade 6 --")
    a = 212 * np.pi / 180
    r = newton_raphson(a, f6)
    r_deg = r * 180 / np.pi
    print(f"raiz = {r_deg:.4}")

if __name__ == "__main__":
    main()
