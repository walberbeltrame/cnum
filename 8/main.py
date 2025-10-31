import numpy as np

from algoritmos import (
    bissecao,
    seidel,
    GN,
    fixed_point,
)


def f_temp(T):
    E = 500.125
    K = 272.975
    return 5.67e-8 * (T**4) + 0.4 * (T - K) - E


def main():
    # Atividade 1
    print("-- Atividade 1 --")
    Ta = 250.0
    Tb = 400.0
    T, i = bissecao(f_temp, Ta, Tb, 1e-10, iter=200)
    print(T)

    # Atividade 2
    print("-- Atividade 2 --")
    A = np.array([[17, -2, -3], [-5, 21, -2], [-5, -5, 22]], dtype=float)
    B = np.array([500, 200, 300], dtype=float)
    R1, R2, R3 = seidel(A, B, k=1000, TOL=1e-12)
    print(f"R1 = {R1}")
    print(f"R2 = {R2}")
    print(f"R3 = {R3}")

    # Atividade 3
    print("-- Atividade 3 --")
    A = np.array([[20, 10], [10, 20]], dtype=float)
    B = np.array([100, 100], dtype=float)
    I = seidel(A, B, k=1000, TOL=1e-12)
    I1, I2 = I
    I_R3 = (100 - I1 * 10) / 10
    print(f"I_R3 = {I_R3}")

    # Atividade 4
    print("-- Atividade 4 --")

    def E(x):
        x1, x2 = x
        return np.array(
            [
                x1**4 + 0.06823 * x1 - x2**4 - 0.05848 * x2 - 0.01753,
                x1**4 + 0.05848 * x1 - 2 * x2**4 - 0.11696 * x2 - 0.00254,
            ]
        )

    x = np.array([0.0, 0.0], dtype=float)
    r = fixed_point(x, lambda x: GN(x, E))
    print(r)


if __name__ == "__main__":
    main()
