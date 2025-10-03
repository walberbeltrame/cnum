import numpy as np

from algoritmos import (
    G,
    GN,
    fixed_point,
)


def main():
    # Atividade 1
    print("-- Atividade 1 --")

    def F(x):
        x1, x2, x3 = x
        return np.array(
            [
                2.0 * x1 - x2 - np.cos(x1),
                -x1 + 2.0 * x2 - x3 - np.cos(x2),
                -x2 + x3 - np.cos(x3),
            ],
            dtype=float,
        )

    def J(x):
        x1, x2, x3 = x
        return np.array(
            [
                [2.0 + np.sin(x1), -1.0, 0.0],
                [-1.0, 2.0 + np.sin(x2), -1.0],
                [0.0, -1.0, 1.0 + np.sin(x3)],
            ],
            dtype=float,
        )

    x = np.array([1.0, 1.0, 1.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)

    # Atividade 2
    print("-- Atividade 2 --")

    def F(x):
        x1, x2, x3 = x
        return np.array([
            6.0*x1 - 2.0*x2 + np.exp(x3) - 2.0,
            np.sin(x1) - x2 + x3,
            np.sin(x1) + 2.0*x2 + 3.0*x3 - 1.0
        ], dtype=float)

    def J(x):
        x1, x2, x3 = x
        return np.array([
            [6.0,            -2.0,             np.exp(x3)],
            [-np.cos(x1),    -1.0,             1.0],
            [-np.cos(x1),     2.0,             3.0]
        ], dtype=float)

    x = np.array([0.0, 0.0, 0.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)

    # Atividade 3
    print("-- Atividade 3 --")

    def F(x):
        x1, x2 = x
        return np.array([
            (1.0/8.0)*(x1**2) + (1.0/5.0)*((x2 - 1.0)**2) - 1.0,
            np.arctan(x1) + x1 - x2 - (x2**3)
        ], dtype=float)

    def J(x):
        x1, x2 = x
        return np.array([
            [x1/4.0,           (1.0/5.0)*(2.0*x2 - 2.0)],
            [((x1**2) + 2.0)/((x1**2) + 1.0),   -(3.0*(x2**2)) - 1.0]
        ], dtype=float)
    
    x = np.array([-1.0, -1.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)

    x = np.array([3.0, 1.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)

    # Atividade 4
    print("-- Atividade 4 --")

    def F(x):
        x1, x2, x3, k = x
        return np.array([
            x1 + x2 + x3 - 1500.0,
            (0.3 + 2e-4*x1 + 4*3.4e-9*(x1**3)) - k,
            (0.25 + 4e-4*x2 + 3*4.3e-7*(x2**2)) - k,
            (0.19 + 2*5e-4*x3 + 4*1.1e-7*(x3**3)) - k
        ], dtype=float)

    def J(x):
        x1, x2, x3, k = x
        return np.array([
            [1.0, 1.0, 1.0, 0.0],
            [2e-4 + 3*4*3.4e-9*(x1**2), 0.0, 0.0, -1.0],
            [0.0, 4e-4 + 2*3*4.3e-7*(x2), 0.0, -1.0],
            [0.0, 0.0, 2*5e-4 + 3*4*1.1e-7*(x3**2), -1.0]
        ], dtype=float)
    
    x = np.array([500.0, 500.0, 500.0, 1.0], dtype=float)
    r = fixed_point(x, lambda x: G(x, F, J))
    print(r)
    r = fixed_point(x, lambda x: GN(x, F))
    print(r)

if __name__ == "__main__":
    main()
