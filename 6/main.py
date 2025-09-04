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
