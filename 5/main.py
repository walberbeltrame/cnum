import numpy as np

from algoritmos import (
    pontofixo,
    newton_raphson,
    secante,
)

f1 = lambda x: np.e**x - x - 2
g1 = lambda x: np.e**x - 2


def main():
    # Atividade 1
    print("-- Atividade 1 --")
    r = pontofixo(-1.8, g1)
    print(f"raiz ponto fixo = {r}")
    r = newton_raphson(-1.8, f1, df=lambda x: np.e**x - 1)
    print(f"raiz newton-raphson df = {r}")
    r = newton_raphson(-1.8, f1)
    print(f"raiz newton-raphson = {r}")
    r = secante(-1.8, -1.7, f1)
    print(f"raiz secante = {r}")


if __name__ == "__main__":
    main()
