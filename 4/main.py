import numpy as np
import matplotlib.pyplot as plt

from algoritmos import bissecao

# Definição da função
def f1(x):
    return x**3 - x - 2

def plot(f, number=1):
    # Intervalo para plotar
    x_vals = np.arange(-2, 3, 0.1)
    y_vals = f(x_vals)

    plt.axhline(0, color="black", linewidth=1)  # eixo x
    plt.axvline(0, color="black", linewidth=1)  # eixo y
    plt.plot(x_vals, y_vals)
    plt.grid(True)
    plt.title("Visualização da função f(x)")

    # Salvar gráfico como imagem
    plt.savefig(f"4/bissecao_{number}.png", dpi=120, bbox_inches="tight")
    plt.close()

def main():
    # Atividade 1
    print("-- Atividade 1 --")
    plot(f1, 1)
    r, i = bissecao(f1, 1, 2, 1e-15)
    print(f"raiz = {r} , i = {i}")

if __name__ == "__main__":
    main()
