import numpy as np
import matplotlib.pyplot as plt

from algoritmos import bissecao

# Função da Atividade1
def f1(x):
    return x**3 - x - 2

# Função da Atividade2
def f2(x):
    return x**(1/2) - np.cos(x)

# Função da Atividade3
def f3(x):
    return (5*np.sin(x**2)) - (np.exp(x/10))

def plot(f, xi, xf, d=0.1, num_img=1):
    # Intervalo para plotar
    x_vals = np.arange(xi, xf, d)
    y_vals = f(x_vals)

    plt.axhline(0, color="black", linewidth=1)  # eixo x
    plt.axvline(0, color="black", linewidth=1)  # eixo y
    plt.plot(x_vals, y_vals)
    plt.grid(True)
    plt.title("Visualização da função f(x)")

    # Salvar gráfico como imagem
    plt.savefig(f"4/bissecao_{num_img}.png", dpi=120, bbox_inches="tight")
    plt.close()

def main():
    # Atividade 1
    print("-- Atividade 1 --")
    plot(f1, -2, -3, num_img=1)
    r, i = bissecao(f1, 1, 2, 1e-15)
    print(f"raiz = {r} , i = {i}")
    # Atividade 2
    print("-- Atividade 2 --")
    plot(f2, 0, 1, num_img=2)
    r, i = bissecao(f2, 0, 1, 1e-4, iter=4)
    print(f"raiz = {r} , i = {i}")
    # Atividade 3
    print("-- Atividade 3 --")
    plot(f3, 0, 3, num_img=3)
    r, i = bissecao(f3, 0.4, 0.5, 1e-5)
    print(f"raiz = {r:.5} , i = {i}")
    r, i = bissecao(f3, 1.7, 1.8, 1e-5)
    print(f"raiz = {r:.5} , i = {i}")
    r, i = bissecao(f3, 2.5, 2.6, 1e-5)
    print(f"raiz = {r:.5} , i = {i}")

if __name__ == "__main__":
    main()
