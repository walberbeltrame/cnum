import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from algoritmos import (
    newton,
    lagrange,
    polinomial,
)


def plot(x, y, num_img=1):
    x_vals = np.linspace(min(x) - 0.25, max(x) + 0.25, 400)
    p = newton(x, y)
    yn = np.polyval(p, x_vals)
    p = lagrange(x, y)
    yl = np.polyval(p, x_vals)
    p = polinomial(x, y)
    yp = np.polyval(p, x_vals)
    spline = CubicSpline(x, y, bc_type="natural")
    ys = spline(x_vals)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, yn, label="Newton")
    plt.plot(x_vals, yl, label="Lagrange", linestyle="--")
    plt.plot(x_vals, yp, label="Polinomial", linestyle=":")
    plt.plot(x_vals, ys, label="Spline cúbica", linewidth=2)
    plt.scatter(x, y, label="Pontos", zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolação: Newton, Lagrange, Polinomial e Spline Cúbica")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig(f"9/interpolacao_{num_img}.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    # Atividade 1
    print("-- Atividade 1 --")

    x = np.array(
        [-2.0, 0.0, 1.0, 2.0],
        dtype=float,
    )
    y = np.array(
        [-47.0, -3.0, 4.0, 41.0],
        dtype=float,
    )

    xr = np.array(
        [-1.0, 0.5, 1.5],
        dtype=float,
    )

    p = newton(x, y)
    yn = np.polyval(p, xr)

    p = lagrange(x, y)
    yl = np.polyval(p, xr)

    p = polinomial(x, y)
    yp = np.polyval(p, xr)

    spline = CubicSpline(x, y, bc_type="natural")
    ys = spline(xr)

    for xi, n, l, p, s in zip(xr, yn, yl, yp, ys):
        print(f"x = {xi:>4}:  N={n: .6f} | L={l: .6f} | P={p: .6f} | S={s: .6f}")

    plot(x, y, 1)


if __name__ == "__main__":
    main()

