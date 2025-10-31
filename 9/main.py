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

    # Atividade 2
    print("-- Atividade 2 --")

    x = np.array([-1.0, 0.5, 1.0, 1.25], dtype=float)
    y = np.array([1.25, 0.5, 1.25, 1.8125], dtype=float)

    xr = np.array([-0.5, 0.0, 0.25], dtype=float)

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

    plot(x, y, 2)

    # Atividade 3a
    print("-- Atividade 3a --")

    x = np.array([-50.0, -5.0, 5.0, 75.0], dtype=float)
    y = np.array([-300.0, -50.0, 180.0, 350.0], dtype=float)

    xr = np.array([0.0], dtype=float)

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

    plot(x, y, 3)

    # Atividade 3b
    print("-- Atividade 3b --")

    x = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=float,
    )
    y = np.array(
        [80, -60, 40, -30, 20, -10, 5, -2.5, 1.25, -0.625, 0.3125],
        dtype=float,
    )

    xr = np.array([8.5], dtype=float)

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

    plot(x, y, 4)


if __name__ == "__main__":
    main()
