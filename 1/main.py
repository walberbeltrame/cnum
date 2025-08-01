# a.x² + b.x + c
"""
Raiz de equação quadrática
"""

from math import sqrt

def zeros(a, b, c):
    delta = b**2 - (4*a*c)

    if delta < 0:
        return None
    elif delta == 0:
        x = -b / (2*a)
        return [x, x]
    else:
        x1 = (-b + sqrt(delta)) / (2*a)
        x2 = (-b - delta**0.5) / (2*a)
        return [x1, x2]

def main():
    x = zeros(1, -5, 6)
    print(x)

if __name__ == "__main__":
    main()


