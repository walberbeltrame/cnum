import math
import sys
from decimal import Decimal, getcontext


def exp_series(x, atol=0.0):
    """Aproxima e^x pela série de Maclaurin com critério de parada numérico."""
    eps = sys.float_info.epsilon
    s = 1.0
    term = 1.0
    n = 0
    tol_abs = max(atol, eps)

    while True:
        n += 1
        term *= x / n
        s += term
        if abs(term) < eps * abs(s) or abs(term) < tol_abs:
            break
        if n > 10_000:
            break
    return s, n, term


def exp_limit(x, n0=10, n_max=10_000_000, growth=2.0, rtol=1e-14, atol=0.0):
    """
    Aproxima e^x por E_n(x) = (1 + x/n)^n, crescendo n até convergir.
    Usa log1p para estabilidade: E_n = exp(n * log1p(x/n)).
    """
    n = max(1, n0)
    prev = None
    while n <= n_max:
        y = math.exp(n * math.log1p(x / n))
        if prev is not None:
            if abs(y - prev) <= max(atol, rtol * abs(y)):
                return y, n
        prev = y
        n = int(max(n + 1, n * growth))
    return prev, n_max  # melhor aprox. dentro do limite


def exp_series_scaling(x, theta=1.0):
    if x == 0.0:
        return 1.0, 0, 0.0, 0
    k = max(0, math.ceil(math.log2(abs(x) / theta))) if abs(x) > theta else 0
    m = x / (2**k)

    em, n_terms, _ = exp_series(m)
    y = em
    for _ in range(k):
        y *= y

    return y, k, n_terms


def cos_series(x, rtol=1e-15, atol=0.0):
    """
    cos(x) pela série de Maclaurin com atualização recursiva do termo.
    Para quando |termo| < max(atol, rtol*|soma|, eps).
    """
    eps = sys.float_info.epsilon
    s = 1.0  # n=0
    term = 1.0
    n = 0
    while True:
        # atualiza para o próximo termo (2n -> 2(n+1))
        term *= -(x * x) / ((2 * n + 1) * (2 * n + 2))
        s_new = s + term
        if abs(term) < max(atol, rtol * abs(s_new), eps):
            s = s_new
            break
        s = s_new
        n += 1
        if n > 10_000:  # segurança
            break
    return s, n


def min_terms_for_tol(x, tol=1e-12):
    term = 1.0
    n = 0
    while term > tol:
        n += 1
        term *= abs(x) / n
        if n > 100_000:
            break
    return n


def exp_decimal(x, prec=80):
    """
    e^x em alta precisão usando decimal:
    - converte x para Decimal
    - usa série de Maclaurin com parada por precisão do contexto
    """
    getcontext().prec = prec
    xd = Decimal(str(x))
    s = Decimal(1)
    term = Decimal(1)
    n = 0
    # tolerância ~ 10^{-(prec-2)}
    tol = Decimal(1) / (Decimal(10) ** (prec - 2))
    while True:
        n += 1
        term *= xd / n
        s_new = s + term
        if abs(term) < tol:
            s = s_new
            break
        s = s_new
        if n > 20_000:
            break
    return +s  # aplica arredondamento do contexto


def compare_float_vs_highprecision(xs=(20, 40, 50), prec=80):
    rows = []
    for x in xs:
        hp = exp_decimal(x, prec=prec)
        fp = (
            math.exp(x) if x < 709.78 else float("inf")
        )  # limiar de overflow do float64
        # erro relativo quando possível
        if math.isfinite(fp):
            rel = abs((Decimal(str(fp)) - hp) / hp)
            rel_str = f"{rel:.2E}"
        else:
            rel_str = "overflow(float64)"
        rows.append((x, str(hp), fp, rel_str))
    return rows


def main():
    # Atividade 1
    print("-- Atividade 1 --")
    for val in [1.0, 5.0, -2.0]:
        approx, nterms, last = exp_series(val)
        print(
            f"x={val:+g} -> e^x ≈ {approx:.16g} (math.exp={math.exp(val):.16g}, termos={nterms})"
        )
    # Atividade 2
    print("-- Atividade 2 --")
    for val in [1.0, 5.0, -2.0]:
        y, n = exp_limit(val)
        print(val, y, math.exp(val), n)
    # Atividade 3
    print("-- Atividade 3 --")
    for val in [10.0, -20.0]:
        y, k, n = exp_series_scaling(val, theta=1.0)
        print(
            f"x={val:+g} -> e^x ≈ {y:.6e} (math.exp={math.exp(val):.6e})  [k={k}, termos série(m)={n}]"
        )
    # Atividade 4
    print("-- Atividade 4 --")
    xs = [-20 + 40 * i / 199 for i in range(200)]
    errs = [abs(cos_series(x)[0] - math.cos(x)) for x in xs]
    print(f"xs={xs[:10]}\nerrs={errs[:10]}...")
    # Atividade 5
    print("-- Atividade 5 --")
    for x in [1, 3, 10]:
        n = min_terms_for_tol(x, 1e-12)
        print(f"x={x}: ~{n} termos para atingir tol=1e-12")
    # Atividade 6
    print("-- Atividade 6 --")
    for r in compare_float_vs_highprecision((20, 40, 50), prec=80):
        print(f"x={r[0]}  high-prec={r[1][:18]}...  float64={r[2]}  erro_rel={r[3]}")


if __name__ == "__main__":
    main()
