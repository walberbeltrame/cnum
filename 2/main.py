import numpy as np
import matplotlib.pyplot as plt

def is_perfect(n: int) -> bool:
    if n < 1:
        return False

    sum_divisors = 0
    for i in range(1, n):
        if n % i == 0:
            sum_divisors += i

    return sum_divisors == n

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("O número deve ser não negativo.")

    result = 1
    i = n
    while i > 1:
        result *= i
        i -= 1

    return result

def is_prime(n: int) -> bool:
    if n < 2:
        return False

    for i in range(2, int(n/2)):
        if n % i == 0:
            return False

    return True

def sum_of_digits(n: int) -> int:
    if n < 0:
        raise ValueError("O número deve ser não negativo.")
    sum = 0
    while n > 0:
        sum += n % 10
        n //= 10
    return sum

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[10, 11, 12],
              [13, 14, 15],
              [16, 17, 18]])

C = A @ B

def plot(n: int) -> None:
    x = np.linspace(-np.pi, np.pi, n)   # n valores entre -pi e pi
    y_sen = np.sin(x)                   # array senos dos valores de x
    y_cos = np.cos(x)                   # array cossenos dos valores de x

    plt.plot(x, y_sen, label='seno')
    plt.plot(x, y_cos, label='cosseno')
    plt.xlim(-np.pi, np.pi)

    plt.xlabel('Ângulo [rad]')
    plt.ylabel('Função trigonométrica(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig("2/plot.png")  # Salva como imagem no ambiente

    print(f'x =\n{x}')
    print(f'y_sen =\n{y_sen}')
    print(f'y_cos =\n{y_cos}')

def main():
    assert is_perfect(6) == True
    assert is_perfect(7) == False
    assert is_perfect(-1) == False
    assert factorial(5) == 120
    assert factorial(0) == 1
    try:
        factorial(-1)
    except ValueError as error:
        assert str(error) == "O número deve ser não negativo."
    assert is_prime(7) == True
    assert is_prime(10) == False
    assert sum_of_digits(123) == 6
    try:
        sum_of_digits(-1)
    except ValueError as error:
        assert str(error) == "O número deve ser não negativo."
    print(C)
    print(C.shape) # Formato linha por coluna
    print(C.size) # Quantidade de valores
    print(len(C)) # Quantidade de linhas
    plot(35)

if __name__ == "__main__":
    main()