# Atividade 1
Faça na linguagem Python uma função que verifica se um número é perfeito.

Um número é perfeito se a soma dos divisores for igual a ele mesmo (ex: 6 = 1 + 2 + 3).

A partir de agora, vamos usar um ambiente virtual para execução dos nossos códigos.

Em Python, *venv* é um módulo que permite criar ambientes virtuais isolados, fundamentais para manter as dependências de cada projeto separadas e evitar conflitos com outras aplicações:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Para sair do ambiente, utilize o comando:

```bash
deactivate
```

Crie uma nova branch (versão) do repositório:

```bash
git branch semana2
```

Faça o checkout nessa nova branch:

```bash
git checkout semana2
```

Gabarito:

```python
def is_perfect(n: int) -> bool:
    if n < 1:
        return False

    sum_divisors = 0
    for i in range(1, n):
        if n % i == 0:
            sum_divisors += i

    return sum_divisors == n


def main():
    assert is_perfect(6) == True
    assert is_perfect(7) == False
    assert is_perfect(-1) == False


if __name__ == "__main__":
    main()
```

<br/>

# Atividade 2
Faça na linguagem Python uma função que calcula o fatorial de um número.

O fatorial de um número é o produto do número por todos os antecessores positivos.

Gabarito:

```python
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("O número deve ser não negativo.")

    result = 1
    i = n
    while i > 1:
        result *= i
        i -= 1

    return result
```

Modifique o main para:

```python
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
```

<br/>

# Atividade 3
Faça na linguagem Python uma função que verifica se um número é primo.

Um número que só pode ser dividido por um e por ele mesmo.

```python
def is_prime(n: int) -> bool:
    ...
```

Adicione o teste para o main:

```python
    assert is_prime(7) == True
    assert is_prime(10) == False
```

<br/>

# Atividade 4
Faça na linguagem Python uma função que calcula a soma dos dígitos.

Caso número negativo, dispare uma mensagem de erro (ex: 123, 1 + 2 + 3 = 6).

```python
def sum_of_digits(n: int) -> int:
    ...
```

Adicione o teste para o main:

```python
    assert sum_of_digits(123) == 6
    try:
        sum_of_digits(-1)
    except ValueError as error:
        assert str(error) == "O número deve ser não negativo."
```

<br/>

# Atividade 5
Instale e importe a biblioteca NumPy, que é fundamental para computação científica em Python, oferecendo suporte para arrays e matrizes multidimensionais, além de uma vasta coleção de funções matemáticas de alto nível para operar eficientemente. Para instalar, utilize o comando:

```bash
pip install numpy
```

Escreva um algoritmo que realize o seguinte produto matricial:

[ [1, 2, 3],  
  [4, 5, 6],  
  [7, 8, 9] ]  

multiplicado por:

[ [11, 12, 13],  
  [14, 15, 16],  
  [17, 18, 19] ]

Gabarito:

```python
import numpy as np

...

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[10, 11, 12],
              [13, 14, 15],
              [16, 17, 18]])

C = A @ B

...

def main():
    ...
    print(C)
    print(C.shape) # Formato linha por coluna
    print(C.size) # Quantidade de valores
    print(len(C)) # Quantidade de linhas
```

<br/>

# Atividade 6
Instale e importe a biblioteca Matplotlib, especializada na confecção de gráficos. Para instalar, utilize o comando:

```bash
pip install matplotlib
```

Escreva um algoritmo que mostra a plotagem simples de duas funções, uma senoidal e uma cossenoidal.

Gabarito:

```python
...
import matplotlib.pyplot as plt
...

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
    plt.savefig("plot.png")  # Salva como imagem no ambiente

    print(f'x =\n{x}')
    print(f'y_sen =\n{y_sen}')
    print(f'y_cos =\n{y_cos}')

...

def main():
    ...
    plot(35)
```
