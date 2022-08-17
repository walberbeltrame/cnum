function e(x)
  termo = 1
  soma = 1
  n = 1
  while termo > 0
    termo = termo * (x/n)
    soma = soma + termo
    n = n + 1
  end
  return [soma n]
end

y = e(1)
println(y)