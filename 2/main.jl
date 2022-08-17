function e(x)
  termo = 1
  soma = 1
  i = 1
  while termo > 0
    termo = termo * (x/i)
    soma = soma + termo
    i = i + 1
  end
  return [soma i]
end

y = e(1)
println(y)