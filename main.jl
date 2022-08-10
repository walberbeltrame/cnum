v = 1

println(v)

f(x) = x^2 + x + 1

z = f(2)

println(z)

function fx(x, a, b, c)
  y(n) = a*n^2 + b*n + c
  return y(x)
end

z = fx(1,2,3,4) + 1

println(z)