v = 1

println(v)

f(x) = x^2 + x + 1

z = f(2)

println(z)

function fx(x, a, b, c)
  y(n) = a*n^2 + b*n + c
  return y(x)
end

z = fx(1,2,3,4)

println(z)

function zeros(a,b,c)
  if a != 0
    delta = b^2 - 4*a*c
    if delta >= 0
      x1 = (-b+sqrt(delta))/(2*a)
      x2 = (-b-sqrt(delta))/(2*a)
      return [x1 x2]
    end
  else
    return -c/b
  end
end

z = zeros(1,4,2)

println(z)
