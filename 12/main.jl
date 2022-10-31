using .MathConstants:e
using Pkg

dependencies = Pkg.project().dependencies

if !haskey(dependencies, "QuadGK")
  Pkg.add("QuadGK")
end

using QuadGK

f(x) = e^(-(x^2))

r = quadgk(f, 0, 1, rtol=0.5)
println(r)

function im(f, a, b)
  h = b - a
  return h*(f((a+b)/2))
end

function it(f, a, b)
  h = b - a
  return h*(((1/2)*f(a))+((1/2)*f(b)))
end

function in(f, a, b)
  h = (b - a) / 2
  return h*(((1/3)*f(a))+((4/3)*f((a+b)/2))+((1/3)*f(b)))
end

function iw(f, i, a, b, t)
  s = 0
  c = a
  d = a + t
  while d <= b
    s += i(f, c, d)
    c = d
    d += t
  end
  return s
end

r = iw(f, im, 0, 1, 0.5)
println(r)

r = iw(f, it, 0, 1, 0.5)
println(r)

r = iw(f, in, 0, 1, 0.5)
println(r)

# Exercicio 9.2.3
f(x) = e^(4-(x^2))
a = 2
b = 5
t(a, b, n) = (b-a)/n

r = iw(f, im, a, b, t(a, b, 3))
println(r)

r = iw(f, it, a, b, t(a, b, 3))
println(r)

r = iw(f, in, a, b, t(a, b, 3))
println(r)