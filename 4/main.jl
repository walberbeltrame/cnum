using .MathConstants:e

function fixedpoint(a, g, error)
  x = g(a)
  while abs(x-a) > error
    a = x
    x = g(a)
  end
  return x
end

g(x) = log(x+2)
error = 10^-8
r = fixedpoint(1.1, g, error)
println(r)

# Exercicio 3.3.1
g(x) = e^x-2
r = fixedpoint(-1.8, g, error)
println(r)

g(x) = x - ((e^x-x-2)/(e^x-1))
r = fixedpoint(1.1, g, error)
println(r)
r = fixedpoint(-1.8, g, error)
println(r)

function secant(a, b, f, error)
  g(a, b) = (a*f(b)-b*f(a))/(f(b)-f(a))
  x = g(a, b)
  while abs(x-b) > error
    a = b
    b = x
    x = g(a, b)
  end
  return x
end

f(x) = e^x-x-2
r = secant(1.1, 1.2, f, error)
println(r)
r = secant(-1.8, -1.9, f, error)
println(r)

# Exercicio 3.4.1
error = 10^-5
g(x) = x + ((cos(x)-(x^2))/(sin(x)+(2*x)))
r = fixedpoint(1, g, error)
println(r)

# Exercicio 3.6.3
f(x) = e^(-(x^2))-(2*x)
r = secant(0.4, 0.5, f, error)
println(r)

# Exercicio 3.6.4
Ir = 10^-12
T = 300
k = 1.380649*(10^-23)
q = 1.60217663*(10^-19)
I(x) = Ir*(e^((x*q)/(k*T)) - 1)
dI(x) = ((Ir*q)/(k*T))*e^((x*q)/(k*T))
g(x,V,R) = R*I(x) + x - V
dg(x,R) = R*dI(x) + 1
d(x,R,f) = x - (f(x)/dg(x,R)) 
error = 10^-3
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-30
f(x) = g(x, 30, 1)
h(x) = d(x, 1, f)
r = fixedpoint(0.8, h, error)
println(r)
r = secant(0.8, 0.9, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-3
f(x) = g(x, 3, 1)
h(x) = d(x, 1, f)
r = fixedpoint(0.7, h, error)
println(r)
r = secant(0.7, 0.8, f, error)
println(r)
# (10*(10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-3
f(x) = g(x, 3, 10)
h(x) = d(x, 10, f)
r = fixedpoint(0.6, h, error)
println(r)
r = secant(0.6, 0.7, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-0.3
f(x) = g(x, 0.3, 1)
h(x) = d(x, 1, f)
r = fixedpoint(0.29, h, error)
println(r)
r = secant(0.29, 0.31, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+0.3
f(x) = g(x, -0.3, 1)
h(x) = d(x, 1, f)
r = fixedpoint(-0.29, h, error)
println(r)
r = secant(-0.29, -0.31, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+30
f(x) = g(x, -30, 1)
h(x) = d(x, 1, f)
r = fixedpoint(-29.9, h, error)
println(r)
r = secant(-29.9, -30.1, f, error)
println(r)
# (10*(10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+30
f(x) = g(x, -30, 10)
h(x) = d(x, 10, f)
r = fixedpoint(-29.9, h, error)
println(r)
r = secant(-29.9, -30.1, f, error)
println(r)