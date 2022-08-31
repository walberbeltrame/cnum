using .MathConstants:e

function bisection(a,b,f,error)
  x = (a+b)/2
  while abs(f(x)) > error
    if f(a)*f(x) > 0
      a = x
    else 
      b = x
    end
    x = (a+b)/2
  end
  return x
end

# Exercicio 3.2.1
f(x) = sqrt(x) - cos(x)
error = 10^-4
r = bisection(0, 1, f, error)
println(r)

# Exercicio 3.2.2
f(x) = 5*sin(x^2) - e^(x/10)
error = 10^-5
r = bisection(0.4, 0.5, f, error)
println(r)
r = bisection(1.7, 1.8, f, error)
println(r)
r = bisection(2.5, 2.6, f, error)
println(r)

# Exercicio 3.2.8
Ir = 10^-12
T = 300
k = 1.380649*(10^-23)
q = 1.60217663*(10^-19)
I(x) = Ir*(e^((x*q)/(k*T)) - 1)
g(x,V,R) = R*I(x) + x - V
error = 10^-3
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-30
f(x) = g(x, 30, 1)
r = bisection(0.8, 0.9, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-3
f(x) = g(x, 3, 1)
r = bisection(0.7, 0.8, f, error)
println(r)
# (10*(10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-3
f(x) = g(x, 3, 10)
r = bisection(0.6, 0.7, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x-0.3
f(x) = g(x, 0.3, 1)
r = bisection(0.29, 0.31, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+0.3
f(x) = g(x, -0.3, 1)
r = bisection(-0.29, -0.31, f, error)
println(r)
# ((10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+30
f(x) = g(x, -30, 1)
r = bisection(-29.9, -30.1, f, error)
println(r)
# (10*(10^-12)*(e^((x*(1.60217663*10^-19))/((1.380649*10^-23)*(300)))-1))+x+30
f(x) = g(x, -30, 10)
r = bisection(-29.9, -30.1, f, error)
println(r)