using .MathConstants:e
using Pkg

dependencies = Pkg.project().dependencies

if !haskey(dependencies, "Calculus")
  Pkg.add("Calculus")
end

using Calculus

r = derivative(sin, 0.0)
println(r)

f(x) = sin(x) - (x^2)

dp(f,x,h) = (f(x+h)-f(x))/h
dr(f,x,h) = (f(x)-f(x-h))/h
dc(f,x,h) = (f(x+h)-f(x-h))/(2*h)

r = derivative(f, 2.0)
println(r)

r = dp(f,2.0,0.1)
println(r)

r = dp(f,2.0,0.01)
println(r)

# Exercicio 8.1.1
f(x) = sin(x)
r = dp(f,2.0,0.01)
println(r)
r = dr(f,2.0,0.01)
println(r)
r = dc(f,2.0,0.01)
println(r)
r = dp(f,2.0,0.001)
println(r)
r = dr(f,2.0,0.001)
println(r)
r = dc(f,2.0,0.001)
println(r)

f(x) = e^(-x)
r = dp(f,1.0,0.01)
println(r)
r = dr(f,1.0,0.01)
println(r)
r = dc(f,1.0,0.01)
println(r)
r = dp(f,1.0,0.001)
println(r)
r = dr(f,1.0,0.001)
println(r)
r = dc(f,1.0,0.001)
println(r)

# Exercicio 8.1.4
# r = (f(1.0+0.5)-f(1.0))/0.5
# r = (f(4.5+0.5)-f(4.5))/0.5
r = (2.69-1.83)/0.5
println(r)
r = (8.29-7.06)/0.5
println(r)
# r = (f(1.0)-f(1.0-0.5))/0.5
# r = (f(4.5)-f(4.5-0.5))/0.5
r = (1.83-1.05)/0.5
println(r)
r = (7.06-6.11)/0.5
println(r)
# r = (f(1.0+0.5)-f(1.0-0.5))/2*0.5
# r = (f(4.5+0.5)-f(4.5-0.5))/2*0.5
r = (2.69-1.05)
println(r)
r = (8.29-6.11)
println(r)

M(x) = [x[1] 0 x[1]^3 ;
        x[2] 0 x[2]^3 ;
        x[3] 0 x[3]^3 ;
        x[4] 0 x[4]^3 ;
        x[5] 0 x[5]^3 ;
        x[6] 0 x[6]^3 ;
        x[7] 0 x[7]^3 ;
        x[8] 0 x[8]^3 ;
        x[9] 0 x[9]^3 ;
        x[10] 0 x[10]^3 ;
        x[11] 0 x[11]^3 ]

A = M([0.0 0.50 1.00 1.50 2.00 2.50 3.00 3.50 4.00 4.50 5.00])
B = [0.0 ; 1.05 ; 1.83 ; 2.69 ; 3.83 ; 4.56 ; 5.49 ; 6.56 ; 6.11 ; 7.06 ; 8.29]

a = A\B
f(x) = a[1]*x + a[3]*x^3
r = derivative(f,1.0)
println(r)
r = derivative(f,4.5)
println(r)