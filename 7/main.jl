using LinearAlgebra

function fixedpoint(a, g, error)
  x = g(a)
  while norm(x-a) > error
    a = x
    x = g(a)
  end
  return x
end

g(x) = x - (inv(J(x))*F(x))

F(x) = [ ((x[1]^2)/3 + x[2]^2 - 1) ;
         ((x[1]^2) + (x[2]^2)/4 - 1) ]

J(x) = [ (2*x[1]/3) (2*x[2]) ;
         (2*x[1]) (x[2]/2) ]

x = [1 ; 1]
error = 10^-8
r = fixedpoint(x, g, error)
println(r)