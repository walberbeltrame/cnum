include("../5/main.jl")

# Exercicio 4.7.2
A = [5 1 1;
     -1 3 -1;
     1 2 10]

B = [50 ; 10 ; -30]

x = A \ B
println(x)

# Exercicio 4.7.3
x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)

# Exercicio 4.7.5
A = [4 0 1;
     1 10 3;
     2 1 4]

B = [6 ; 27 ; 12]

x = A \ B
println(x)

x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)

# Exercicio 4.9.1 e 4.9.2
A = [1 0 0 0 0;
     (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2) 0 0;
     0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2) 0;
     0 0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2);
     0 0 0 (1/2) ((-1/2)+(-1/50))]

B = [127 ; 0 ; 0 ; 0 ; 0]

x = A \ B
println(x)

x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)

A = [1 0 0 0 0;
     (1/2) ((-1/2)+(-1/2)+(-1/50)) (1/2) 0 0;
     0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2) 0;
     0 0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2);
     0 0 0 (1/2) ((-1/2)+(-1/100))]

B = [127 ; 0 ; 0 ; 0 ; 0]

x = A \ B
println(x)

x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)

A = [((-1/2)+(-1/2)+(-1/100)) (1/2) 0 0;
     (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2) 0;
     0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2);
     0 0 (1/2) ((-1/2)+(-1/50))]

B = [(-127/2) ; 0 ; 0 ; 0]

x = A \ B
println(x)

x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)

A = [((-1/2)+(-1/2)+(-1/50)) (1/2) 0 0;
     (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2) 0;
     0 (1/2) ((-1/2)+(-1/2)+(-1/100)) (1/2);
     0 0 (1/2) ((-1/2)+(-1/100))]

B = [(-127/2) ; 0 ; 0 ; 0]

x = A \ B
println(x)

x = jacobi(A,B,100,error)
println(x)

x = seidel(A,B,100,error)
println(x)