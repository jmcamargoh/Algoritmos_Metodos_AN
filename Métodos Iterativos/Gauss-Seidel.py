# Para ejecutar el método de Jacobi se tiene que poner 0 en el parámetro METHOD
# Para ejecutar el método de Gauss-Seidel se tiene que poner 1 en el parámetro METHOD

import numpy as np
from tabulate import tabulate

def JacobiSeidel(A,b,x0,Tol,Niter,method):
    c = 0
    error = Tol+1
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    tabla = []
    while error > Tol and c < Niter:
        if method == 0:
            T = np.linalg.inv(D)@(L+U)
            C = np.linalg.inv(D)@b
            x1 = T@x0+C
        if method == 1:
            T = np.linalg.inv(D-L)@U
            C = np.linalg.inv(D-L)@b
            x1 = T@x0+C
        E = (np.linalg.norm(x1-x0, ord=np.inf))/(np.linalg.norm(x1, ord=np.inf)) # Con la división si se piden cifras significativas, si no, se quita
        error = E
        tabla.append([c] + list(x0) + [E])
        x0 = x1
        c += 1
    if error < Tol:
        s = x0
        print(f"La aproximación de la solución del sistema con una tolerancia = {Tol} es: ")
        print(s)
    else:
        s = x0
        print(f"Fracasó en {Niter} iteraciones")

    tabla.append([c] + list(x0) + [E]) 
    headers = ['Iteración'] + [f'x{i+1}' for i in range(len(x0))] + ['Error']
    print(tabulate(tabla, headers=headers, floatfmt=".8f"))
    return (E,s)

# Ejemplo de uso
A = np.array([[11, 5, 4],
              [5, 25, 4],
              [5, 4, 10]])

b = np.array([10, 10, 10])

x0 = np.array([1, 1, 1])

Tol = 5e-5

Niter = 100

JacobiSeidel(A, b, x0, Tol, Niter, 1)