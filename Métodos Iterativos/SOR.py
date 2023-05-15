import numpy as np
from tabulate import tabulate

def sor_method(A,b,x0,Tol,Niter,w):
    c=0
    error = Tol+1
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    tabla = []
    while error > Tol and c < Niter:
        T = np.linalg.inv(D-w*L)@((1-w)*D+w*U)
        C = w*np.linalg.inv(D-w*L)@b
        x1 = T@x0+C
        E = (np.linalg.norm(x1-x0, ord=np.inf))
        error = E
        if c==0:
            tabla.append([c] + list(x0) + [0])
        else:
            tabla.append([c] + list(x0) + [E_anterior])
        x0 = x1
        c += 1
        E_anterior = E
    if error < Tol:
        s = x0
        print(f"La aproximación de la solución del sistema con una tolerancia = {Tol} es: ")
        print(s)
    else:
        s = x0
        print(f"Fracasó en {Niter} iteraciones")
    
    tabla.append([c] + list(x0) + [E]) 
    headers = ['Iteración'] + [f'x{i+1}' for i in range(len(x0))] + ['Error']
    print(tabulate(tabla, headers=headers, floatfmt=".8f", tablefmt="grid"))
    return (E,s)


A = np.array([[11, 5, 4],
              [5, 25, 4],
              [5, 4, 10]])

b = np.array([10, 10, 10])

x0 = np.array([1, 1, 1])

Tol = 1e-5

Niter = 100

w = 1.5

sor_method(A, b, x0, Tol, Niter, w)