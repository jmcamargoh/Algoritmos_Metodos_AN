import numpy as np
import sympy as sym
from tabulate import tabulate

def vandermonde(x,y):
    Xi = np.array(x)
    b = np.array(y)
    n = len(x)

    vander = np.zeros(shape=(n,n), dtype=float)

    for i in range(0,n,1):
        for j in range(0,n,1):
            potencia = (n-1)-j
            vander[i,j] = Xi[i]**potencia
    
    coeficiente = np.linalg.solve(vander, b)

    x = sym.Symbol('x')
    polinomio = 0

    for i in range(0,n,1):
        potencia = (n-1)-i
        multiplicador = coeficiente[i]*(x**potencia)
        polinomio = polinomio + multiplicador
    
    print("Matriz de Vandermonde: ")
    print(vander)
    print("")
    print("Coeficientes: ")
    print(coeficiente)
    print("")
    print("Polinomio de Vandermonde: ")
    print(polinomio)

x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]

vandermonde(x,y)