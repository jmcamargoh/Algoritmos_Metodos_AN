import numpy as np
import sympy as sym
from tabulate import tabulate
import matplotlib.pyplot as plt

def diferencias_divididas(Xi,y):
    n = len(Xi)
    tabla = np.zeros(shape=(n,n+1),dtype=float)

    for i in range(n):      # Puntos x y
        tabla[i,0] = Xi[i]
        tabla[i,1] = y[i]
    
    coeficientes = []
    coeficientes.append(tabla[0,1])     # Primer Coeficiente de la tabla   

    x = sym.Symbol('x')
    polinomio = str(tabla[0,1])     # Primer elemento del Polinomio

    for j in range(2,n+1):
        for i in range(j-1,n):
            tabla[i,j] = (tabla[i,j-1] - tabla[i-1,j-1])/(tabla[i,0] - tabla[i-j+1,0])      # Llenado de la tabla
            if i==j-1:
                coeficientes.append(tabla[i,j])     # Resto de Coeficientes
                if tabla[i,j] < 0:
                    polinomio += str(tabla[i,j])        # Construcción del Polinomio
                else:
                    polinomio += "+" + str(tabla[i,j])
                for i in range(0,i):
                    polinomio += "(x - " + str(tabla[i,0]) + ")"


    headers = ["X"] + ["Y"] + [f'{x+1}A' for x in range(n-1)]
    print("Tabla de Diferencias Divididas: ")
    print(tabulate(tabla, headers=headers, floatfmt=".8f", tablefmt="grid"))
    print("")
    print("Coeficientes: ")
    print(coeficientes)
    print("")
    print("Polinomio de Diferencias Divididas de Newton: ")
    print(polinomio)


x = [19, 22, 25, 31]
y = [2.8, -1.3, 0.5, 3.8]

diferencias_divididas(x,y)