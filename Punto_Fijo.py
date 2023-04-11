import math
from math import *
import numpy as np
from tabulate import tabulate

def g(x):
    return (math.sqrt((-math.e**-x)+5*x+20));  # evaluación


def f1(x):
    return ((-math.e**-x)-x**2-5*x-20);    #función


def punto_fijo(x0, tol, itermax):
    iter = 0
    resultados = [[iter, x0,  g(x0), f1(x0), "NA"]]
    while iter <= itermax:
        x1 = g(x0)  # evaluar la función en el último punto
        error = abs(x1-x0)
        x0 = x1
        iter += 1
        resultados.append([iter,x0,g(x0), f1(x0), error])
        if error < tol:  # parar al alcanzar la tolerancia indicada
            break
    if iter > itermax:
        print("Solución no encontrada, iteraciones utilizadas: ", iter)

    print(tabulate(resultados, headers=["Iteraciones", "Xi", "g(xi)", "f(x)", "Error"], tablefmt="github", floatfmt=(".10f",".10f",".10f")))

punto_fijo(7, 10**-3, 100)