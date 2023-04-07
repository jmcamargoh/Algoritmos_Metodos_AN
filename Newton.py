import numpy as np
import math
from math import *
from tabulate import tabulate


def f(x):
    return (np.cos(x)*(math.e**-x)-13);
def df(x):
    return ((math.e**-x)*(-np.sin(x)-np.cos(x)));
    
def newton(f, df, p_0, tol, n):
    print("Iteración: ", 0, " En el punto inicial = ", p_0);
    resultados=[[0, p_0, f(p_0),""]]
    e_abs = 1;
    i = 1;
    while i <= n:
        if df(p_0) == 0: #Division por 0 al reemplazar el punto en la derivada
            print("Solución no encontrada. La derivada es igual a 0");
            break;
            
        p_1 = p_0 - (f(p_0))/(df(p_0)); #Fórmula del Método
        e_abs = abs(p_1 - p_0);
        resultados.append([i,p_1,f(p_1),e_abs])
        if e_abs < tol: #Criterio de Parada
            print("Solución encontrada en x = ", p_1, "--- En ", i, " iteraciones");
            break;
        p_0 = p_1;
        i += 1;
    if i > n:
        print("Solución no encontrada para la tolerancia:" , tol,"Iteraciones Utilizadas:", i-1);
    print(tabulate(resultados, headers=["Iteraciones", "Xi", "f(xi)", "Error"], tablefmt="github", floatfmt=(".10f",".10f")))


newton(f,df,-4.1,10**-5,100);