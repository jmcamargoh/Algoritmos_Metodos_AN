import numpy as np
from math import *
from tabulate import tabulate

def f(x):
    return (abs(x)**(x-1)-2.5*x-5); # función no lineal a evaluar

def secant(f, p_0, p_1, tol, n):
    e_abs = abs(p_1 - p_0);
    i = 2;
    resultados =[[0,p_0,f(p_0),""]]
    resultados.append([1,p_1,f(p_1),""])
    while i <= n:
        if f(p_1) == f(p_0): #división por cero
            print('solution not found (error in the initial values)');
            break;
        
        p_2 = p_0 - (f(p_0)*(p_1 - p_0))/(f(p_1) - f(p_0)); # fórmula del método de la secante
        e_abs = abs(p_2- p_1);
        resultados.append([i,p_2,f(p_2),e_abs]) 
        
        if e_abs < tol: # condición de parada
            break;
        
        p_0 = p_1;
        p_1 = p_2;
        
        i+=1;
    if i > n:
        print("Solución no encontrada para la tolerancia de:" , tol,"--- Iteraciones Usadas:", i-1);
    print(tabulate(resultados, headers=["Iteraciones", "Xi", "f(xi)", "Error"], tablefmt="github",floatfmt=(".10f",".10f")))
    if i < n:
        print('Aproximación de la raíz encontrada en x = ', p_2);

secant(f,-3,-2,10**-15,2);