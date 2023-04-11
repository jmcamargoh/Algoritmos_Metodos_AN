import numpy as np
from math import *
from tabulate import tabulate

def f(x):
        return (-7*np.log(x)+x-13); # función no lineal a evaluar             
    
def regla_falsa(f,a,b,tol,n):
    if f(a)*f(b)>=0:
        print("El intervalo no cambia de signo");
    resultados=[]
    e_abs = abs(b-a);
    i = 1;
    c = a - (f(a)*(b-a))/(f(b)-f(a));
    while i <= n:
        c_1 = c;
        resultados.append([i,'%.10f'%a, b,c_1,f(c_1), e_abs ])
        if f(c_1)==0:
            break;
        if f(a)*f(c)<0:
            b = c_1;
        else:
            a = c_1;
        c = a - (f(a)*(b-a))/(f(b)-f(a));
        if e_abs < tol:
            break;
        e_abs = abs(c_1 - c);
        i += 1;
    if i > n:
        print("Solución no encontrada para la tolerancia de:" , tol,"--- Iteraciones Utilizadas:", i-1);
    print(tabulate(resultados, headers=["Iteraciones", "a", "b", "xm", "f(m)", "Error"], tablefmt="github", floatfmt=(".0f",".10f",".10f",".10f")))

regla_falsa(f,20,50,0.5*10**-5,10);  