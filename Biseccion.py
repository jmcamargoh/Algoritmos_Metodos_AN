import numpy as np
from math import *
from tabulate import tabulate

def f(x):
    return (x-np.cos(x));
    
def bisection(f,a,b,tol,n):
    resultados=[]
    if f(a)*f(b)>=0:
        print("El intervalo no cambia de signo");
        
    e_abs = abs(b-a);
    i = 1;
    while i <= n and e_abs > tol:
        c = (a + b)/2;
        if f(c)==0:
            print("Solución encontrada en x=", c);
            break;
        if f(a)*f(c)<0:
            b = c;
            c_t = a;
        else:
            a = c;
            c_t = b;
        e_abs = abs(c_t - c);
        if(i!=1):
            resultados.append([i,a,c,b,f(c),e_abs])
        else:
            resultados.append([i,a,c,b,f(c),""])


        if e_abs < tol:
            print("Solución encontrada en x=", c, ", Iteración:", i);
            break;
        i += 1;
    if i > n:
        print("Solución no encontrada para la tolerancia:" , tol," Iteraciones Utilizadas", i-1);
    print(tabulate(resultados, headers=["Iteraciones", "a", "xm", "b", "f(xm)", "Error"], tablefmt="github", floatfmt=(".0f",".10f",".10f",".10f")))
            
bisection(f,-10,10,10**-10,50);