import re
import numpy as np
import matplotlib.pyplot as plt
from math import *
from tabulate import tabulate

def reemplazar_funciones_matematicas(expr):
    # Expresión regular para buscar nombres de funciones matemáticas y operadores matemáticos
    pattern = r'\b(sin|cos|tan|sqrt|exp|log|log10)|(\*\*|\^|\+|\-|\*|\/)'
    # Función para reemplazar cada nombre de función y operador matemático
    def replace(match):
        # Si es una función matemática, devuelve su versión con prefijo 'numpy.'
        if match.group(1):
            return f'np.{match.group(1)}'
        # Si es el carácter '^', devuelve el operador '**'
        elif match.group(2) == '^':
            return '**'
        # De lo contrario, devuelve el operador o carácter original
        else:
            return match.group(2)
    # Reemplaza los nombres de funciones y operadores en la expresión por sus equivalentes
    return re.sub(pattern, replace, expr)

expr = 'log(sin(x)^2 + 1)-(1/2)'
expr_with_numpy = reemplazar_funciones_matematicas(expr)   # Convertir función a string
func = eval(f"lambda x: {expr_with_numpy}") # Convertir string a función

def f(x):
    return (func(x));
    
def bisection(f,a,b,tol,n):
    resultados=[]
    if f(a)*f(b)>=0:
        print("El intervalo no cambia de signo");
        
    e_abs = abs(b-a);
    i = 1;
    while i <= n and e_abs > tol:
        a_ant = a
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
            resultados.append([i,a_ant,c,b,f(c),e_abs])
        else:
            resultados.append([i,a_ant,c,b,f(c),""])


        if e_abs < tol:
            print("Solución encontrada en x=", c, ", Iteración:", i);
            break;
        i += 1;
    if i > n:
        print("Solución no encontrada para la tolerancia:" , tol," Iteraciones Utilizadas", i-1);
    print(tabulate(resultados, headers=["Iteraciones", "a", "xm", "b", "f(xm)", "Error"], tablefmt="github", floatfmt=(".0f",".10f",".10f",".10f")))

    x = np.linspace(-10, 10, 1000)
    y = f(x)

    plt.plot(x,y, color='red')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Gráfico de la Función: {expr}")
    plt.grid(True)
    plt.show()
            
bisection(f,0,1,10**-7,100);