import re
import numpy as np
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

regla_falsa(f,0,1,10**-7,100);  