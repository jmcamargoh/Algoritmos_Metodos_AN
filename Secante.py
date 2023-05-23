import re
import numpy as np
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
    return (func(x)); # función no lineal a evaluar

def secante(f, p_0, p_1, tol, n):
    e_abs = abs(p_1 - p_0);
    i = 2;
    resultados =[[0,p_0,f(p_0),""]]
    resultados.append([1,p_1,f(p_1),""])
    while i <= n:
        if f(p_1) == f(p_0): #división por cero
            print('solution not found (error in the initial values)');
            break;
        
        p_2 = p_1 - ((f(p_1)*(p_1 - p_0))/(f(p_1) - f(p_0))); # fórmula del método de la secante
        e_abs = abs(p_1 - p_2);
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

secante(f,0.5,1,10**-7,100);