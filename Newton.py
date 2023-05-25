import re
import numpy as np
import matplotlib.pyplot as plt
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

exprf = 'log(sin(x)^2 + 1)-(1/2)'
expr_with_numpy_f = reemplazar_funciones_matematicas(exprf)   # Convertir función a string
funcf = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

exprdf = '2*(1/(sin(x)^2 + 1))*(sin(x)*cos(x))'
expr_with_numpy_df = reemplazar_funciones_matematicas(exprdf)   # Convertir función a string
funcdf = eval(f"lambda x: {expr_with_numpy_df}") # Convertir string a función

def f(x):
    return (funcf(x));

def df(x):
    return (funcdf(x));
    
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
    
    x = np.linspace(-10, 10, 1000)
    y = f(x)
    dy = df(x)

    plt.plot(x,y, color='red', label='Función')
    plt.plot(x,dy, color='blue', label='Derivada')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("f(x)/f'(x)")
    plt.title(f"Gráfico de la Función: {exprf} y su Derivada: {exprdf}")
    plt.legend()
    plt.grid(True)
    plt.show()

newton(f,df,0.5,10**-7,100);