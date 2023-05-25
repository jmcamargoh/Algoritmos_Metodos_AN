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

exprf = 'log(sin(x)^2 + 1)-(1/2)-x'
expr_with_numpy_f = reemplazar_funciones_matematicas(exprf)   # Convertir función a string
funcf = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

exprg = 'log(sin(x)^2 + 1)-(1/2)'
expr_with_numpy_g = reemplazar_funciones_matematicas(exprg)   # Convertir función a string
funcg = eval(f"lambda x: {expr_with_numpy_g}") # Convertir string a función

def f1(x):
    return (funcf(x));    #función

def g(x):
    return (funcg(x));  # evaluación

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

    x = np.linspace(-10, 10, 1000)
    y = f1(x)
    evaluacion = g(x)

    plt.plot(x,y, color='red', label='Función')
    plt.plot(x,evaluacion, color='blue', label='g(x)')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("f(x)/g(x)")
    plt.title(f"Gráfico de la Función: {exprf} y su la g(x) escogida: {exprg}")
    plt.legend()
    plt.grid(True)
    plt.show()

punto_fijo(-0.5, 10**-7, 100)