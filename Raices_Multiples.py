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

exprf = 'exp(x)-x-1'
expr_with_numpy_f = reemplazar_funciones_matematicas(exprf)   # Convertir función a string
funcf = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

exprdf = 'exp(x)-1'
expr_with_numpy_df = reemplazar_funciones_matematicas(exprdf)   # Convertir función a string
funcdf = eval(f"lambda x: {expr_with_numpy_df}") # Convertir string a función

exprd2f = 'exp(x)'
expr_with_numpy_d2f = reemplazar_funciones_matematicas(exprd2f)   # Convertir función a string
funcd2f = eval(f"lambda x: {expr_with_numpy_d2f}") # Convertir string a función

def f(x):
  return (funcf(x));  

def df(x):
  return (funcdf(x));

def d2f(x):
  return (funcd2f(x));

def raices_multiples(f,df,d2f,x0,tol,n):
  xant = x0;
  fant = f(xant);
  e_abs=1000;
  iteration = 0;
  resultados =[[iteration,xant,f(xant),""]]
  
  while iteration<=n:
    xact = xant - fant * df(xant) / ((df(xant))**2 - fant * d2f(xant));
    fact = f(xact);
    e_abs = abs(xact-xant);
    iteration += 1;
    xant = xact;
    fant = fact;
    resultados.append([iteration,xant,f(xant),e_abs])    
    
    if e_abs<tol:
      print("Solución encontrada en x =", xact, "     Iteraciones:", iteration-1, "    Error =", e_abs);
      break;
  
  if iteration > n:
    print("Solution not found for tolerance = ", tol);
  print(tabulate(resultados, headers=["Iteraciones", "Xi", "f(x)", "Error"], tablefmt="github"))

raices_multiples(f,df,d2f,1,10**-7,100);