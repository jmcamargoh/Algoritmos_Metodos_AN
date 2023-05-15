import numpy as np
import sympy as sym
import matplotlib.pyplot as plt    

def vandermonde(x,y):
    Xi = np.array(x)
    B = np.array(y)
    n = len(x)

    vander = np.zeros(shape=(n,n), dtype=float)

    for i in range(0,n,1):
        for j in range(0,n,1):
            potencia = (n-1)-j
            vander[i,j] = Xi[i]**potencia
    
    coeficiente = np.linalg.solve(vander, B)

    x = sym.Symbol('x')
    polinomio = 0

    for i in range(0,n,1):
        potencia = (n-1)-i
        multiplicador = coeficiente[i]*(x**potencia)
        polinomio = polinomio + multiplicador

    px = sym.lambdify(x, polinomio)
    a = np.min(Xi)
    b = np.max(Xi)
    xin = np.linspace(a,b)
    yin = px(xin)
    
    print("Matriz de Vandermonde: ")
    print(vander)
    print("")
    print("Coeficientes: ")
    print(coeficiente)
    print("")
    print("Polinomio de Vandermonde: ")
    print(polinomio)
    #sym.pprint(polinomio) # Para "visualizar" la potencia

    plt.plot(Xi,B,'o', label='[x,y]')
    plt.plot(xin,yin, label='p(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Gráfico del Polinomio de Vandermonde")
    plt.grid(True)
    plt.show()

x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]

vandermonde(x,y)