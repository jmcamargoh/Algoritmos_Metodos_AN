import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def lagrange(xi,y):
    n = len(xi)
    x = sym.Symbol('x')
    polinomio = 0
    divisorL = np.zeros(n, dtype = float)
    for i in range(0,n,1):
        numerador = 1
        denominador = 1
        for j  in range(0,n,1):
            if (j!=i):
                numerador = numerador*(x-xi[j])
                denominador = denominador*(xi[i]-xi[j])
        terminoLi = numerador/denominador

        polinomio = polinomio + terminoLi*y[i]
        divisorL[i] = denominador

    polisimple = polinomio.expand()

    px = sym.lambdify(x,polisimple)

    muestras = 101
    a = np.min(xi)
    b = np.max(xi)
    pxi = np.linspace(a,b,muestras)
    pfi = px(pxi)

    print('Divisores L(i): ',divisorL)
    print("")
    print('Expresiones del Polinomio de Lagrange')
    print(polinomio)
    print("")
    print('Polinomio de Lagrange')
    print(polisimple)

    plt.plot(xi,y,'o', label = 'Punto')
    plt.plot(pxi,pfi, label = 'Polinomio')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfico del Polinomio de Lagrange')
    plt.grid(True)
    plt.show()


x = [-1, 0, 3, 4]
y = [15.5, 3, 8, 1]

lagrange(x,y)