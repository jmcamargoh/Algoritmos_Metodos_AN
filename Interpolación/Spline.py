import numpy as np
import sympy as sym

def Spline(xi, fi, d):
    n = len(xi)
    x = sym.Symbol('x')
    tabla_px = []

    if d==1:
        for i in range(1,n,1):
            numerador = fi[i]-fi[i-1]
            denominador = xi[i]-xi[i-1]
            m = numerador/denominador
            px = y[i-1]
            px = px + m*(x-xi[i-1])
            tabla_px.append(px)

        for i in range(1,n,1):      # Salen al revés, para evitar todo el problema que tuve con Diferencias
            px = tabla_px[i-1]
            print(px)


x = [-2, -1, 2, 3]
y = [12.1353, 6.3678, -4.6109, 2.08553]

Spline(x,y,1)