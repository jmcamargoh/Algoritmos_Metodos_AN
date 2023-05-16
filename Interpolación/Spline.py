import numpy as np
import sympy as sym
import matplotlib.pyplot as plt 

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
        
        # Gráfica
        xcoordenadas = np.array([])
        ycoordenadas = np.array([])
        for seccion in range(1,n,1):    # Recorre cada sección del trazador
            a = xi[seccion-1]       # A y B para cada sección del trazador (si no se hace, quedan las fuciones completas e infinitas)
            b = xi[seccion]
            xseccion = np.linspace(a,b)     # Puntos equiespaciados entre a y b
            pxseccion = tabla_px[seccion-1]     # La función actual del trazador (en esa sección)
            pxt = sym.lambdify(x,pxseccion)     # Convertir a función
            yseccion = pxt(xseccion)            # Evaluación en Y
            xcoordenadas = np.concatenate((xcoordenadas,xseccion))      # Se agregan los puntos anteriores a los arreglos de las coordenadas para que el programa grafique
            ycoordenadas = np.concatenate((ycoordenadas,yseccion))

        plt.plot(xi,fi, 'ro', label='puntos')
        plt.plot(xcoordenadas,ycoordenadas, label='trazador')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title("Gráfico de Trazador Lineal")
        plt.grid(True)
        plt.show()


x = [-2, -1, 2, 3]
y = [12.1353, 6.3678, -4.6109, 2.08553]

Spline(x,y,1)