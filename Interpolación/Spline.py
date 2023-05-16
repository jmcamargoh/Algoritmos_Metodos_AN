import numpy as np
import sympy as sym
import matplotlib.pyplot as plt 

def Spline(xi, fi, d):
    n = len(xi)
    x = sym.Symbol('x')
    tabla_px = []

    # Lineal
    if d==1:
        for i in range(1,n,1):
            numerador = fi[i]-fi[i-1]
            denominador = xi[i]-xi[i-1]
            m = numerador/denominador
            px = y[i-1]
            px = px + m*(x-xi[i-1])
            tabla_px.append(px)

        print("Trazadores Lineales: ")
        for i in range(1,n,1):      # Salen al revés, para evitar todo el problema que tuve con Diferencias
            px = tabla_px[i-1]
            print(px)
        
        # Gráfica
        graficar(n,tabla_px,xi,fi, 1)


    # Cúbico
    if d==3:
        h = np.zeros(n-1, dtype = float)
        for j in range(0,n-1,1):
            h[j] = xi[j+1] - xi[j]
    
        A = np.zeros(shape=(n-2,n-2), dtype = float)        # Sistema de Ecuaciones
        B = np.zeros(n-2, dtype = float)
        S = np.zeros(n, dtype = float)
        A[0,0] = 2*(h[0]+h[1])
        A[0,1] = h[1]
        B[0] = 6*((fi[2]-fi[1])/h[1] - (fi[1]-fi[0])/h[0])
        for i in range(1,n-3,1):
            A[i,i-1] = h[i]
            A[i,i] = 2*(h[i]+h[i+1])
            A[i,i+1] = h[i+1]
            B[i] = 6*((fi[i+2]-fi[i+1])/h[i+1] - (fi[i+1]-fi[i])/h[i])
        A[n-3,n-4] = h[n-3]
        A[n-3,n-3] = 2*(h[n-3]+h[n-2])
        B[n-3] = 6*((fi[n-1]-fi[n-2])/h[n-2] - (fi[n-2]-fi[n-3])/h[n-3])
    
        r = np.linalg.solve(A,B)        # Resolver sistema de ecuaciones
        for j in range(1,n-1,1):
            S[j] = r[j-1]
        S[0] = 0
        S[n-1] = 0
    
        a = np.zeros(n-1, dtype = float)        # Coeficientes
        b = np.zeros(n-1, dtype = float)
        c = np.zeros(n-1, dtype = float)
        d = np.zeros(n-1, dtype = float)
        for j in range(0,n-1,1):
            a[j] = (S[j+1]-S[j])/(6*h[j])
            b[j] = S[j]/2
            c[j] = (fi[j+1]-fi[j])/h[j] - (2*h[j]*S[j]+h[j]*S[j+1])/6
            d[j] = fi[j]
    
        x = sym.Symbol('x')                 # Polinomio trazador
        polinomio = []
        for j in range(0,n-1,1):
            ptramo = a[j]*(x-xi[j])**3 + b[j]*(x-xi[j])**2 + c[j]*(x-xi[j])+ d[j]
            ptramo = ptramo.expand()
            polinomio.append(ptramo)

        print("Trazadores Cúbicos: ")
        for i in range(1,n,1):      # Trazadores
            px = polinomio[i-1]
            print(px)

        graficar(n, polinomio, xi, fi, 3)


def graficar(n, arreglo, xi, fi, grado):
    x = sym.Symbol('x')
    xcoordenadas = np.array([])
    ycoordenadas = np.array([])
    for seccion in range(1,n,1):    # Recorre cada sección del trazador
        a = xi[seccion-1]       # A y B para cada sección del trazador (si no se hace, quedan las fuciones completas e infinitas)
        b = xi[seccion]
        xseccion = np.linspace(a,b)     # Puntos equiespaciados entre a y b
        pxseccion = arreglo[seccion-1]     # La función actual del trazador (en esa sección)
        pxt = sym.lambdify(x,pxseccion)     # Convertir a función
        yseccion = pxt(xseccion)            # Evaluación en Y
        xcoordenadas = np.concatenate((xcoordenadas,xseccion))      # Se agregan los puntos anteriores a los arreglos de las coordenadas para que el programa grafique
        ycoordenadas = np.concatenate((ycoordenadas,yseccion))

    plt.plot(xi,fi, 'ro', label='puntos')
    plt.plot(xcoordenadas,ycoordenadas, label='trazador')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if grado==1:
        plt.title("Gráfico de Trazadores Lineales")
    if grado==2:
        plt.title("Gráfico de Trazadores Cuadráticos")
    if grado==3:
        plt.title("Gráfico de Trazadores Cúbicos")
    plt.grid(True)
    plt.show()


x = [-2, -1, 2, 3]
y = [12.1353, 6.3678, -4.6109, 2.08553]

Spline(x,y,3)