#importando las librerias
import streamlit as st
import numpy as np
from tabulate import tabulate
import pandas as pd
import re
import matplotlib.pyplot as plt
import sympy as sym

PAGE_CONFIG = {
    'page_title': 'Analitik',
    'page_icon': 'https://www.lahaus.com/assets/favicon-ea5cca58cf16c3d74dbb5d7a70850186978fc251b3833a8670109a274f488a71.ico',
    'layout': 'wide',
    'initial_sidebar_state':'auto'
}

st.set_page_config(**PAGE_CONFIG)
st.set_option('deprecation.showPyplotGlobalUse', False)

#Metodos
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

# Crear la interfaz de usuario con Streamlit
def create_matrix_entry():
    rows = st.number_input("Número de filas", min_value=1, value=1, step=1)
    columns = st.number_input("Número de columnas", min_value=1, value=1, step=1)
    
    # Crear una matriz vacía
    matrix = np.zeros((rows, columns))
    
    # Rellenar la matriz con los valores introducidos por el usuario
    for i in range(rows):
        for j in range(columns):
            matrix[i][j] = st.number_input(f"Valor en posición ({i}, {j})", key=f"{i}-{j}")
    
    # Guardar la matriz en una variable
    return matrix

def create_row_entry_x0():
    rows = st.number_input("Número de fila", min_value=1, value=1, step=1)
    x0 = np.zeros(rows)

    for i in range(rows):
        x0[i] = st.number_input(f"Valor en posición {i}", key=f"{i}")

    return x0

def create_row_entry_b():
    rows = st.number_input("Número de fila", min_value=1, value=1, step=1, key='13')
    x0 = np.zeros(rows)
    
    for i in range(rows):
        x0[i] = st.number_input(f"Valor en posición {i}", key=f'{14+i}')

    return x0

def create_data_lists_X_Y():
    num_values = st.number_input("Cantidad de valores", min_value=1, value=1, step=1)
    
    x_values = []
    y_values = []
    
    for i in range(num_values):
        x = st.number_input(f"Valor de X {i+1}", key=f"x_{i}")
        y = st.number_input(f"Valor de Y {i+1}", key=f"y_{i}")
        
        x_values.append(x)
        y_values.append(y)
    
    return x_values, y_values

def f(x):
    return (func(x))

def f1(x):
    return (func_f(x))

def g(x):
    return (func_g(x))

def df(x):
    return (func_df(x))

def df2(x):
  return (func_df2(x))

#Metodo bisection
def bisection(f,a,b,tol,n):
    resultados=[]
    if f(a)*f(b)>=0:
        st.error("Error mismo intervalo", icon="🚨")
        return None
    elif a>b:
        st.error("Error A no puede ser mayor a B", icon="🚨")
        return None 
        
    e_abs = abs(b-a)
    i = 1
    while i <= n and e_abs > tol:
        c = (a + b)/2
        if f(c)==0:
            st.write("Solución encontrada en x=", c)
            break
        if f(a)*f(c)<0:
            b = c
            c_t = a
        else:
            a = c
            c_t = b
        e_abs = abs(c_t - c)
        if(i!=1):
            resultados.append([i,a,c,b,f(c),e_abs])
        else:
            resultados.append([i,a,c,b,f(c),""])

        if e_abs < tol:
            st.write("Solución encontrada en x=", c, ", Iteración:", i)
            break
        i += 1
    if i > n:
        st.write("Solución no encontrada para la tolerancia:" , tol," Iteraciones Utilizadas", i-1)

    df = pd.DataFrame(resultados, columns=['Iteraciones', 'a', 'xm', "b", "f(xm)", "Error"])

    # st.dataframe(df)
    st.write(tabulate(resultados, headers=["Iteraciones", "a", "xm", "b", "f(xm)", "Error"], tablefmt="github", floatfmt=(".0f",".10f",".10f",".10f")))


            
def plot_function(func):
    x = np.linspace(-10, 10, 1000)
    y = eval(func)
    return x, y

#Metodo Punto Fijo
def fixed_point(x0, tol, itermax): 
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
        st.write("Solución no encontrada, iteraciones utilizadas: ", iter)

    st.write(tabulate(resultados, headers=["Iteraciones", "Xi", "g(xi)", "f(x)", "Error"], tablefmt="github", floatfmt=(".10f",".10f",".10f")))

#Regla falsa
def false_position(f,a,b,tol,n):
    resultados=[]
    if f(a)*f(b)>=0:
        st.error("Error mismo intervalo", icon="🚨")
        return None
    elif a>b:
        st.error("Error A no puede ser mayor a B", icon="🚨")
        return None 
    e_abs = abs(b-a)
    i = 1
    c = a - (f(a)*(b-a))/(f(b)-f(a))
    while i <= n:
        c_1 = c
        resultados.append([i,'%.10f'%a, b,c_1,f(c_1), e_abs ])
        if f(c_1)==0:
            break
        if f(a)*f(c)<0:
            b = c_1
        else:
            a = c_1
        c = a - (f(a)*(b-a))/(f(b)-f(a))
        if e_abs < tol:
            break
        e_abs = abs(c_1 - c)
        i += 1
    if i > n:
        st.write("Solución no encontrada para la tolerancia de:" , tol,"--- Iteraciones Utilizadas:", i-1)
    st.write(tabulate(resultados, headers=["Iteraciones", "a", "b", "xm", "f(m)", "Error"], tablefmt="github", floatfmt=(".0f",".10f",".10f",".10f")))

#Newton
def newton(f, df, p_0, tol, n):
    st.write("Iteración: ", 0, " En el punto inicial = ", p_0)
    resultados=[[0, p_0, f(p_0),""]]
    e_abs = 1
    i = 1
    while i <= n:
        if df(p_0) == 0: #Division por 0 al reemplazar el punto en la derivada
            st.write("Solución no encontrada. La derivada es igual a 0")
            break
            
        p_1 = p_0 - (f(p_0))/(df(p_0)) #Fórmula del Método
        e_abs = abs(p_1 - p_0)
        resultados.append([i,p_1,f(p_1),e_abs])
        if e_abs < tol: #Criterio de Parada
            st.write("Solución encontrada en x = ", p_1, "--- En ", i, " iteraciones")
            break
        p_0 = p_1
        i += 1
    if i > n:
        st.write("Solución no encontrada para la tolerancia:" , tol,"Iteraciones Utilizadas:", i-1)
    st.write(tabulate(resultados, headers=["Iteraciones", "Xi", "f(xi)", "Error"], tablefmt="github", floatfmt=(".10f",".10f")))

#Raices Multiples

def multiple_roots(f,df,df2,x0,tol,n):
  xant = x0
  fant = f(xant)
  e_abs=1000
  iteration = 0
  resultados =[[iteration,xant,f(xant),""]]
  
  while iteration<=n:
    xact = xant - fant * df(xant) / ((df(xant))**2 - fant * df2(xant))
    fact = f(xact)
    e_abs = abs(xact-xant)
    iteration += 1
    xant = xact
    fant = fact
    resultados.append([iteration,xant,f(xant),e_abs])    
    
    if e_abs<tol:
      st.write("Solución encontrada en x =", xact, "     Iteraciones:", iteration-1, "    Error =", e_abs)
      break
  
  if iteration > n:
    st.write("Solution not found for tolerance = ", tol)
  st.write(tabulate(resultados, headers=["Iteraciones", "Xi", "f(x)", "Error"], tablefmt="github"))

def secante(f, p_0, p_1, tol, n):
    if p_0==p_1:
        st.error("X0 no puede ser igual a X1", icon="🚨")
        return None
    e_abs = abs(p_1 - p_0)
    i = 2
    resultados =[[0,p_0,f(p_0),""]]
    resultados.append([1,p_1,f(p_1),""])
    while i <= n:
        if f(p_1) == f(p_0): #división por cero
            st.write('solution not found (error in the initial values)')
            break
        
        p_2 = p_1 - ((f(p_1)*(p_1 - p_0))/(f(p_1) - f(p_0))); # fórmula del método de la secante
        e_abs = abs(p_1 - p_2)
        resultados.append([i,p_2,f(p_2),e_abs]) 
        
        if e_abs < tol: # condición de parada
            break
        
        p_0 = p_1
        p_1 = p_2
        
        i+=1
    if i > n:
        st.write("Solución no encontrada para la tolerancia de:" , tol,"--- Iteraciones Usadas:", i-1)
    st.write(tabulate(resultados, headers=["Iteraciones", "Xi", "f(xi)", "Error"], tablefmt="github",floatfmt=(".10f",".10f")))
    if i < n:
        st.write('Aproximación de la raíz encontrada en x = ', p_2)

#Métodos de capitulo 2

#Jacobi
def JacobiSeidel(A,b,x0,Tol,Niter,method):
    c = 0
    error = Tol+1
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    tabla = []
    while error > Tol and c < Niter:
        if method == 0:
            T = np.linalg.inv(D)@(L+U)
            C = np.linalg.inv(D)@b
            x1 = T@x0+C
        if method == 1:
            T = np.linalg.inv(D-L)@U
            C = np.linalg.inv(D-L)@b
            x1 = T@x0+C
        E = (np.linalg.norm(x1-x0, ord=np.inf))/(np.linalg.norm(x1, ord=np.inf)) # Con la división si se piden cifras significativas, si no, se quita
        error = E
        if c==0:
            tabla.append([c] + list(x0) + [0])
        else:
            tabla.append([c] + list(x0) + [E_anterior])
        x0 = x1
        c += 1
        E_anterior = E
    if error < Tol:
        s = x0
        eigenvalores = np.linalg.eigvals(T)
        max_eig = np.max(np.abs(eigenvalores))
        st.write("Matriz T: ")
        st.write(T)
        st.write("")
        st.write(f"Eigenvalues: {max_eig}")
        st.write("")
        st.write(f"La aproximación de la solución del sistema con una tolerancia = {Tol} es: ")
        st.write(s)
    else:
        s = x0
        st.error(f"Fracasó en {Niter} iteraciones", icon="🚨")

    tabla.append([c] + list(x0) + [E]) 
    df = pd.DataFrame(tabla, columns=['Iteración', 'x1', 'x2', 'x3', 'Error'])
    st.write(df, floatfmt=".8f", tablefmt="grid")
    return (E,s)

#Sor
def sor_method(A,b,x0,Tol,Niter,w):
    c=0
    error = Tol+1
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    tabla = []
    while error > Tol and c < Niter:
        T = np.linalg.inv(D-w*L)@((1-w)*D+w*U)
        C = w*np.linalg.inv(D-w*L)@b
        x1 = T@x0+C
        E = (np.linalg.norm(x1-x0, ord=np.inf))
        error = E
        if c==0:
            tabla.append([c] + list(x0) + [0])
        else:
            tabla.append([c] + list(x0) + [E_anterior])
        x0 = x1
        c += 1
        E_anterior = E
    if error < Tol:
        s = x0
        eigenvalores = np.linalg.eigvals(T)
        max_eig = np.max(np.abs(eigenvalores))
        st.write("Matriz T: ")
        st.write(T)
        st.write("")
        st.write(f"Eigenvalues: {max_eig}")
        st.write("")
        st.write(f"La aproximación de la solución del sistema con una tolerancia = {Tol} es: ")
        st.write(s)
    else:
        s = x0
        st.error(f"Fracasó en {Niter} iteraciones", icon="🚨")
    
    tabla.append([c] + list(x0) + [E]) 
    df = pd.DataFrame(tabla, columns=['Iteración', 'x1', 'x2', 'x3', 'Error'])
    st.write(df, floatfmt=".8f", tablefmt="grid")
    return (E,s)


#Métodos de capitulo 3

#vandermonde
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
    
    st.subheader("Matriz de Vandermonde: ")
    st.write(vander)
    st.subheader("Coeficientes: ")
    st.write(coeficiente)
    st.subheader("Polinomio de Vandermonde: ")
    st.write(polinomio)
    #sym.pprint(polinomio) # Para "visualizar" la potencia

    plt.plot(Xi,B,'o', label='[x,y]')
    plt.plot(xin,yin, label='p(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Gráfico del Polinomio de Vandermonde")
    plt.grid(True)
    # plt.show()
    st.pyplot()
        

#Newton
def diferencias_divididas(Xi,y):
    n = len(Xi)
    tabla = np.zeros(shape=(n,n+1),dtype=float)

    for i in range(n):      # Puntos x y
        tabla[i,0] = Xi[i]
        tabla[i,1] = y[i]
    
    coeficientes = []
    coeficientes.append(tabla[0,1])     # Primer Coeficiente de la tabla   

    x = sym.Symbol('x')
    polinomio = str(tabla[0,1])     # Primer elemento del Polinomio

    for j in range(2,n+1):
        for i in range(j-1,n):
            tabla[i,j] = (tabla[i,j-1] - tabla[i-1,j-1])/(tabla[i,0] - tabla[i-j+1,0])      # Llenado de la tabla
            if i==j-1:
                coeficientes.append(tabla[i,j])     # Resto de Coeficientes
                if tabla[i,j] < 0:
                    polinomio += str(tabla[i,j])        # Construcción del Polinomio
                else:
                    polinomio += "+" + str(tabla[i,j])
                for i in range(0,i):
                    polinomio += "*(x - " + str(tabla[i,0]) + ")"

    polinomio_imprimir = polinomio.replace("- -", "+ ")        # Reemplaza en el str los - - por + (en la l+ogica matemática el programa lo entiende)

    expr = sym.sympify(polinomio)   # De string a expresión
    func = sym.lambdify(x,expr)    # De expresión a función
    a = np.min(Xi)
    b = np.max(Xi)
    xin = np.linspace(a,b)
    yin = func(xin)

    headers = ["X"] + ["Y"] + [f'{x+1}A' for x in range(n-1)]
    st.subheader("Tabla de Diferencias Divididas: ")
    
    df = pd.DataFrame(tabla, columns=headers)
    st.write(df, floatfmt=".8f", tablefmt="grid")
    st.subheader("Coeficientes: ")
    st.write(coeficientes)
    st.subheader("Polinomio de Diferencias Divididas de Newton: ")
    st.write(polinomio_imprimir)

    plt.plot(Xi,y, 'o', label='[x,y]')      # Impresión de la gráfica
    plt.plot(xin,yin, label='p(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title("Gráfico de Diferencias Divididas de Newton")
    plt.grid(True)
    # plt.show()
    st.pyplot()


def Spline(xi, fi, d):
    n = len(xi)
    x = sym.Symbol('x')

    # Lineal
    if d==1:
        tabla_px = []
        for i in range(1,n,1):
            numerador = fi[i]-fi[i-1]
            denominador = xi[i]-xi[i-1]
            m = numerador/denominador
            px = fi[i-1]
            px = px + m*(x-xi[i-1])
            tabla_px.append(px)

        st.write("Trazadores Lineales: ")
        for i in range(1,n,1):      # Salen al revés, para evitar todo el problema que tuve con Diferencias
            px = tabla_px[i-1]
            st.write(px)
 
        graficar(n,tabla_px,xi,fi, 1)
    # Cúbico
    elif d==3:
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

        st.write("Trazadores Cúbicos: ")
        for i in range(1,n,1):      # Trazadores
            px = polinomio[i-1]
            st.write(px)

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
    elif grado==2:
        plt.title("Gráfico de Trazadores Cuadráticos")
    elif grado==3:
        plt.title("Gráfico de Trazadores Cúbicos")
    plt.grid(True)
    # plt.show()
    st.pyplot()




#Streamlit
options_cap1= ['Bisección', 'Punto Fijo', 'Regla Falsa', 'Newton', 'Raices Multiples', 'Secante']
options_cap2= ['Jacobi', 'Gauss', 'Sor']
options_cap3= ['Vandermonde', 'Spline', 'Newton']
options_spline =['Lineal', 'Cubica']

#Sidebar
with st.sidebar:
    st.subheader("Trabajo Analisis Numerico")
    
    st.text("Capitulo 1")
    metodo_seleccionado_capitulo1 = st.selectbox('¿Cual es el metodo que deseas seleccionar del capitulo 1?', [""] + options_cap1 )

    st.text("Capitulo 2")
    metodo_seleccionado_capitulo2 = st.selectbox('¿Cual es el metodo que deseas seleccionar del capitulo 2?', [""] + options_cap2)

    st.text("Capitulo 3")
    metodo_seleccionado_capitulo3 = st.selectbox('¿Cual es el metodo que deseas seleccionar del capitulo 3?',[""] + options_cap3)

#Area principal capitulo 1
#Biseccion
if metodo_seleccionado_capitulo1 == 'Bisección':
    input_function = st.text_input('Digite la función a evaluar')
    function_name = st.latex(input_function)
    st.info('Recuerde que A debe ser menor a B', icon="ℹ️")
    interval_a = st.number_input('Digite el intervalo a', min_value=-500, max_value=500, step=1)
    interval_b = st.number_input('Digite el intervalo b', min_value=-500, max_value=500, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy = reemplazar_funciones_matematicas(input_function)
    st.text(expr_with_numpy) 
    if expr_with_numpy:
        func = eval(f"lambda x: {expr_with_numpy}") # Convertir string a función
    if st.button('Graficar'):    
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

#Punto Fijo
elif metodo_seleccionado_capitulo1 == 'Punto Fijo':
    st.info('Asegurese que F(x) sea continua y G(x) uniforme y continua en el intervalo', icon="ℹ️")
    input_function_f = st.text_input('Digite la función f a evaluar')
    function_name_f = st.latex(input_function_f)
    input_function_g = st.text_input('Digite la función g a evaluar')
    function_name_g = st.latex(input_function_g)
    initial_value = st.number_input('Digite el valor inicial X0', min_value=-500.0, max_value=500.0, step=0.5,format="%.2f")
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy_f = reemplazar_funciones_matematicas(input_function_f)
    st.text(expr_with_numpy_f)
    expr_with_numpy_g = reemplazar_funciones_matematicas(input_function_g)
    st.text(expr_with_numpy_g)
    if expr_with_numpy_f:
        func_f = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

    if expr_with_numpy_g:
        func_g = eval(f"lambda x: {expr_with_numpy_g}") # Convertir string a función
    if st.button('Graficar f(x)'):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_f)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_f}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    if st.button('Graficar g(x)'):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_g)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_g}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

#Regla Falsa
elif metodo_seleccionado_capitulo1 == 'Regla Falsa':
    input_function = st.text_input('Digite la función a evaluar')
    function_name = st.latex(input_function)
    interval_a = st.number_input('Digite el intervalo a', min_value=-500, max_value=500, step=1, )
    interval_b = st.number_input('Digite el intervalo b', min_value=-500, max_value=500, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy = reemplazar_funciones_matematicas(input_function)
    st.text(expr_with_numpy) 
    if expr_with_numpy:
        func = eval(f"lambda x: {expr_with_numpy}") # Convertir string a función
    if st.button('Graficar'):    
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

#Newton
elif metodo_seleccionado_capitulo1 == 'Newton':
    st.info('Debe asegurarse que f(x) sea continua para el intervalo y que la derivada no sea igual a cero en ninguno de los puntos del intervalo que se analiza', icon="ℹ️")
    input_function_f = st.text_input('Digite la función f a evaluar')
    function_name_f = st.latex(input_function_f)
    input_function_df = st.text_input('Digite la función df a evaluar')
    function_name_df = st.latex(input_function_df)
    initial_value = st.number_input('Digite el valor inicial X0', min_value=-500.0, max_value=500.0, step=0.5,format="%.2f")
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy_f = reemplazar_funciones_matematicas(input_function_f)
    st.text(expr_with_numpy_f)
    expr_with_numpy_df = reemplazar_funciones_matematicas(input_function_df)
    st.text(expr_with_numpy_df)
    if expr_with_numpy_f:
        func_f = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

    if expr_with_numpy_df:
        func_df = eval(f"lambda x: {expr_with_numpy_df}") # Convertir string a función
    if st.button('Graficar f(x)'):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_f)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_f}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    if st.button("Graficar f'(x)"):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_df)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_df}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

#Raices Multiples
elif metodo_seleccionado_capitulo1 == 'Raices Multiples':
    st.info('Debe asegurarse que f(x) sea continua para el intervalo y que la derivada y la segunda derivada esten correctas', icon="ℹ️")
    input_function_f = st.text_input('Digite la función f a evaluar')
    function_name_f = st.latex(input_function_f)
    input_function_df = st.text_input('Digite la función df a evaluar')
    function_name_df = st.latex(input_function_df)
    input_function_df2 = st.text_input('Digite la función df2 a evaluar')
    function_name_df2 = st.latex(input_function_df2)
    initial_value = st.number_input('Digite el valor inicial X0', min_value=-500, max_value=500, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy_f = reemplazar_funciones_matematicas(input_function_f)
    st.text(expr_with_numpy_f)
    expr_with_numpy_df = reemplazar_funciones_matematicas(input_function_df)
    st.text(expr_with_numpy_df)
    expr_with_numpy_df2 = reemplazar_funciones_matematicas(input_function_df2)
    st.text(expr_with_numpy_df2)
    if expr_with_numpy_f:
        func_f = eval(f"lambda x: {expr_with_numpy_f}") # Convertir string a función

    if expr_with_numpy_df:
        func_df = eval(f"lambda x: {expr_with_numpy_df}") # Convertir string a función

    if expr_with_numpy_df2:
        func_df2 = eval(f"lambda x: {expr_with_numpy_df2}") # Convertir string a función
    if st.button('Graficar f(x)'):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_f)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_f}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    if st.button("Graficar f'(x)"):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_df)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_df}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    if st.button("Graficar f''(x)"):
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy_df2)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function_df2}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)

#Secante
elif metodo_seleccionado_capitulo1 == 'Secante':
    input_function = st.text_input('Digite la función a evaluar')
    function_name = st.latex(input_function)
    interval_a = st.number_input('Digite el valor inicial (x0)', min_value=-500.0, max_value=500.0, step=0.5)
    interval_b = st.number_input('Digite el valor inicial (x1)', min_value=-500, max_value=500, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)
    max_iterations = st.number_input('Digite la iteración máxima', min_value=1, step=1)
    expr_with_numpy = reemplazar_funciones_matematicas(input_function)
    st.text(expr_with_numpy) 
    if expr_with_numpy:
        func = eval(f"lambda x: {expr_with_numpy}") # Convertir string a función
    if st.button('Graficar'):    
        # Procesar la función ingresada por el usuario
        x = np.linspace(-8, 8, 1000)
        y = eval(expr_with_numpy)  # Evaluar la expresión matemática

        # Crear el gráfico con tamaño ajustado
        fig, ax = plt.subplots(figsize=(8, 6))  # Ajustar el tamaño de la figura aquí
        ax.plot(x, y, color='red', label='Función')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gráfico de la Función: {input_function}")
        ax.legend()
        ax.grid(True)

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig) 

#Area principal capitulo 2
if metodo_seleccionado_capitulo2 == 'Jacobi':

    st.subheader('Datos Matriz A')
    # Llamar a la función para crear la entrada de la matriz
    A_matrix_entry = create_matrix_entry()
    # Mostrar la matriz en la interfaz de usuario
    st.write("Matriz A creada:")
    st.write(A_matrix_entry)

    st.subheader('Datos X0')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada X0
    row_x0 = create_row_entry_x0()
    st.write('X0 Creado:')
    st.write(row_x0)

    st.subheader('Datos B')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada B
    row_b = create_row_entry_b()
    st.write('B Creado:')
    st.write(row_b)

    Niter = st.number_input('Ingrese el número de iteraciones: ', min_value=1, value=100, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)

elif metodo_seleccionado_capitulo2 == 'Gauss':

    st.subheader('Datos Matriz A')
    # Llamar a la función para crear la entrada de la matriz
    A_matrix_entry = create_matrix_entry()
    # Mostrar la matriz en la interfaz de usuario
    st.write("Matriz A creada:")
    st.write(A_matrix_entry)
    
    st.subheader('Datos X0')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada X0
    row_x0 = create_row_entry_x0()
    st.write('X0 Creado:')
    st.write(row_x0)

    st.subheader('Datos B')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada B
    row_b = create_row_entry_b()
    st.write('B Creado:')
    st.write(row_b)

    Niter = st.number_input('Ingrese el número de iteraciones: ',min_value=1, value=100, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)

elif metodo_seleccionado_capitulo2 == 'Sor':

    st.subheader('Datos Matriz A')
    # Llamar a la función para crear la entrada de la matriz
    A_matrix_entry = create_matrix_entry()
    # Mostrar la matriz en la interfaz de usuario
    st.write("Matriz A creada:")
    st.write(A_matrix_entry)
    
    st.subheader('Datos X0')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada X0
    row_x0 = create_row_entry_x0()
    st.write('X0 Creado:')
    st.write(row_x0)

    st.subheader('Datos B')
    st.info('Recuerde que debe ser proporcional al tamaño de la matriz', icon="ℹ️")
    # Llamar a la función para crear la entrada B
    row_b = create_row_entry_b()
    st.write('B Creado:')
    st.write(row_b)

    W = st.number_input('parámetro de relajación',min_value=0.0, max_value=2.0, value=0.4, step=0.1)
    Niter = st.number_input('Ingrese el número de iteraciones: ',min_value=1, value=100, step=1)
    tolerance = st.text_input('Digite la tolerancia',value=1e-7)
    valor = float(tolerance)

#Area principal capitulo 3
if metodo_seleccionado_capitulo3 == 'Vandermonde':
    # Llamar a la función para crear las listas de datos
    x_list, y_list = create_data_lists_X_Y()
    # Mostrar las listas de datos en la interfaz de usuario
    st.write("Lista de valores de X:")
    x = x_list
    st.text(x)

    st.write("Lista de valores de Y:")
    y = y_list
    st.text(y)


elif metodo_seleccionado_capitulo3 == 'Newton':
    # Llamar a la función para crear las listas de datos
    x_list, y_list = create_data_lists_X_Y()
    # Mostrar las listas de datos en la interfaz de usuario

    st.write("Lista de valores de X:")
    x = x_list
    st.text(x)

    st.write("Lista de valores de Y:")
    y = y_list
    st.text(y)

elif metodo_seleccionado_capitulo3 == 'Spline':
    spline = st.selectbox('¿Cual quieres probar?', [""] + options_spline )
    if spline == 'Lineal':
        # Llamar a la función para crear las listas de datos
        x_list, y_list = create_data_lists_X_Y()
        # Mostrar las listas de datos en la interfaz de usuario

        st.write("Lista de valores de X:")
        x = x_list
        st.text(x)

        st.write("Lista de valores de Y:")
        y = y_list
        st.text(y)
    elif spline == 'Cubica':
        # Llamar a la función para crear las listas de datos
        x_list, y_list = create_data_lists_X_Y()
        # Mostrar las listas de datos en la interfaz de usuario

        st.write("Lista de valores de X:")
        x = x_list
        st.text(x)

        st.write("Lista de valores de Y:")
        y = y_list
        st.text(y)


# consultar metodos
if metodo_seleccionado_capitulo1 != '':
    if st.button('consultar'):
        if metodo_seleccionado_capitulo1 == 'Bisección':
            bisection(func,interval_a,interval_b,valor,max_iterations)
        elif metodo_seleccionado_capitulo1 == 'Punto Fijo':
            fixed_point(initial_value,valor,max_iterations)
        elif metodo_seleccionado_capitulo1 == 'Regla Falsa':
            false_position(func,interval_a,interval_b,valor,max_iterations)
        elif metodo_seleccionado_capitulo1 == 'Newton':
            newton(func_f,func_df,initial_value,valor,max_iterations)
        elif metodo_seleccionado_capitulo1 == 'Raices Multiples':
            multiple_roots(func_f,func_df,func_df2,initial_value,valor,max_iterations)
        elif metodo_seleccionado_capitulo1 == 'Secante':
            secante(func,interval_a,interval_b,valor,max_iterations)

elif metodo_seleccionado_capitulo2 != '' :
    if st.button('Consultar'):
        if metodo_seleccionado_capitulo2 == 'Jacobi':
            JacobiSeidel(A_matrix_entry, row_b, row_x0, valor, Niter, 0)
        elif metodo_seleccionado_capitulo2 == 'Gauss':
            JacobiSeidel(A_matrix_entry, row_b, row_x0, valor, Niter, 1)
        elif metodo_seleccionado_capitulo2 == 'Sor':
            sor_method(A_matrix_entry, row_b, row_x0, valor, Niter, W)

elif metodo_seleccionado_capitulo3 != '' :
    if st.button('Consultar'):
        if metodo_seleccionado_capitulo3 == 'Vandermonde':
            vandermonde(x,y)
        elif metodo_seleccionado_capitulo3 == 'Newton':
            diferencias_divididas(x,y)
        elif metodo_seleccionado_capitulo3 == 'Spline':
            if spline == 'Lineal':
                Spline(x, y, 1)
            elif spline == 'Cubica':
                Spline(x, y, 3)
       

            
else:
    col1, col2, col3 = st.columns(3)

    with col2:
        st.title("Analitik")
        st.markdown("AnalitiK is a website where you'll find several methods used to solve numerical analysis problems.")
        st.markdown("---")
        st.latex(r"\def\arraystretch{1.5} \begin{array}{c:c:c} a & b & c \\ \hline d & e & f \\\hdashline g & h & i\end{array}")

st.markdown(
    '''
    <style>
    .css-zt5igj.e16nr0p33 {
        text-align: center;
    }
    </style>
    ''', unsafe_allow_html=True
)

