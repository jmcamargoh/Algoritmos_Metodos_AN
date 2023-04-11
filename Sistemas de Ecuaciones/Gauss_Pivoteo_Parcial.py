import numpy as np
from numpy import *
import math
from math import *

#Ingreso
a = ([[-7, 2, -3, 4],
     [5, -1, 14, -1],
     [1, 9, -7, 5],
     [-12, 13, -8, -4]])

b = ([[-12],
      [13],
      [31],
      [-32]])

#procedimiento
#matriz aumentada

M = np.concatenate((a,b), axis=1)
M = M.astype(np.float64)
#pivoteo parcial por filas
tamano = np.shape(a)
n = tamano[0]

print("Matriz aumentada")
print(M)
print("")

for i in range(0,n-1,1):
    print("iteracion" , i)
    #Cambio de filas
    aux0=np.max(np.abs(M[i:n,i]))
    aux = 0
    for j in range (i+1,n,1):
      if(aux0 == M[j,i]):
        exit
      else:
        aux=aux+1
    if aux0>abs(M[i,i]):
        aux2=np.copy(M[i+aux,i:n])
        M[aux+i,i:n]=M[i,i:n]
        M[i,i:n]=aux2

    for j in range(i+1,n,1):
        if M[j,i]!=0:
          M[j,i:n+1]=M[j,i:n+1]-(M[j,i]/M[i,i])*M[i,i:n+1]
    print(M)
    print("")

ultfila = n-1;
ultcolumna = n;
x = np.zeros(n,dtype=float);
i = ultfila;
for i in range(ultfila,0-1,-1):
    suma = 0;
    for j in range(i+1,ultcolumna,1):
        suma = suma + M[i,j]*x[j];
    b = M[i,ultcolumna];
    x[i] = (b - suma) / M[i,i];
x = np.transpose([x]);

print(x)