#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:08:19 2025

@author: lia

GVF +CBF segun la tesis de Weija para evitar choques entre robots
Version inicial para 3 agentes que se comunican todos entre sí en 2D

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import qp as qp
import graph_utils as gu


class Agente:
    def __init__(self, posicion):
        """
        Inicializa un agente con una posición dada.
        :param posicion: np.array con las coordenadas [x, y] del agente
        """
        self.posicion = np.array(posicion, dtype=float)
        self.vecinos = []  # Lista de posiciones de los vecinos (apilada) 
    
    def actualizar_vecinos(self, lista_vecinos):
        """
        Actualiza la lista de vecinos del agente.
        :param lista_vecinos: Lista de np.array con las posiciones de los vecinos
        """
        self.vecinos = [np.array(v, dtype=float) for v in lista_vecinos]
    
   
#CASO DE 3 ROBOTS EN UN ESPACIO 2-DIMENSIONAL

#todos los agentes siguen las mismas ecuaciones de trayectoria

np.random.seed(123)

#Definimos en una función las ecuaciones de la trayectoria 

def fun(w):
    f1 = 15*np.sin(2*w)
    f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    f3 = 3+5*np.cos(2*w)
    return np.array([f1, f2, f3])

#funciones phi
def phi(vec_pos):
    f = fun(vec_pos[-1])
    return vec_pos[:-1] - f

# calculamos la derivada de las ecuaciones de la trayectoria
def d_fun(w):
    #derivada central
    h = 1e-8 #paso
    df = (fun(w+h)-fun(w-h))/(2*h)
    return df

#matriz lapaciana de un grafo (los agentes están comunicados con los vecinos adyacentes)
def grafo_L(N):
    #N: nº de agentes 
    L = np.zeros([N,N])  
    
    # Llenar la diagonal con 2 
    np.fill_diagonal(L, 2)
    
    # Llenar las diagonales justo arriba y abajo de la diagonal principal con -1
    np.fill_diagonal(L[:, 1:], -1)
    np.fill_diagonal(L[1:, :], -1)
    
    # Llenar las esquinas con -1
    L[0, -1] = -1
    L[-1, 0] = -1
    
    return L

def vector_field(xi,t,k,n,N,ww,kc,L):
    #xi: posiciones de todos los agentes (vector columna con las 4 coordenadas )
    #t: tiempo
    #k: ganancias positivas
    #n: dimensiones 
    #N: nº de agentes 
    #L: matriz laplaciana de un grafo

    lista = np.arange(0,(n+1)*N+1,(n+1))
    k = k*N

    PHI = np.zeros(n*N) #lista vacía de funciones surface
    df = np.zeros(n*N)  #lista vacía de las derivadas 
    w = np.zeros(N)     #lista vacía para las coordenadas virtuales
    j = 0 
    for i in lista[:-1]: 
        x = xi[i:i+(n+1)]               #coordenada inicial + virtual de cada agente
        PHI[i-j:(i-j)+n] = phi(x)       #función surface 
        df[i-j:(i-j)+n] = d_fun(x[-1])  #derivada 
        w[j] = x[-1]                    #coordenadas virtuales 
        j += 1

    xi_eta = (-1)**n*df-k*PHI

    #sumatorio

    l = k*PHI*df
    # Convertir la matriz l en una matriz de n filas
    matriz_l = l.reshape((N, -1))
    #sumar a los largo del eje de las filas 
    suma = np.sum(matriz_l, axis=1)

    #función de coordinación 
    c = -L@np.transpose(w-ww)
    
    xi4 = (-1)**n+suma+kc*c

    #path-following guiding vector field 
    xi_eta = np.concatenate((xi_eta.reshape((N,n)),np.array([xi4]).T),axis=1)
    xi_eta = xi_eta.reshape((N*(n+1),-1)).T

    return xi_eta[0]


def vector_field_CBF(xi,Chi,n,N,R,alpha):
    #xi: posiciones de todos los agentes (vector columna con las 4 coordenadas )
    #eta: campo guiado para todos los agentes
    #k: ganancias positivas
    #n: dimensiones 
    #N: nº de agentes 
    #R: distancia de seguridad entre los robots
    #alpha: ganancia de la barrera
    # Desapilo el campo (cuidado que lleva la coordenada ampliada)
    Chi=gu.unstack(xi,n+1)
    # Desapilo posiciones
    pi=gu.unstack(xi,n+1)
    Chi_cbf=np.zeros(N,n)
    #Para cada agente
    #Construyo el problema para optimizar
    # min 0.5*Chi_hat'*Chi_hat-Xi'*Chi_hat
    # s.t A*Chi_hat<=b  
    #Aj=(pj-pi)'P'
    #bj=alpha*eta3/4
    #Construyo las matrices
    M=np.eye(n+1,n+1)
    Pr=np.zeros((n+1,n+1))
    Pr[-1,-1]=1
    P=M-Pr
    eta=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            eta[i,j]=np.linalg.norm(P@(pi[i,:]-pi[j,:]))**2-R**2
    
    for i in range(N):
        A=np.zeros(N-1)
        b=np.zeros(N-1)
        for k in range(i):
            A[k]=(pi[k]-pi[i])@P
            b[k]=alpha*eta[i,k]**3/4.0
        for k in range(i+1,N-1):
            A[k-1]=(pi[k-1]-pi[i])@P
            b[k-1]=alpha*eta[i,k-1]**3/4.0
        q=-eta
        Chi_cbf[i,:]=qp.qp_solve(M,-Chi[i,:],G=None,h=None,A=A,b=b,lb=None,ub=None)
    return Chi_cbf
    
 
   
  
#------------------------------------------------------------------------

#Parámetro de simulación 
n = 3           #dimensiones del espacio
N =5          #nº de agentes 
t = np.linspace(0,200,1000) #tiempo de integración 
ki =[1,1,2] #ganancias 

#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
                           #columnas: nº de robots
pos = np.random.randint(-50, 51, size=(N, n))


#añadimos a la matriz de posiciones la coordenada virtual w 
w = np.ones((N,1)) #ejemplo: todos valen 1 

#concatenamos las dos matrices a lo largo de las filas 
Xi = np.concatenate((pos, w), axis=1)
Xi = Xi.reshape((N*(n+1),-1)).T #apilamos en un vector

#matriz lapaciana de un grafo 
L = grafo_L(N)

#parámetros para la componente de coordinación 
T = 2*np.pi #periodo 
Delta =  T/(2*N)
kc = 300

ww = np.zeros(N)
for i in range(N):
    ww[i] = i*Delta
    
R=2
alpha=1
Chi=vector_field(Xi,0,ki,n,N,ww,kc,L)
Chi_hat=vector_field_CBF(Xi,Chi,n,N,R,alpha)


'''  
#resolvemos las ecuación diferencial 
sol = odeint(vector_field,Xi[0],t,args=(ki,n,N,ww,kc,L))

#representación gráfica 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

lista = np.arange(0,(n+1)*N+1,(n+1))
for i in lista[:-1]: 
    ax.plot(sol[:,i], sol[:,i+1], sol[:,i+2])
    ax.scatter(sol[0,i],sol[0,i+1], sol[0,i+2], marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Trayectoria de los agentes')
plt.show()'''