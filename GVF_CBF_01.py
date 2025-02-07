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
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

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
    Chi=gu.unstack(Chi,n+1)
    # Desapilo posiciones
    pi=gu.unstack(xi,n+1)
    Chi_cbf=np.zeros((N,n+1))
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
    #TODO: Revisar el calculo de A y b
    for i in range(N):
        A=np.zeros((N-1,n+1))
        b=np.zeros(N-1)
        s=0
        for k in range(N-1):
            if k==i:
                continue
            A[s,:]=(pi[k]-pi[i])@P.T
            b[s]=alpha*eta[i,k]**3/4.0
            s+=1
        q=-eta
        Chi_cbf[i,:]=qp.qp_solve(M,-Chi[i,:],G=A,h=b,A=None,b=None,lb=None,ub=None)
    return Chi_cbf
    
 
def vector_field_completo(xi,t,k,n,N,ww,kc,L):
    Chi=vector_field(xi,t,ki,n,N,ww,kc,L)
    Chi_hat=vector_field_CBF(xi,Chi,n,N,R,alpha)
    xi_eta = Chi_hat.reshape((N*(n+1),-1)).T
    return xi_eta[0]
  
#------------------------------------------------------------------------

#Parámetro de simulación 
n = 3           #dimensiones del espacio
N =2          #nº de agentes 
t = np.linspace(0,1,10) #tiempo de integración 
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


#Voy a resolver por Euler para ir parando y viendo los campos
h=t[1]-t[0]
sol=np.zeros((len(t)*(n+1)*N))
Chi=np.zeros((len(t)*(n+1)*N))
Chi_hat=np.zeros((len(t)*(n+1)*N))
sol[0:(n+1)*N]=vector_field_completo(Xi[0],t,ki,n,N,ww,kc,L)
valores = np.linspace(-4,4,100)
# La curva es la misma para todos los robots

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#nuestra curva 
[x,y,z]=fun(valores)
ax.plot(x,y,z,color="royalblue",linewidth = 2.5,zorder = 2)

intervalo=(n+1)*N
fin=(len(t)-1)*intervalo
for i in range(1,fin,intervalo):
    xi=sol[i-1:i-1+(n+1)*N].copy()
    Chi[i-1:i-1+(n+1)*N]=vector_field(xi,t,ki,n,N,ww,kc,L)
    Chi_hat[i-1:i-1+(n+1)*N]=vector_field_completo(xi,t,ki,n,N,ww,kc,L)
    sol[i-1+(n+1)*N:i-1+2*(n+1)*N]=xi+h*Chi_hat[i-1:i-1+(n+1)*N]
    pi=gu.unstack(sol[i-1+(n+1)*N:i-1+2*(n+1)*N],n+1)
    #Desapilo el campo
    Chi2=gu.unstack(Chi, n+1)
    Chi2_hat=gu.unstack(Chi_hat,n+1)
    for j in range(N):
        ax.scatter(pi[j,0],pi[j,1],pi[j,2],linewidth = 2.5, zorder = 3)

        #campo vectorial - gradiente 
        plt.quiver(pi[j,0],pi[j,1],pi[j,2],Chi2[j,0], Chi2[j,1], Chi2[j,2], length=1, color = "olive", zorder = 1)
        plt.quiver(pi[j,0],pi[j,1],pi[j,2],Chi2_hat[j,0], Chi2_hat[j,1], Chi2_hat[j,2], length=1, color = "red", zorder = 1)



''' Pinto para ver qué tal'''
'''
# Desapilo las posiciones
pi=gu.unstack(Xi[0],n+1)
#Desapilo el campo
Chi2=gu.unstack(Chi, n+1)
#TODO: revisar el dibujo
for i in range(N):
    ax.scatter(pi[i,0],pi[i,1],pi[i,2],linewidth = 2.5, zorder = 3)

    #campo vectorial - gradiente 
    plt.quiver(pi[i,0],pi[i,1],pi[i,2],Chi2[i,0], Chi2[i,1], Chi2[i,2], length=1, color = "olive", zorder = 1)
    plt.quiver(pi[i,0],pi[i,1],pi[i,2],Chi_hat[i,0], Chi_hat[i,1], Chi_hat[i,2], length=1, color = "red", zorder = 1)

'''  
#representación gráfica 


#resolvemos las ecuación diferencial 
t = np.linspace(0,100,10000) #tiempo de integración 
pos = np.random.randint(-50, 51, size=(N, n))
pos=np.array([[0,10,0],[-1.,10,0]])
#añadimos a la matriz de posiciones la coordenada virtual w 
w = np.ones((N,1)) #ejemplo: todos valen 1 

#concatenamos las dos matrices a lo largo de las filas 
Xi = np.concatenate((pos, w), axis=1)
Xi = Xi.reshape((N*(n+1),-1)).T #apilamos en un vector


sol2 = odeint(vector_field_completo,Xi[0],t,args=(ki,n,N,ww,kc,L))
sol1 = odeint(vector_field,Xi[0],t,args=(ki,n,N,ww,kc,L))
lista = np.arange(0,(n+1)*N+1,(n+1))

# Configuración de la figura
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z,color="royalblue",linewidth = 2.5,zorder = 2)


for i in lista[:-1]: 
    ax.plot(sol1[:,i], sol1[:,i+1], sol1[:,i+2], label='Trayectorias sin CBF')
    ax.scatter(sol1[0,i],sol1[0,i+1], sol1[0,i+2], marker='o')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Trayectoria de los agentes')



n_frames = 10000
# Línea y punto que se animarán
line1, = ax.plot([], [], [], 'b-', label='Trayectoria1')
point1, = ax.plot([], [], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], [], 'go', label='Posición actual2')
ax.legend()

def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    point1.set_data([], [])
    point1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    point2.set_data([], [])
    point2.set_3d_properties([])
    return line1,line2,point1,point2

def update(frame):
    line1.set_data(sol2[:frame+1, 0], sol2[:frame+1, 1])
    line1.set_3d_properties(sol2[:frame+1, 2])
    point1.set_data(sol2[frame, 0], sol2[frame, 1])
    point1.set_3d_properties(sol2[frame, 2])
    line2.set_data(sol2[:frame+1, 4], sol2[:frame+1, 5])
    line2.set_3d_properties(sol2[:frame+1, 6])
    point2.set_data(sol2[frame, 4], sol2[frame, 5])
    point2.set_3d_properties(sol2[frame, 6])
    return line1, point1,line2,point2

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=100)
plt.show()
















