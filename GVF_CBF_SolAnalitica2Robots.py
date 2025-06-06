#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:08:19 2025

@author: lia

GVF +CBF segun la tesis de Weija para evitar choques entre robots
Version inicial para 2 agentes que se comunican todos entre sí en 2D

"""
# TODO: Revisar por qué no funciona la solución analítica ni siquiera GVF normal

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import qp as qp
import graph_utils as gu
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

   
#CASO DE 2 ROBOTS EN UN ESPACIO 2-DIMENSIONAL

#todos los agentes siguen las mismas ecuaciones de trayectoria

np.random.seed(123)

#Definimos en una función las ecuaciones de la trayectoria 

def fun(w):
    f1 = 15*np.sin(2*w)
    f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    f3 = 3+5*np.cos(2*w)
    return np.array([f1, f2])

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



def vector_field_CBF(xi,Chi_ap,n,N,R,alpha):
    #xi: posiciones de todos los agentes (vector columna con las 4 coordenadas )
    #eta: campo guiado para todos los agentes
    #k: ganancias positivas
    #n: dimensiones 
    #N: nº de agentes 
    #R: distancia de seguridad entre los robots
    #alpha: ganancia de la barrera
    # Desapilo el campo (cuidado que lleva la coordenada ampliada)
    Chi=gu.unstack(Chi_ap,n+1)
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
    eta=np.zeros((N+1,N+1))
    for i in range(N):
        for j in range(N):
            eta[i,j]=np.linalg.norm(P@(pi[i,:]-pi[j,:]))**2-R**2
            
    dx=pi[0,0]-pi[1,0]
    dy=pi[0,1]-pi[1,1]
    d2=(dx)**2+(dy)**2
    lam=0
    lam2=0
    if(d2<R**2):
        if (-alpha*eta[0,1]**3+(dx)*Chi[0,0]+dy*Chi[0,1])>0:
            lam=(-alpha*eta[0,1]**3+(dx)*Chi[0,0]+dy*Chi[0,1])/(4*d2)
        else:
            lam=0
        if ((-alpha*eta[1,0]**3-(dx)*Chi[1,0]-dy*Chi[1,1])>0):
            lam2=(-alpha*eta[1,0]**3-(dx)*Chi[1,0]-dy*Chi[1,1])/(4*d2)
        else: 
            lam2=0

    Chi_cbf[0,0]=Chi[0,0]+lam*dx
    Chi_cbf[0,1]=Chi[0,1]+lam*dy
    Chi_cbf[0,2]=Chi[0,2]
    Chi_cbf[1,0]=Chi[1,0]-lam2*dx
    Chi_cbf[1,1]=Chi[1,1]-lam2*dy
    Chi_cbf[1,2]=Chi[1,2]
    
        #Chi_cbf[i,:]=qp.qp_solve(M,-Chi[i,:],G=A,h=b,A=None,b=None,lb=None,ub=None)
    return Chi_cbf
    
 
def vector_field_completo(xi,t,k,n,N,ww,kc,L):
    alpha=1
    R=2
    Chi=vector_field(xi,t,ki,n,N,ww,kc,L)
    Chi_hat=vector_field_CBF(xi,Chi,n,N,R,alpha)
    xi_eta = Chi_hat.reshape((N*(n+1),-1)).T
    return xi_eta[0]
  
#------------------------------------------------------------------------

#Parámetro de simulación 
n = 2           #dimensiones del espacio
N =2          #nº de agentes 
ki =[1,1] #ganancias 

#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
                           #columnas: nº de robots

pos=np.array([[-10.0,-10],[-11.,-12]])

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

#representación gráfica 


#resolvemos las ecuación diferencial 
t = np.linspace(0,20,1000) #tiempo de integración 

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
ax = fig.add_subplot(111)
valores = np.linspace(-4,4,100)
[x,y]=fun(valores)
ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)


for i in lista[:-1]: 
    ax.plot(sol1[:,i], sol1[:,i+1], 'k-',label='Trayectorias sin CBF')
    ax.scatter(sol1[0,i],sol1[0,i+1],marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Trayectoria de los agentes')



n_frames = 1000
# Línea y punto que se animarán
line1, = ax.plot([], [], 'b-', label='Trayectoria1')
point1, = ax.plot([], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], 'go', label='Posición actual2')

#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
  
    return line1,line2,point1,point2

def update(frame):
    line1.set_data(sol2[:frame+1, 0], sol2[:frame+1, 1])
    point1.set_data(sol2[frame, 0], sol2[frame, 1])
    line2.set_data(sol2[:frame+1, 3], sol2[:frame+1, 4])
    point2.set_data(sol2[frame, 3], sol2[frame, 4])
    return line1, point1,line2,point2

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=50)
# Guardar animación como GIF
ani.save("animacion.gif", writer=PillowWriter(fps=10))
plt.show()
















