#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:08:19 2025

@author: lia

GVF +CBF segun la tesis de Weija para evitar choques entre robots
Version inicial para 2 agentes que se comunican todos entre sí en 2D

"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import graph_utils as gu
from ldl import inverse_using_ldlt, ldlt_decomposition
#CASO DE N ROBOTS EN UN ESPACIO n-DIMENSIONAL

#todos los agentes siguen las mismas ecuaciones de trayectoria

np.random.seed(123)

#Definimos en una función las ecuaciones de la trayectoria 

# def fun(w):
#     f1 = 15*np.sin(2*w)
#     f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
#     f3 = 3+5*np.cos(2*w)
#     return np.array([f1, f2, f3])

def fun(w):
    f1 = 15*np.sin(2*w)
    f2 = 30*np.sin(w)*(np.sqrt(0.5*(1-0.5*(np.sin(w))**2)))
    
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



def vector_field_CBF_analitico(t,xi,n,N,R,alpha,vecinos,k,ww,kc,L):
    #xi: posiciones de todos los agentes (vector columna con las 4 coordenadas )
    #eta: campo guiado para todos los agentes
    #k: ganancias positivas
    #n: dimensiones 
    #N: nº de agentes 
    #R: distancia de seguridad entre los robots
    #alpha: ganancia de la barrera
    #vecinos: matriz con los indices de los vecinos de cada robot
    # Desapilo el campo (cuidado que lleva la coordenada ampliada)
    Chi_ap=vector_field(t,xi,k,n,N,ww,kc,L)
    Chi=gu.unstack(Chi_ap,n+1)
    # Desapilo posiciones
    pi=gu.unstack(xi,n+1)
    Chi_cbf=np.zeros((N,n+1))
    #Construyo las matrices
    #Construyo las matrices
    M=np.eye(n+1,n+1)
    Pr=np.zeros((n+1,n+1))
    Pr[-1,-1]=1
    P=M-Pr
    #Construyo la función de seguridad para la barrera
    eta=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            eta[i,j]=np.linalg.norm(P@(pi[i,:]-pi[j,:]))**2-R**2
    
    for i in range(N):
        A=np.zeros((N-1,n+1))
        b=np.zeros(N-1)
        s=0
        nvecinos=len(vecinos[i,:])
        for k in range(nvecinos):
            v=vecinos[i,k]
            A[s,:]=(pi[v]-pi[i])@P.T
            b[s]=alpha*eta[i,v]**3/4.0
            s+=1
        #[1] Veo si el óptimo global verifica las restricciones
        m=A@Chi[i,:]
        Aa = np.empty([0,3])
        ba = np.empty([0])
        for j in range(nvecinos):
            if m[j]>b[j]:
                bb=[b[j]]
                aa=[A[j]]
                Aa=np.append(Aa,aa,axis=0)
                ba=np.append(ba,bb)
        #Aai=(Aa@Aa.T)**-1
        L,D=ldlt_decomposition(Aa@Aa.T)
        Aai=inverse_using_ldlt(L,D)
        lA=Aai@(Aa@Chi[i,:]-ba)
        Chi_cbf[i,:]=Chi[i,:]-Aa.T@lA
        print(i,Chi_cbf[i,:])
    Chi_cbf_ap=Chi_cbf.reshape((N*(n+1),-1)).T
    #print("Chi_cbf",Chi_cbf_ap)
    #print("Chi_pf",Chi_ap)
    return Chi_cbf_ap
    
 
 
def vector_field_completo(t,xi,k,n,N,ww,kc,L):
    alpha=0.1
    R=2
    #TODO una lista con los vecinos de cada robot
    vecinos=np.zeros((N,N-1))
    vecinos=np.array([[1,2,3,4],[0,2,3,4],[0,1,3,4],[0,1,2,4],[0,1,2,3]])
    Chi_hat=vector_field_CBF_analitico(t,xi,n,N,R,alpha,vecinos,ki,ww,kc,L)
    xi_eta = Chi_hat.reshape((N*(n+1),-1)).T
    return xi_eta[0]
  
def barrera(R,p):
    x=np.linspace(p[0]-R, p[0]+R,20)
    ymas=p[1]+np.sqrt(R**2-(x-p[0])**2)
    ymenos=p[1]-np.sqrt(R**2-(x-p[0])**2)
    return x,ymas,ymenos
#------------------------------------------------------------------------
#------------------------------------------------------------------------

#Parámetro de simulación 
n = 2           #dimensiones del espacio
N = 5          #nº de agentes 
t = np.linspace(0,10,100) #tiempo de integración 
ki =[1,1] #ganancias 

#coordenadas iniciales de los robots 
#pos = np.random.rand(N, n)*100 #filas: dimensiones
                           #columnas: nº de robots
pos = np.random.randint(-20, 21, size=(N, n))

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


#concatenamos las dos matrices a lo largo de las filas 
Xi = np.concatenate((pos, w), axis=1)
Xi = Xi.reshape((N*(n+1),-1)).T #apilamos en un vector


#sol2 = odeint(vector_field_completo,Xi[0],t,args=(ki,n,N,ww,kc,L))
sol2 = odeint(vector_field_completo,Xi[0],t,args=(ki,n,N,ww,kc,L))
lista = np.arange(0,(n+1)*N+1,(n+1))

# Configuración de la figura

valores = np.linspace(-4,4,100)
[x,y]=fun(valores)



# Línea y punto que se animarán
plt.ion()  # Modo interactivo
fig, ax = plt.subplots()
ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
line1, = ax.plot([], [], 'r.-')  # línea con puntos rojos
line2, = ax.plot([], [], 'b.-')  # línea con puntos rojos
line3, = ax.plot([], [], 'g.-')  # línea con puntos verdes

ax.set_xlim(-35,25)
ax.set_ylim(-35, 25)
x1=sol2[:,0]
y1=sol2[:,1]
x2=sol2[:,3]
y2=sol2[:,4]
x3=sol2[:,6]
y3=sol2[:,7]


# Pintar punto a punto
for i in range(len(x1)):
    line1.set_xdata(x1[:i+1])
    line1.set_ydata(y1[:i+1])
    line2.set_xdata(x2[:i+1])
    line2.set_ydata(y2[:i+1])
    line3.set_xdata(x3[:i+1])
    line3.set_ydata(y3[:i+1])
  
    xb,ybM,ybm=barrera(R, [x1[i],y1[i]])
    ax.plot(xb,ybM,'m-')
    ax.plot(xb,ybm,'m-')
    plt.draw()
    plt.pause(0.1)  # pausa entre fotogramas (0.5 segundos)

plt.ioff()  # Apaga el modo interactivo
plt.show()














