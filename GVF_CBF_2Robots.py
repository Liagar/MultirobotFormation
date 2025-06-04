 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:08:19 2025

@author: lia

GVF +CBF segun la tesis de Weija para evitar choques entre robots
Version inicial para 2agentes que se comunican todos entre sí en 2D

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import qp as qp
import graph_utils as gu
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation

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

def vector_field(t,xi,k,n,N,ww,kc,L):
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

   
#Solucion analítica del problema

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
    dx=pi[0,0]-pi[1,0]
    dy=pi[0,1]-pi[1,1]
    
    #Robot 0
    eta=dx*dx+dy*dy-R**2
    
    A=np.zeros(2)
    for i in range(2):
        if eta <0:
            if i==0:
                A[0]=dx
                A[1]=dy
            else:
                A[0]=-dx
                A[1]=-dy
            b=alpha*eta**3/4.0
            m=A[0]*Chi[i,0]+A[1]*Chi[i,1]
            if m>b:
                nA=A[0]**2+A[1]**2
                la=(A[0]*Chi[i,0]+A[1]*Chi[i,1]-b)/nA
                Chi_cbf[i,0]=Chi[i,0]-A[0]*la
                Chi_cbf[i,1]=Chi[i,1]-A[1]*la
            else:
                Chi_cbf[i,:]=Chi[i,:]
    Chi_cbf_ap=Chi_cbf.reshape((N*(n+1),-1)).T
    #print("Chi_cbf",Chi_cbf_ap)
    #print("Chi_pf",Chi_ap)
    return Chi_cbf_ap

# CBF solucion numerica
   
 
def vector_field_completo(t,xi,k,n,N,ww,kc,L):
    alpha=0.1
    R=2
    #TODO una lista con los vecinos de cada robot
    vecinos=np.zeros((N,N-1))
    vecinos=np.array([[1],[0]])
    Chi_hat=vector_field_CBF_analitico(t,xi,n,N,R,alpha,vecinos,ki,ww,kc,L)
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

pos=np.array([[-15.0,0],[-16,0]])
#pos=np.array([[-15.0,0],[-10,-10],[10,10]])
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
    

#representación gráfica 


#resolvemos las ecuación diferencial 
t = np.linspace(0,5,100) #tiempo de integración 
tspan=[0,10]
h=t[1]-t[0]
nf=len(t)
#añadimos a la matriz de posiciones la coordenada virtual w 
w = np.ones((N,1)) #ejemplo: todos valen 1 

#concatenamos las dos matrices a lo largo de las filas 
Xi = np.concatenate((pos, w), axis=1)
Xi = Xi.reshape((N*(n+1),-1)).T #apilamos en un vector



sol1 = solve_ivp(vector_field,tspan,Xi[0],args=(ki,n,N,ww,kc,L))
sol2 = solve_ivp(vector_field_completo,tspan,Xi[0],args=(ki,n,N,ww,kc,L))

plt.figure()
plt.title("Solución analítica")
plt.plot(sol2.y[0,:],sol2.y[1,:],'r-',label='A1')
plt.plot(sol2.y[3,:],sol2.y[4,:],'b-',label='A2')

plt.plot(sol1.y[0,:],sol1.y[1,:],'m--',label='A1 sin CBF')
plt.plot(sol1.y[3,:],sol1.y[4,:],'c--',label='A2 sin CBF')



lista = np.arange(0,(n+1)*N+1,(n+1))
'''
plt.figure()
plt.subplot(3,1,1)
plt.plot(sol1.t,sol1.y[0,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[0,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[0,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada x. Agente 1")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("x(m)")

plt.subplot(3,1,2)
plt.plot(sol1.t,sol1.y[3,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[3,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[3,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada x. Agente 2")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("x(m)")

plt.subplot(3,1,3)
plt.plot(sol1.t,sol1.y[6,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[6,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[6,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada x. Agente 3")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("x(m)")

plt.figure()
plt.subplot(3,1,1)
plt.plot(sol1.t,sol1.y[1,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[1,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[1,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada y. Agente 1")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("y(m)")

plt.subplot(3,1,2)
plt.plot(sol1.t,sol1.y[4,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[4,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[4,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada y. Agente 2")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("y(m)")

plt.subplot(3,1,3)
plt.plot(sol1.t,sol1.y[7,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[7,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[7,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada y. Agente 3")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("y(m)")

plt.figure()
plt.subplot(3,1,1)
plt.plot(sol1.t,sol1.y[2,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[2,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[2,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada virtual. Agente 1")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("omega")

plt.subplot(3,1,2)
plt.plot(sol1.t,sol1.y[5,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[5,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[5,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada virtual. Agente 2")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("omega")

plt.subplot(3,1,3)
plt.plot(sol1.t,sol1.y[8,:],'r.',label='Sin CBF') 
plt.plot(sol2.t,sol2.y[8,:],'b', label='CBF analítico') 
plt.plot(sol3.t, sol3.y[8,:],'g',label='CBF numérico')
plt.title("Evolucion de la coordenada virtual. Agente 3")
plt.legend()
plt.xlabel("t(s)")
plt.ylabel("omega")
'''

'''
#Animacion 3D
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
valores = np.linspace(-4,4,100)
[x,y]=fun(valores)
#ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)
n_frames = len(sol2.t)-2
# Línea y punto que se animarán
line1, = ax.plot([], [], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], [], 'mo', label='Posición actual2')
line3, = ax.plot([], [], [], 'g-', label='Trayectoria3')
point3, = ax.plot([], [], [], 'go', label='Posición actual3')

ax.set_title("Solucion analítica")
#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    line3.set_data([], [])
    point3.set_data([], [])
    
  
    return line1,line2,line3,point1,point2,point3

def update(frame):
    line1.set_data(sol2.y[0,:frame+1], sol2.y[1,:frame+1])
    line1.set_3d_properties(sol2.y[2,:frame+1])
    point1.set_data(sol2.y[0,frame], sol2.y[1,frame])
    point1.set_3d_properties(sol2.y[2,frame])
    line2.set_data(sol2.y[3,:frame+1], sol2.y[4,:frame+1])
    line2.set_3d_properties(sol2.y[5,:frame+1])
    point2.set_data(sol2.y[3,frame], sol2.y[4,frame])
    point2.set_3d_properties(sol2.y[5,frame])
    line3.set_data(sol2.y[6,:frame+1], sol2.y[7,:frame+1])
    line3.set_3d_properties(sol2.y[8,:frame+1])
    point3.set_data(sol2.y[6,frame], sol2.y[7,frame])
    point3.set_3d_properties(sol2.y[8,frame])
    return line1, point1,line2,point2,line3,point3

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
valores = np.linspace(-4,4,100)
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_title('Trayectoria de los agentes. Solucion numérica')



fig, ax = plt.subplots()
ax = fig.add_subplot(111,projection='3d')
valores = np.linspace(-4,4,100)
[x,y]=fun(valores)
#ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)
n_frames = len(sol3.t)-2
# Línea y punto que se animarán
line1, = ax.plot([], [], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], [], 'mo', label='Posición actual2')
line3, = ax.plot([], [], [], 'g-', label='Trayectoria3')
point3, = ax.plot([], [], [], 'go', label='Posición actual3')

#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    line3.set_data([], [])
    point3.set_data([], [])
    
  
    return line1,line2,line3,point1,point2,point3

def update(frame):
    line1.set_data(sol3.y[0,:frame+1], sol3.y[1,:frame+1])
    line1.set_3d_properties(sol3.y[2,:frame+1]) 
    point1.set_data(sol3.y[0,frame], sol3.y[1,frame])
    point1.set_3d_properties(sol3.y[2,frame]) 
    line2.set_data(sol3.y[3,:frame+1], sol3.y[4,:frame+1])
    line2.set_3d_properties(sol3.y[5,:frame+1]) 
    point2.set_data(sol3.y[3,frame], sol3.y[4,frame])
    point2.set_3d_properties(sol3.y[5,frame]) 
    line3.set_data(sol3.y[6,:frame+1], sol3.y[7,:frame+1])
    line3.set_3d_properties(sol3.y[8,:frame+1]) 
    point3.set_data(sol3.y[6,frame], sol3.y[7,frame])
    point3.set_3d_properties(sol3.y[8,frame]) 
    
    return line1, point1,line2,point2,line3,point3

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots_Num.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")




fig, ax = plt.subplots()
ax = fig.add_subplot(111,projection='3d')
valores = np.linspace(-4,4,100)
#ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(-25, 25)
n_frames = len(sol3.t)-2
ax.set_title('Trayectoria de los agentes. Solucion sin CBF')
# Línea y punto que se animarán
line1, = ax.plot([], [], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], [], 'mo', label='Posición actual2')
line3, = ax.plot([], [], [], 'g-', label='Trayectoria3')
point3, = ax.plot([], [], [], 'go', label='Posición actual3')

#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    line3.set_data([], [])
    point3.set_data([], [])
    
  
    return line1,line2,line3,point1,point2,point3

def update(frame):
    line1.set_data(sol1.y[0,:frame+1], sol1.y[1,:frame+1])
    line1.set_3d_properties(sol1.y[2,:frame+1]) 
    point1.set_data(sol1.y[0,frame], sol1.y[1,frame])
    point1.set_3d_properties(sol1.y[2,frame]) 
    line2.set_data(sol1.y[3,:frame+1], sol1.y[4,:frame+1])
    line2.set_3d_properties(sol1.y[5,:frame+1]) 
    point2.set_data(sol1.y[3,frame], sol1.y[4,frame])
    point2.set_3d_properties(sol1.y[5,frame]) 
    line3.set_data(sol1.y[6,:frame+1], sol1.y[7,:frame+1])
    line3.set_3d_properties(sol1.y[8,:frame+1]) 
    point3.set_data(sol1.y[6,frame], sol1.y[7,frame])
    point3.set_3d_properties(sol1.y[8,frame]) 
    
    return line1, point1,line2,point2,line3,point3

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots_sinCBF.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")
'''

#2D
fig, ax = plt.subplots()
ax = fig.add_subplot(111)
valores = np.linspace(-4,4,100)
[x,y]=fun(valores)
ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
n_frames = len(sol2.t)-2
# Línea y punto que se animarán
line1, = ax.plot([], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], 'mo', label='Posición actual2')

ax.set_title("Solucion analítica")
#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    
  
    return line1,line2,point1,point2

def update(frame):
    line1.set_data(sol2.y[0,:frame+1], sol2.y[1,:frame+1])
    point1.set_data(sol2.y[0,frame], sol2.y[1,frame])
    line2.set_data(sol2.y[3,:frame+1], sol2.y[4,:frame+1])
    point2.set_data(sol2.y[3,frame], sol2.y[4,frame])
    return line1, point1,line2,point2

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots_2.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")

'''
ax.set_title('Trayectoria de los agentes. Solucion numérica')



fig, ax = plt.subplots()
ax = fig.add_subplot(111)
valores = np.linspace(-4,4,100)
[x,y]=fun(valores)
ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)

n_frames = len(sol3.t)-2
# Línea y punto que se animarán
line1, = ax.plot([], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [], 'm-', label='Trayectoria2')
point2, = ax.plot([], [], 'mo', label='Posición actual2')
line3, = ax.plot([], [],'g-', label='Trayectoria3')
point3, = ax.plot([], [], 'go', label='Posición actual3')

#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    line3.set_data([], [])
    point3.set_data([], [])
    
  
    return line1,line2,line3,point1,point2,point3

def update(frame):
    line1.set_data(sol3.y[0,:frame+1], sol3.y[1,:frame+1])
    point1.set_data(sol3.y[0,frame], sol3.y[1,frame])
    line2.set_data(sol3.y[3,:frame+1], sol3.y[4,:frame+1])
    point2.set_data(sol3.y[3,frame], sol3.y[4,frame])
    line3.set_data(sol3.y[6,:frame+1], sol3.y[7,:frame+1])
    point3.set_data(sol3.y[6,frame], sol3.y[7,frame])
    
    return line1, point1,line2,point2,line3,point3

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots_Num.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")




fig, ax = plt.subplots()
ax = fig.add_subplot(111)
valores = np.linspace(-4,4,100)
ax.plot(x,y,color="royalblue",linewidth = 2.5,zorder = 2)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
n_frames = len(sol3.t)-2
ax.set_title('Trayectoria de los agentes. Solucion sin CBF')
# Línea y punto que se animarán
line1, = ax.plot([], [], 'r-', label='Trayectoria1')
point1, = ax.plot([], [], 'ro', label='Posición actual1')
line2, = ax.plot([], [],  'm-', label='Trayectoria2')
point2, = ax.plot([], [], 'mo', label='Posición actual2')
line3, = ax.plot([], [], 'g-', label='Trayectoria3')
point3, = ax.plot([], [], 'go', label='Posición actual3')

#ax.legend(loc='upper right')

def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    line3.set_data([], [])
    point3.set_data([], [])
    
  
    return line1,line2,line3,point1,point2,point3

def update(frame):
    line1.set_data(sol1.y[0,:frame+1], sol1.y[1,:frame+1])
    point1.set_data(sol1.y[0,frame], sol1.y[1,frame])
    line2.set_data(sol1.y[3,:frame+1], sol1.y[4,:frame+1])
    point2.set_data(sol1.y[3,frame], sol1.y[4,frame])
    line3.set_data(sol1.y[6,:frame+1], sol1.y[7,:frame+1])
    point3.set_data(sol1.y[6,frame], sol1.y[7,frame])
    
    return line1, point1,line2,point2,line3,point3

ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=10)
# Guardar animación como GIF
#ani.save("animacion.gif", writer=PillowWriter(fps=10))
#plt.show()

# Guardar la animación como un video MP4
output_file = "trayectoria_robots_sinCBF.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_file, writer=writer)

print(f"Animación guardada como {output_file}")

'''




















