#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:02:03 2024

@author: lia
two robots
one has bias in the relative position sensing
"""

import RobotModels as rob
import numpy as np
import matplotlib.pyplot as plt
import graph_utils as graf
import robot_plot_utils as rpu


# Posiciones iniciales de los agentes
pi=np.array([[0,0],[0,5],[5, 0]],dtype=np.float64)
nagentes,dim=np.shape(pi)

# Grafo de conexiones
Z=np.array([[1,2],[1,3]])
#Array duplicado. Para cada arco pongo dos (ida y vuelta)
Zd=np.array([[1,2],[2,1],[1,3],[3,1]])
narcos,m=np.shape(Z)

# Dibujo posiciones iniciales con grafo
plt.plot(pi[:,0],pi[:,1],'mo')
plt.grid()

#Grafo
rpu.dibuja_grafo_2D(pi,Z)


# Posiciones relativas objetivo
zstar=np.array([[-5, 0],[5,0],[0,5],[-5, 0]])
zstar_s=np.zeros([narcos*2*dim,1])
zstar_s=graf.stack(zstar)

dt=0.01
tini=0
tfin=1
t=tini
nptos=(int)((tfin-tini)/dt)

#Array para guardar posiciones
posiciones=np.zeros([2*nagentes , nptos+1])
j=0
for i in range(nagentes):
    posiciones[j,0]=pi[i,0]
    j+=1
    posiciones[j,0]=pi[i,1]
    j+=1
    
z=np.zeros([narcos*m*2,1])
k=0
K=0.5
p=np.zeros([nagentes,2],dtype=np.float64)
p=pi
posiciones_us=np.zeros([nagentes,nptos+1])
pos_s=np.zeros([nagentes,2])
pos_us=np.zeros([2*nagentes,1])

pdot=np.zeros(6)

#Matriz bias con los errores de medida
# Bias nagentesxncaminos*2
Bias=np.zeros([nagentes,narcos*2*dim]) 
Bias[0,0]=2 #El robot 1 mete un error de 2 en la medida de la distancia del 1 al 2
#Integrador
#TODO: Revisar la ejecución paso a paso para ver si tiene sentido y esta bien programado
#TODO: Tratar de formalizar esto de alguna manera
#TODO: ¿Podemos conocer por cómo se deforma qué agente tiene la culpa? ¿Podemos conocer de alguna manera por cómo se deforma la formación qué está ocurriendo?
for k in range(1,nptos):
    l=0
    nc=0
    pos_us[:,0]=posiciones[:,k-1]
    posiciones_us=graf.unstack(pos_us, 2)
    for i in range(narcos):
        head=Z[i,0]
        tail=Z[i,1]
        z[l,0]=posiciones_us[head-1,0]-posiciones_us[tail-1,0]+Bias[head-1,nc]
        l+=1
        z[l,0]=posiciones_us[head-1,1]-posiciones_us[tail-1,1]+Bias[head-1,nc+1]
        l+=1
        nc+=1
        z[l,0]=-z[l-2,0]+Bias[tail-1,nc]
        l+=1
        z[l,0]=-z[l-2,0]+Bias[tail-1,nc+1]
        nc+=2
        l+=1
        
    zdif=z-zstar_s
    print("zdif")
    print(zdif)
    for i in range(nagentes*2):
        pdot[i]=-K*zdif[i,0]
    
    # pdot[0]=-K*(zdif[0,0])
    # pdot[1]=-K*(zdif[1,0])
    # pdot[2]=-K*zdif[2,0]
    # pdot[3]=-K*zdif[3,0]
    l=0
    
    for j in range(nagentes):
        xdot,ydot,thetadot=rob.integrator_point_model(pdot[l],pdot[l+1])
        p[j,0]+=(float(dt*xdot))
        p[j,1]+=dt*ydot
        l+=2
    
    p_s=graf.stack(p)
    posiciones[:,k]=p_s[:,0]
    t+=dt
print("pstack")
print(p_s)
    
colors_f=['bo','go','ro','co','yo','mo','ko','bo','go']
colors_i=['bx','gx','rx','cx','yx','mx','kx','bx','gx']
colors_l=['b:','g:','r:','c:','y:','m','k','b:','g:']

j=0
pf=np.zeros([nagentes,2])
#Dibujo final
plt.figure()
plt.grid()
for i in range(nagentes):
    plt.plot(posiciones[j,0],posiciones[j+1,0],colors_i[i])
    plt.plot(posiciones[j,nptos-1],posiciones[j+1,nptos-1],colors_f[i])
    plt.plot(posiciones[j,0:nptos-1],posiciones[j+1,0:nptos-1],colors_l[i])
    pf[i,0]=posiciones[j,nptos-1]
    pf[i,1]=posiciones[j+1,nptos-1]
    j=j+2
rpu.dibuja_grafo_2D(pf,Z)


print("zfinal")
print(z)

print("zdiff_medido")
print(zdif)