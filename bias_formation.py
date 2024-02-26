#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:02:03 2024

@author: lia
"""

import RobotModels as rob
import numpy as np
import matplotlib.pyplot as plt
import graph_utils as graf
#import time
import robot_plot_utils as rpu

#TODO: Revisar. Creo que hay un problema con la coordenada y
# Posiciones iniciales de los agentes
#pi=np.array([[-5 ,0],[-5, 5],[5, 5],[5, 0]])
pi=np.array([[-5 ,0],[-5, 5],[5, 5]])
nagentes,dim=np.shape(pi)

# Grafo de conexiones
Z=np.array([[1,2],[2,3]])
narcos,m=np.shape(Z)

# Dibujo
#plt.plot(pi[:,0],pi[:,1],'mo')
plt.grid()

#Grafo
rpu.dibuja_grafo_2D(pi,Z)

B= graf.matriz_incidencia(Z, nagentes, narcos)
print(np.transpose(B))

# Posiciones relativas objetivo
zstar=np.array([[15, 0],[15, 0]])
zstar_s=np.zeros([narcos*2,1])
zstar_s=graf.stack(zstar)

dt=0.1
tini=0
tfin=50
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
    

I2=np.ones([dim,dim])

print(I2)
Bbar=np.kron(B,I2)
Bbar=np.transpose(Bbar)
z=np.zeros([narcos*m,1])
k=0
Im=np.ones([dim*nagentes,1])

Bias=np.zeros([nagentes,nagentes])
#Bias[0,1]=0.5
BiasBar=np.kron(Bias,I2)
BiasBar=np.transpose(BiasBar)
K=0.01
p=np.zeros([nagentes,2])
#Integrador
for k in range(1,nptos):
    l=0
    for i in range(narcos):
        head=Z[i,0]
        z[l,0]=posiciones[head,k-1]
        l+=1
        z[l,0]=posiciones[head+1,k-1]
        l+=1
       
    print(np.shape(z))         
    zdif=z-zstar_s
    print(np.shape(zdif)) 
    print(np.shape(Bbar))
    M1=np.dot(Bbar,zdif)
    print(np.shape(M1))
    pdot=-(K*M1+np.dot(BiasBar,Im))
    l=0
    for j in range(nagentes):
        xdot,ydot,thetadot=rob.integrator_point_model(pdot[l],pdot[l+1])
        p[j,0]+=dt*xdot
        p[j,1]+=dt*ydot
        l+=2
    p_s=graf.stack(p)
    print(np.shape(p_s))
    posiciones[:,k]=p_s[:,0]
    t+=dt

    
colors_f=['bo','go','ro','co','yo','mo','ko','bo','go']
colors_i=['bx','gx','rx','cx','yx','mx','kx','bx','gx']
colors_l=['b:','g:','r:','c:','y:','m','k','b:','g:']

j=0
pf=np.zeros([nagentes,2])
for i in range(nagentes):
    plt.plot(posiciones[j,0],posiciones[j+1,0],colors_i[i])
    plt.plot(posiciones[j,nptos-1],posiciones[j+1,nptos-1],colors_f[i])
    plt.plot(posiciones[j,:],posiciones[j+1,:],colors_l[i])
    pf[i,0]=posiciones[j,nptos-1]
    pf[i,1]=posiciones[j,nptos-1]
    j=j+2
print(pf)
rpu.dibuja_grafo_2D(pf,Z)
plt.grid()