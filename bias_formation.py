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

# Posiciones iniciales de los agentes
pi=np.array([[-5 ,0],[-5, 5],[5, 5],[5, 0]])

nagentes,dim=np.shape(pi)

# Grafo de conexiones
Z=np.array([[1,2],[2,3],[1,4]])
narcos,m=np.shape(Z)

# Dibujo
plt.plot(pi[:,0],pi[:,1],'mo')
plt.grid()

#Grafo
rpu.dibuja_grafo_2D(pi,Z)

B= graf.matriz_incidencia(Z, nagentes, narcos)
print(np.transpose(B))

# Posiciones relativas objetivo
zstar=np.array([[15, 0],[15, 0],[15,0]])
zstar_s=np.zeros([narcos*2,1])
zstar_s=graf.stack(zstar)

dt=0.01
tini=0
tfin=100
t=tini
nptos=(int)((tfin-tini)/dt)

#Array para guardar posiciones
posiciones=np.zeros([2*nagentes , nptos])
j=0
for i in range(nagentes):
    posiciones[j,0]=pi[i,0]
    j+=1
    posiciones[j,0]=pi[i,1]
    j+=1
    

I2=np.ones([dim,1])
#TODO Revisar todas las dimensiones

Bbar=np.kron(B,I2)
Bbar=np.transpose(Bbar)
z=np.zeros([narcos*m,1])
k=0
Im=np.ones([dim*nagentes,1])

Bias=np.zeros([nagentes,nagentes])
Bias[0,1]=0.5
BiasBar=np.kron(Bias,I2)
BiasBar=np.transpose(BiasBar)
#Integrador
while t<tfin:
    p=0
    for i in range(narcos):
        head=Z[i,0]
        z[p,0]=posiciones[head,k-1]
        p+=1
        z[p,0]=posiciones[head+1,k-1]
        p+=1
              
    zdif=z-zstar_s
    M1=np.dot(Bbar,zdif)
    print(M1)
    pdot=-np.dot(Bbar,zdif)+np.dot(Im,BiasBar)
    l=0
    for j in range(nagentes):
        xdot,ydot,thetadot=rob.integrator_point_model(pdot[l],pdot[l+1])
        p[j,0]+=dt*xdot
        p[j,1]+=dt*ydot
        l+=2
    p_s=graf.stack(p)
    posiciones[:,k+1]=p_s[:]
    t+=dt
    k+=1
    
colors_f=['bo','go','ro','co','yo','mo','ko','bo','go']
colors_i=['bx','gx','rx','cx','yx','mx','kx','bx','gx']
colors_l=['b','g','r','c','y','m','k','b:','g:']

j=0
for i in range(nagentes):
    plt.plot(posiciones[j,0],posiciones[j+1,0],colors_i[i])
    plt.plot(posiciones[j,nptos-1],posiciones[j+1,nptos-1],colors_f[i])
    plt.plot(posiciones[j,:],posiciones[j+1,:],colors_l[i])
    j=j+2
plt.grid()