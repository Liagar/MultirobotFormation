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

#TODO: Revisar muy bien las matrices y dimensiones. 
# Posiciones iniciales de los agentes
#pi=np.array([[-5 ,0],[-5, 5],[5, 5],[5, 0]])
pi=np.array([[0,0],[5, 0]],dtype=np.float64)
nagentes,dim=np.shape(pi)

# Grafo de conexiones
#TODO: Ver, al ser no dirigido necesito duplicarlo
Z=np.array([[1,2]])
#Array duplicado. Para cada arco pongo dos (ida y vuelta)
Zd=np.array([[1,2],[2,1]])
narcos,m=np.shape(Z)

# Dibujo
#plt.plot(pi[:,0],pi[:,1],'mo')
plt.grid()

#Grafo
rpu.dibuja_grafo_2D(pi,Z)

#B= graf.matriz_incidencia(Z, nagentes, narcos)
B= graf.matriz_incidencia(Zd, nagentes, 2*narcos)
print(np.transpose(B))

# Posiciones relativas objetivo
zstar=np.array([[-5, 0],[5, 0]])
zstar_s=np.zeros([narcos*4,1])
zstar_s=graf.stack(zstar)

dt=0.01
tini=0
tfin=5
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

#print(I2)
Bbar=np.kron(B,I2)
Bbar=np.transpose(Bbar)
z=np.zeros([narcos*m*2,1])
k=0
Im=np.ones([dim*nagentes,1])

Bias=np.array([2,0,0,0]) #El robot 1 mete un error de 0.5 
BiasBar=np.kron(Bias,I2)
BiasBar=np.transpose(BiasBar)
#print(BiasBar)
K=0.5
p=np.zeros([nagentes,2],dtype=np.float64)
p=pi
#TODO Revisar la matriz de incidencia M
M=np.array([[-1,1],[1,-1]])
Mbar=np.kron(M,I2)
#print(Mbar)
posiciones_us=np.zeros([nagentes,nptos+1])
pos_s=np.zeros([nagentes,2])
pos_us=np.zeros([2*nagentes,1])

pdot=np.zeros(4)
#Integrador
for k in range(1,nptos):
    l=0
    pos_us[:,0]=posiciones[:,k-1]
    posiciones_us=graf.unstack(pos_us, 2)
    #print("posiciones_us")
    #print(posiciones_us)
    for i in range(narcos):
        head=Z[i,0]
        tail=Z[i,1]
        z[l,0]=posiciones_us[head-1,0]-posiciones_us[tail-1,0]
        l+=1
        z[l,0]=posiciones_us[head-1,1]-posiciones_us[tail-1,1]
        l+=1
        z[l,0]=-z[l-2,0]
        l+=1
        z[l,0]=-z[l-2,0]
        l+=1
       
    #print(np.shape(z))     
    zdif=z-zstar_s + np.atleast_2d(Bias).T
    print("zdif")
    print(zdif)
    #print(np.shape(Bbar))
    #M1=np.dot(Bbar,zdif)
    #print(np.shape(M1))
    #pdot=-K*(M1+np.dot(BiasBar,Im))
    A=np.transpose((Mbar))
    #print(A)
    #M1=np.dot()
    #print("M1")
    #print(M1)
    #pdot=-K*np.dot(np.transpose(Mbar),zdif)
    pdot[0]=-K*(zdif[0,0]+1)
    pdot[1]=-K*(zdif[1,0]+1)
    pdot[2]=-K*zdif[2,0]
    pdot[3]=-K*zdif[3,0]
    print("pdot")
    print(pdot)
    l=0
    
    for j in range(nagentes):
        xdot,ydot,thetadot=rob.integrator_point_model(pdot[l],pdot[l+1])
        p[j,0]+=(float(dt*xdot))
        p[j,1]+=dt*ydot
        l+=2
        #print("p")
        #print(p)

    p_s=graf.stack(p)
    #print(np.shape(p_s))
    posiciones[:,k]=p_s[:,0]
    t+=dt
print("pstack")
print(p_s)
    
colors_f=['bo','go','ro','co','yo','mo','ko','bo','go']
colors_i=['bx','gx','rx','cx','yx','mx','kx','bx','gx']
colors_l=['b:','g:','r:','c:','y:','m','k','b:','g:']

j=0
pf=np.zeros([nagentes,2])
error=np.zeros([4])
error[0]=posiciones[0,nptos-1]-posiciones[2,nptos-1]
error[1]=posiciones[1,nptos-1]-posiciones[3,nptos-1]
error[2]=-error[0]
error[3]=-error[1]

#TODO: Por que tengo un error de 2.5???
print("zfinal")
print(error)
print("zdiff(real)")
print(zstar_s[:,0]-np.transpose(error))
print("zdiff_medido")
print(zstar_s[:,0]-np.transpose(error)+Bias)
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
