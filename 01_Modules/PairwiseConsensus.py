#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:29:51 2025

@author: lia
"""

'''
Pairwise coordination
Consensus
'''

from Utils import RobotModels as rob
from Utils import robot_plot_utils as rplot
import numpy as np
import matplotlib.pyplot as plt
from Utils import graph_utils as grf
from matplotlib.animation import FuncAnimation

def unstack(ps,m):
    n1=np.shape(ps)
    n=n1[0]
    npuntos=int(n/m)
    p=np.zeros([npuntos,m])
    i=0
    k=0
    while i<n:
        for j in range(m):
            p[k,j]=ps[i+j]

        k=k+1
        i=i+m
    return p

def neighbours(Z,i):
    n = []
    num_arcos, _ = np.shape(Z)
    for j in range(num_arcos):
        if Z[j,0] == i:
            n.append(Z[j,1])
        if Z[j,1] == i:
            n.append(Z[j,0])
    return n

# TODO: Revisar por qué se separan

def update_robots():
    n=8
    p=np.array([-1,1,0,1,1,1,2,0,1,-1,0,-1,-1,-1,-2,0])
    p_us=unstack(p,2)
    Z=np.array([[1,1,1,1,2,2,3,3,4,4,5,6,7,4,5],
                [2,3,4,5,4,7,5,6,5,6,7,8,8,8,8]])
    Z=Z.T
    kappa=np.ones(n)
    N=[]
    u=np.zeros((n,2))
    tf=10
    dt=0.01
    t=np.arange(0,tf,dt )
    
    pos=np.zeros((len(t),n,2))
    pos[0,:,:]=p_us
    
    for k in range(len(t)-1):
        for i in range(1,n+1):
            
            #Busca los vecinos del robot i
            Ni=neighbours(Z, i)
            N.append(Ni)
            #Cambia las coordenadas de los vecinos a locales de i y calcula u
            u[i-1,0]=0
            u[i-1,1]=0
            nni=len(Ni)
            pi=np.zeros((nni,2))
            for j in range(nni):
                pi[j,0]=p_us[j,0]-p_us[i-1,0]
                pi[j,1]=p_us[j,1]-p_us[i-1,1]
                u[i-1,0]+=pi[j,0]
                u[i-1,1]+=pi[j,1]
            
            u[i-1,:]=kappa[i-1]*u[i-1,:]
            
        #Integración
        pos[k+1,:,:]=pos[k,:,:]+u[:,:]*dt
    return pos,t


# ============================
# Animación con Matplotlib
# ============================

pos, t = update_robots()
n = pos.shape[1]

fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Coordinación de robots (consenso)")

lines = [ax.plot([], [], 'o-', lw=2)[0] for _ in range(n)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(frame):
    for i, line in enumerate(lines):
        line.set_data(pos[:frame, i, 0], pos[:frame, i, 1])
    return lines

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(t), interval=50, blit=True)

plt.show()
