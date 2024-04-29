#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:43:34 2024

@author: lia
Uncerstanding control barrier functions
"""
import RobotModels as mbm
import numpy as np
import matplotlib.pyplot as plt
from robot_plot_utils import  pinta_robot_c
from qpsolvers import solve_qp

r1i=np.array([-10,10,0])
r2i=np.array([10,-10,0])
R=0.5
fig, axs = plt.subplots(1)
pinta_robot_c(r1i[0], r1i[1], r1i[2], R, axs,'red')
pinta_robot_c(r2i[0], r2i[1], r2i[2], R, axs,'cyan')

'''Control sin CBF'''
kw=1
kv=0.015
r1_star=np.array([10,-10])
r2_star=np.array([-10,10])
t=0
dt=0.01
tf=8
n=int(tf/dt)
r1=np.zeros([n,3])
r2=np.zeros([n,3])
r1[0,:]=r1i
r2[0,:]=r2i
i=0
while i<n-1:
    '''Robot 1'''
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v=kv*np.linalg.norm(r1[i,0:1]-r1_star)**2
    w=kw*(fi-r1[i,2])
    sdot=mbm.unicycle(v,w,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    '''Robot 2'''
    fi=np.arctan2(r2_star[1]-r2[i,1],r2_star[0]-r2[i,0])
    v=kv*np.linalg.norm(r2[i,0:1]-r2_star)**2
    w=kw*(fi-r2[i,2])
    sdot=mbm.unicycle(v,w,r2[i,2])
    r2[i+1,0]=r2[i,0]+dt*sdot[0]
    r2[i+1,1]=r2[i,1]+dt*sdot[1]
    r2[i+1,2]=r2[i,2]+dt*sdot[2]
    
    i=i+1
    pinta_robot_c(r1[i,0], r1[i,1], r1[i,2], R, axs,'red')
    pinta_robot_c(r2[i,0], r2[i,1], r2[i,2], R, axs,'magenta')
    t=t+dt

''' Control con CBF'''
i=0
t=0
Ds=0.5
gamma=0.45
Q=np.eye(4)
#TODO: Revisar, algo no esta bien en la expresion del problema cuadratico
print("Q ",Q)
fig, axs = plt.subplots(1)
pinta_robot_c(r1i[0], r1i[1], r1i[2], R, axs,'red')
pinta_robot_c(r2i[0], r2i[1], r2i[2], R, axs,'cyan')
while i<n-1:
    ''' Velocidades objetivos'''
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v1=kv*np.linalg.norm(r1[i,0:1]-r1_star)**2
    w1=kw*(fi-r1[i,2])
    fi=np.arctan2(r2_star[1]-r2[i,1],r2_star[0]-r2[i,0])
    v2=kv*np.linalg.norm(r2[i,0:1]-r2_star)**2
    w2=kw*(fi-r2[i,2])
    '''CBF'''
    d1=np.linalg.norm(r1[i,0:1]-r2[i,0:1])
    c=np.array([v1,w1,v2,w2]).T
    print("c: ",c)
    G=-2*np.array([-d1,0,d1,0])   
    h=gamma*(np.linalg.norm(r1[i,0:1]-r2[i,0:1])**2-Ds**2)*np.array([1.0])
    print("h: ",h)
    u=solve_qp(Q,c,G,h,A=None,b=None,solver='quadprog',verbose=True)
    v1=u[0]
    w1=u[1]
    v2=u[2]
    w2=u[3]
    sdot=mbm.unicycle(v1,w1,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    '''Robot 2'''
    sdot=mbm.unicycle(v2,w2,r2[i,2])
    r2[i+1,0]=r2[i,0]+dt*sdot[0]
    r2[i+1,1]=r2[i,1]+dt*sdot[1]
    r2[i+1,2]=r2[i,2]+dt*sdot[2]
    
    i=i+1
    pinta_robot_c(r1[i,0], r1[i,1], r1[i,2], R, axs,'red')
    pinta_robot_c(r2[i,0], r2[i,1], r2[i,2], R, axs,'magenta')
    t=t+dt
