#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:31:53 2022

@author: lia
Cyclic pursuit for integrator points
"""

import RobotModels as mbm
import numpy as np
import matplotlib.pyplot as plt

from robot_plot_utils import  pinta_robot_c

n=5

theta=np.zeros(n)
theta[0]=0
theta[1]=0.5
theta[2]=-1.2
theta[3]=2.2
theta[4]=3
z=np.zeros([n,2])
z[0][1]=10
z[0][0]=0
z[1][0]=-10
z[1][1]=-1
z[2][0]=12
z[2][1]=-5
z[3][0]=-5
z[3][1]=10
z[4][0]=13
R=np.zeros(n)
R[0]=0.2
R[1]=0.2
R[2]=0.2
R[3]=0.2
R[4]=0.2
fig, axs = plt.subplots(1)
t=0
dt=0.1
centroid=np.zeros(2)
c=['r','b','g','c','k']
while t<5:
   centroid[0]=0
   centroid[1]=0
   for i in range(4):
          # Robot number 4
       if i==4:
           v=(z[0][0]-z[4][0])
           omega=z[0][1]-z[4][1]
       else:
           v=z[i+1][0]-z[i][0]
           omega=z[i+1][1]-z[i][1]
       sdot=mbm.integrator_point_model(v,omega)
       theta[i]=theta[i]+dt*sdot[2]
       z[i][0]=z[i][0]+dt*sdot[0]
       z[i][1]=z[i][1]+dt*sdot[1]
       pinta_robot_c(z[i][0],z[i][1],theta[i],R[i],axs,c[i])
       centroid[0]+=z[i][0]
       centroid[1]+=z[i][1]
   centroid[0]=centroid[0]/n
   centroid[1]=centroid[1]/n
   axs.plot(centroid[0],centroid[1],'m+')
   t=t+dt