#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test models
Created on Wed Dec 28 11:58:43 2022

@author: lia
"""

import RobotModels as mbm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

x=[]
y=[]
theta=[]
u1=[]
u2=[]

x.append(1)
y.append(1)
theta.append(0.5)
u1.append(1)
u2.append(0)
dt=0.01
v=1 
w=0.1
t=0;
i=0

while t<100:
    i=i+1
    sdot=mbm.unicycle(v,w,theta[i-1])
    x.append(x[i-1]+dt*sdot[0])
    y.append(y[i-1]+dt*sdot[1])
    theta.append(theta[i-1]+dt*sdot[2]) 
    u1.append(x[i]*np.cos(theta[i]))
    u2.append(y[i]*np.sin(theta[i]))
    t=t+dt

fig, axs = plt.subplots()
    
plt.plot(x,y)
for i in range(10):
    arrow = mpatches.FancyArrow(x[100*i],y[100*i], u1[100*i],u2[100*i],width=.1, length_includes_head=True, color="C1")
    axs.add_patch(arrow)    
