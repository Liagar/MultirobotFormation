#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:44:22 2025

@author: lia
"""
import numpy as np
import matplotlib.pyplot as plt

def barrera(R,p):
    x=np.linspace(p[0]-R, p[0]+R,20)
    ymas=p[1]+np.sqrt(R**2-(x-p[0])**2)
    ymenos=p[1]-np.sqrt(R**2-(x-p[0])**2)
    return x,ymas,ymenos

R=2
p=np.array([1.0,5.0])
xb,ybM,ybm=barrera(R,p)


fig, ax = plt.subplots()
ax.plot(p[0],p[1],'bo')
ax.plot(xb,ybM,'m-')
ax.plot(xb,ybm,'m-')
plt.show()