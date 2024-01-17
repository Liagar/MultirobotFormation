#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:53:20 2022

@author: lia
Useful functions for robot ploting
"""

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

def pinta_robot(x,y,theta,R,axs):
    c=mpatches.Circle((x,y),R)
    patches=[]
    patches.append(c)
    p=PatchCollection(patches)
    axs.add_collection(p)
    Path=mpath.Path
    path_data=[
            (Path.MOVETO,[x,y]),
            (Path.LINETO,[x+R*np.cos(theta),y+R*np.sin(theta)])
            ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='r')
    axs.add_patch(patch)
    axs.grid()
    axs.axis('equal')    
    return path

def pinta_robot_c(x,y,theta,R,axs,color):
    c=mpatches.Circle((x,y),R)
    patches=[]
    patches.append(c)
    p=PatchCollection(patches)
    axs.add_collection(p)
    Path=mpath.Path
    path_data=[
            (Path.MOVETO,[x,y]),
            (Path.LINETO,[x+R*np.cos(theta),y+R*np.sin(theta)])
            ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color)
    axs.add_patch(patch)
    axs.grid()
    axs.axis('equal')    
    return path

def dibuja_grafo_2D(V,Z):
    g=np.zeros([2,2])
    n,m =np.shape(Z)
   # print("n: ",n)
    #print("m: ",m)
    for i in range(n):
        g[0,:]=V[Z[i,0]-1,:]
        g[1,:]=V[Z[i,1]-1,:]
        plt.plot(g[:,0],g[:,1],'k--')
        
    