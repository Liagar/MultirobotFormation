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

import RobotModels as rob
from RobotModels import robot_plot_utils as rplot
import numpy as np
import matplotlib.pyplot as plt
from RobotModels import graph_utils as graf

n=8
p=np.array([-1,1,0,1,1,1,2,0,1,-1,0,-1,-1,-1,-2,0])
Z=np.array([[1,1,1,1,2,2,3,3,4,4,5,6,7,4,5],
            [2,3,4,5,4,7,5,6,5,6,7,8,8,8,8]])

kappa=np.ones(n)
N=[[]]
for i in range(n):
    #Busca los vecinos del robot i
    Ni=graf.neighbours(Z,i)
    N.append(Ni)

print(N)