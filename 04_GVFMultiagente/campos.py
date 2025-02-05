#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:41:04 2024

@author: juan
"""

import numpy as np
import matplotlib.pyplot as pl

#ocho
x = np.arange(-2,2+0.001,0.001)
y1 = np.sqrt(x**2-x**4/4)
y2 = -np.sqrt(x**2-x**4/4)

pl.plot(x,y1,'r',)
pl.plot(x,y2,'r')


#campo
x = np.block([np.arange(-2,0,0.2),np.arange(0,2.2,0.2)])

y = x.copy()

[mx,my] = np.meshgrid(x,y)
K = 0.5
phi = my**2-mx**2+mx**4/4
gradphix = -2*mx+mx**3
gradphiy = 2*my

campox = -gradphiy - K*phi*gradphix
campoy = gradphix -  K*phi*gradphiy

pl.quiver(mx,my,campox,campoy,100)
pl.axis('equal')

#circulo
x = np.arange(-1,1+0.0001,0.0001)
y1 = np.sqrt(1-x**2)
y2 = -np.sqrt(1-x**2)
pl.figure()
pl.plot(x,y1,'r')
pl.plot(x,y2,'r')

#campo
x = np.arange(-1.5,1.5+0.25,0.25)
y = x.copy()
[mx,my] = np.meshgrid(x,y)
K = 0.5
phi = mx**2+my**2-1
gradphix = 2*mx
gradphiy = 2*my
campox = -gradphiy - K*phi*gradphix
campoy = gradphix -  K*phi*gradphiy

pl.quiver(mx,my,campox,campoy)
pl.axis('equal')