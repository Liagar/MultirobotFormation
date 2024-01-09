# -*- coding: utf-8 -*-
"""
Lia Garcia-Perez
Common models of mobile robots
"""
from math import cos, sin, tan

def integrator_point_model(v,w):
    xdot=v
    ydot=w
    thetadot=0
    sdot=[xdot,ydot,thetadot]
    return sdot


def unicycle(v,w,theta):
    xdot=v*cos(theta)
    ydot=v*sin(theta)
    thetadot=w
    sdot=[xdot,ydot,thetadot]
    return sdot

def bicycle(v,w,theta,gamma,B):
    xdot = v*cos(theta)
    ydot=v*sin(theta)
    try:
        thetadot=(v/B)*tan(gamma)
    except ZeroDivisionError:
        thetadot=0
    gammadot=w
    sdot=[xdot,ydot,thetadot,gammadot]
    return sdot

