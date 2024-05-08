#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:31:10 2024

@author: lia
CBF Robotarium

"""

import RobotModels as mbm
import numpy as np
import matplotlib.pyplot as plt
 
from qpsolvers import Problem, solve_problem
    
import matplotlib.animation as manimation

def robot_arena(ancho,largo):
    cuadrado=np.zeros([4,2])
    cuadrado[0,0]=0.0
    cuadrado[0,1]=0.0
    cuadrado[1,0]=0.0
    cuadrado[1,1]=largo
    cuadrado[2,0]=ancho
    cuadrado[2,1]=largo
    cuadrado[3,0]=ancho
    cuadrado[3,1]=0.0 
    return cuadrado

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    print("Q: ",qp_G)
    qp_a = -q
    print("c: ",qp_a)
    if A is not None: 
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        #meq = A.shape[0]
    else:  # no equality constraint
        #qp_C = -G.T
        qp_C=-G
        qp_b=-h
        #qp_b = -h
        #meq = 0
    print("A: ",qp_C)
    print("b: ",qp_b)
    problem=Problem(qp_G,qp_a,qp_C,qp_b)
    solution=solve_problem(problem,solver='quadprog')
    print(f"Primal: x = {solution.x}")
    print(f"Dual (Gx <= h): z = {solution.z}")
    print(f"Dual (Ax == b): y = {solution.y}")
    print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")
    return solution

def cbf_function(x,y,l,a):
    y=np.cos(np.pi*x/a-0.5)*np.cos(np.pi*y/l-0.5)
    return y

#Robot arena
ancho= 50
largo= 30
r1i=np.array([ancho/2,largo/2,np.pi/2])


R=0.5
kw=0.9
kv=0.015
r1_star=np.array([ancho,largo])
t=0
dt=0.1
tf=20
n=int(tf/dt)
r1=np.zeros([n,3])
r1[0,:]=r1i
fi_1=np.zeros([n])
vr=np.zeros([n])
wr=np.zeros([n])
i=0
fi=2
v=5
while i<n-1:
    '''Robot 1'
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v=kv*np.linalg.norm(r1[i,0:1].T-r1_star.T)**2
    delta=fi-r1[i,2]
    if np.abs(delta)>0.8:
        v=0
    w=kw*(fi-r1[i,2])'''
    fi-=0.01
    w=fi
    fi_1[i]=fi
    vr[i]=v
    wr[i]=w
    
    sdot=mbm.unicycle(v,w,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    
    i=i+1
    t=t+dt


fig=plt.subplot(2,1,1)
plt.title("Without CBF")
t=np.linspace(0, tf,n)
plt.plot(t,fi_1,'g-',label='fi_1')
plt.plot(t,wr[:],'b-',label='w1')
plt.legend()
plt.subplot(2,1,2)  
plt.plot(t,vr[:],'g-',label='v1')
plt.legend()


#Animacion
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=15, metadata=metadata)
fig, ax=plt.subplots()
arena=robot_arena(ancho,largo)
plt.plot(r1[:,0], r1[:,1],'r')

plt.plot(r1_star[0],r1_star[1],'rx')
plt.plot(r1[0,0],r1[0,1],'rs')
plt.fill(arena[:,0],arena[:,1],'c',alpha=0.4)
red_robot, = plt.plot([], [], 'ro', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')
ax.set_ylim(top=largo+10,bottom=-10) 
ax.set_xlim(right=10+ancho,left=-10) 

with writer.saving(fig, "Robotarium_Ncbf_1.mp4", 200):
    for i in range(n):
        x0 = r1[i,0]
        y0 = r1[i,1]
        red_robot.set_data(x0, y0)
        writer.grab_frame()

#Control con CBF
i=0
t=0
Ds=3
gamma=0.2
M = np.array([[1., 0.], [0., 1.]])
P = np.dot(M.T, M)
#TODO: Revisar al salir de la zona pega un petardazo la velocidad ¿va al revés?
v_star=np.zeros([n,2])
v_opt=np.zeros([n,2])

a_1=0.01
fi=2
v1=5

while i<n-1:
    #Velocidades objetivos
    ''' Control para ir al objetivo
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v1=kv*np.linalg.norm(r1[i,0:1]-r1_star)**2
    delta=fi-r1[i,2]
    if np.abs(delta)>0.8:
        v=0
    w1=kw*(fi-r1[i,2])'''
    ''' Control de movimiento espiral'''
    fi-=0.01
    w1=fi
    fi_1[i]=fi
    vr[i]=v1
    wr[i]=w1
    
    #CBF
    q = -np.dot(M.T, np.array([v1, w1]))
    #G = np.array([[1., 0., -1.,0.], [0., 1., 0.,0.], [0., 0., 0.,1.], [-1., 0., 1.,0.]])
    #G = -2*np.array([[dx, dy, -dx,-dy], [dx, dy, -dx,-dy], [dx, dy, -dx,-dy], [dx, dy, -dx,-dy]])
    #a1=dx*r1[i,0]*np.cos(r1[i,2])+dy*r1[i,1]*np.sin(r1[i,2])
    #a2=-dx*r2[i,0]*np.cos(r2[i,2])-dy*r2[i,1]*np.sin(r2[i,2])
    #Modelo ampliado
    cx=np.cos(np.pi*(r1[i,0]/ancho-0.5))
    cy=np.cos(np.pi*(r1[i,1]/largo-0.5))
    sx=np.sin(np.pi*(r1[i,0]/ancho-0.5))
    sy=np.sin(np.pi*(r1[i,1]/largo-0.5))
    c=np.cos(r1[i,2])
    s=np.sin(r1[i,2])
    a1=-(np.pi/ancho)*sx*cy*c-(a_1*np.pi/largo)*cx*sy*s
    a2=(a_1*np.pi/ancho)*sx*cy*s+(a_1*np.pi/largo)*cx*sy*c
    #Cambio de signo toda la desigualdad porque en solver es Gu<=h
    G=np.array([a1,a2])
    mG=np.array([-a1,-a2])
    #h = gamma*(d1**2-Ds**2)*np.array([1., 1., 1.,1.]).reshape((4,))
    hf=cbf_function(r1[i,0], r1[i,1], largo, ancho)
    h = np.array([-gamma*hf])
    #Voy a comprobar si el vector u está dentro de Kcbf
    Lg_h=np.dot(G,q)
    Kcbf=Lg_h+gamma*hf
    print("Kcbf: ",Kcbf)
    if Kcbf<0:
        problem = Problem(P, q, G, h)
        solution = solve_problem(problem, solver="quadprog")
        print(f"Primal: x = {solution.x}")
        print(f"Dual (Gx <= h): z = {solution.z}")
        print(f"Dual (Ax == b): y = {solution.y}")
        print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")
        u=solution.x
        print("v: ",v1,"\t w: ",w1,"\t v_opt: ",u[0],"\t w_opt:",u[1])
        v_opt[i,:]=u
        v1=u[0]
        w1=u[1]
        
    else:
        v_opt[i,0]=v1
        v_opt[i,1]=w1
        
    
    v_star[i,0]=v1
    v_star[i,1]=w1
    sdot=mbm.extended_unicycle(v1,w1,r1[i,2], a_1)
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    
    i=i+1
    t=t+dt


#Animacion
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig, ax=plt.subplots()
arena=robot_arena(ancho,largo)
plt.plot(r1[:,0], r1[:,1],'r')

plt.plot(r1_star[0],r1_star[1],'rx')
plt.plot(r1[0,0],r1[0,1],'rs')
plt.fill(arena[:,0],arena[:,1],'c',alpha=0.4)
red_robot, = plt.plot([], [], 'ro', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')
ax.set_ylim(top=10+largo,bottom=-10) 
ax.set_xlim(right=10+ancho,left=-10) 

with writer.saving(fig, "Robotarium_cbf_1.mp4", 100):
    for i in range(n):
        x0 = r1[i,0]
        y0 = r1[i,1]
        red_robot.set_data(x0, y0)
        writer.grab_frame()

t=np.linspace(0, tf,n)
plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, v_opt[:,0],'r',label='v_real') 
ax1.plot(t,v_star[:,0],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v1(m/s)')
plt.legend()

ax1 = plt.subplot(2, 1, 2)
ax1.plot(t, v_opt[:,1],'r',label='v_real') 
ax1.plot(t,v_star[:,1],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('w1(rad/s)')
plt.legend()
