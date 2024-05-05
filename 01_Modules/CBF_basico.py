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
from qpsolvers import solve_qp, Problem, solve_problem
from time import sleep
import matplotlib.animation as manimation

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

r1i=np.array([-10 + 5 * np.random.randn() ,10 + 5 * np.random.randn(),np.pi/2 + np.pi/4 * np.random.randn()])
r2i=np.array([10 + 5 * np.random.randn() ,-10 + 5 * np.random.randn(),np.pi/2 + np.pi/4 * np.random.randn()])
R=0.5
fig, axs = plt.subplots(1)
#pinta_robot_c(r1i[0], r1i[1], r1i[2], R, axs,'red')
#pinta_robot_c(r2i[0], r2i[1], r2i[2], R, axs,'cyan')

'''Control sin CBF'''

kw=0.9
kv=0.015
r1_star=np.array([10,-10])
r2_star=np.array([-11,12])
t=0
dt=0.1
tf=15
n=int(tf/dt)
r1=np.zeros([n,3])
r2=np.zeros([n,3])
r1[0,:]=r1i
r2[0,:]=r2i
fi_1=np.zeros([n])
fi_2=np.zeros([n])
vr=np.zeros([n,2])
wr=np.zeros([n,2])
i=0
while i<n-1:
    '''Robot 1'''
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v=kv*np.linalg.norm(r1[i,0:1].T-r1_star.T)**2
    delta=fi-r1[i,2]
    if np.abs(delta)>0.8:
        v=0
    w=kw*(fi-r1[i,2])
    fi_1[i]=fi
    vr[i,0]=v
    wr[i,0]=w
    sdot=mbm.unicycle(v,w,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    '''Robot 2'''
    fi=np.arctan2(r2_star[1]-r2[i,1],r2_star[0]-r2[i,0])
    v=kv*np.linalg.norm(r2[i,0:1]-r2_star)**2
    if np.abs(fi-r2[i,2])>0.8:
        v=0
    w=kw*(fi-r2[i,2])
    fi_2[i]=fi
    vr[i,1]=v
    wr[i,1]=w

    sdot=mbm.unicycle(v,w,r2[i,2])
    r2[i+1,0]=r2[i,0]+dt*sdot[0]
    r2[i+1,1]=r2[i,1]+dt*sdot[1]
    r2[i+1,2]=r2[i,2]+dt*sdot[2]
    
    i=i+1
    pinta_robot_c(r1[i,0], r1[i,1], r1[i,2], R, axs,'r')
    pinta_robot_c(r2[i,0], r2[i,1], r2[i,2], R, axs,'c')
    t=t+dt
plt.title("Without CBF")
fig=plt.subplot(2,2,1)
t=np.linspace(0, tf,n)
plt.plot(t,fi_1,'g-',label='fi_1')
plt.plot(t,wr[:,0],'b-',label='w1')
plt.legend()
plt.subplot(2,2,2)
plt.plot(t,vr[:,0],'g-',label='v1')
plt.legend()
plt.subplot(2,2,3)
plt.plot(t,fi_2,'g-',label='fi_2')
plt.plot(t,wr[:,1],'b-',label='w2')
plt.legend()
plt.subplot(2,2,4)
plt.plot(t,vr[:,1],'g-',label='v2')
plt.legend()

#Animacion
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()

plt.plot(r1[:,0], r1[:,1],'r')
plt.plot(r2[:,0], r2[:,1],'b')
plt.plot(r1_star[0],r1_star[1],'rx')
plt.plot(r2_star[0],r2_star[1],'bx')
plt.plot(r1[0,0],r1[0,1],'rs')
plt.plot(r2[0,0],r2[0,1],'bs')
red_robot, = plt.plot([], [], 'ro', markersize = 10)
cyan_robot,= plt.plot([], [], 'bo', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')

with writer.saving(fig, "Ncbf_1.mp4", 100):
    for i in range(n):
        x0 = r1[i,0]
        y0 = r1[i,1]
        red_robot.set_data(x0, y0)
        x0 = r2[i,0]
        y0 = r2[i,1]
        cyan_robot.set_data(x0, y0)
        writer.grab_frame()

'''Prueba con un problema inventado para ver que lo entiendo
i=0
t=0
Ds=0.5
gamma=0.01
P=np.eye(4,4)
M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
P = np.dot(M.T, M)
fig, axs = plt.subplots(1)
pinta_robot_c(r1i[0], r1i[1], r1i[2], R, axs,'r')
pinta_robot_c(r2i[0], r2i[1], r2i[2], R, axs,'c')
# Para comparar las soluciones del optimizador con las propuestas
v_star=np.zeros([n,4])
v_opt=np.zeros([n,4])
while i<n-1:
    #Velocidades objetivos
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v1=kv*np.linalg.norm(r1[i,0:1]-r1_star)**2
    w1=kw*(fi-r1[i,2])
    fi=np.arctan2(r2_star[1]-r2[i,1],r2_star[0]-r2[i,0])
    v2=kv*np.linalg.norm(r2[i,0:1]-r2_star)**2
    w2=kw*(fi-r2[i,2])
    '#CBF
    d1=np.linalg.norm(r1[i,0:1]-r2[i,0:1])
    #q=np.array([[v1],[w1],[v2],[w2]])
    #G=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])   
    #h=0.01*np.array([[1],[1],[1],[1]])
    q = -np.dot(M.T, np.array([v1, w1, v2,w2]))
    G = np.array([[1., 0., 0.,0.], [0., 1., 0.,0.], [0., 0., 1.,0.], [0., 0., 0.,1.]])
    h = np.array([5., 1., 5.,1.]).reshape((4,))
    problem = Problem(P, q, G, h)
    solution = solve_problem(problem, solver="quadprog")
    u=solution.x
    v_opt[i,:]=u
    v_star[i,0]=v1
    v_star[i,1]=w1
    v_star[i,2]=v2
    v_star[i,3]=w2
    v1=u[0]
    w1=u[1]
    v2=u[2]
    w2=u[3]
    sdot=mbm.unicycle(v1,w1,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    #Robot 2
    sdot=mbm.unicycle(v2,w2,r2[i,2])
    r2[i+1,0]=r2[i,0]+dt*sdot[0]
    r2[i+1,1]=r2[i,1]+dt*sdot[1]
    r2[i+1,2]=r2[i,2]+dt*sdot[2]
    
    i=i+1
    pinta_robot_c(r1[i,0], r1[i,1], r1[i,2], R, axs,'red')
    pinta_robot_c(r2[i,0], r2[i,1], r2[i,2], R, axs,'magenta')

    t=t+dt

t=np.linspace(0, tf,n)
plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t, v_opt[:,0],'r',label='v_real') 
ax1.plot(t,v_star[:,0],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 2)
ax1.plot(t, v_opt[:,1],'r',label='v_real') 
ax1.plot(t,v_star[:,1],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 3)
ax1.plot(t, v_opt[:,2],'r',label='v_real') 
ax1.plot(t,v_star[:,2],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 4)
ax1.plot(t, v_opt[:,3],'r',label='v_real') 
ax1.plot(t,v_star[:,3],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')
plt.legend()

'''

#Control con CBF
i=0
t=0
Ds=2
gamma=0.4
M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
P = np.dot(M.T, M)
#TODO: Revisar,creo que lo que pasa es que no hay restricción para omega y si
# el robot está ya alineado omega es cero y cumple con la restricción así
# que no cambia y por lo tanto no hay manera de salir de ahí
# Para comparar las soluciones del optimizador con las propuestas
v_star=np.zeros([n,4])
v_opt=np.zeros([n,4])
fig, axs = plt.subplots(1)
pinta_robot_c(r1i[0], r1i[1], r1i[2], R, axs,'r')
pinta_robot_c(r2i[0], r2i[1], r2i[2], R, axs,'c')
while i<n-1:
    #Velocidades objetivos
    fi=np.arctan2(r1_star[1]-r1[i,1],r1_star[0]-r1[i,0])
    v1=kv*np.linalg.norm(r1[i,0:1]-r1_star)**2
    delta=fi-r1[i,2]
    if np.abs(delta)>0.8:
        v=0
    w1=kw*(fi-r1[i,2])
    fi=np.arctan2(r2_star[1]-r2[i,1],r2_star[0]-r2[i,0])
    if np.abs(fi-r2[i,2])>0.8:
        v2=0
    v2=kv*np.linalg.norm(r2[i,0:1]-r2_star)**2
    w2=kw*(fi-r2[i,2])
    #CBF
    d1=np.linalg.norm(r1[i,0:1]-r2[i,0:1])
    dx=r1[i,0]-r2[i,0]
    dy=r1[i,1]-r2[i,1]
    q = -np.dot(M.T, np.array([v1, w1, v2,w2]))
    #G = np.array([[1., 0., -1.,0.], [0., 1., 0.,0.], [0., 0., 0.,1.], [-1., 0., 1.,0.]])
    #G = -2*np.array([[dx, dy, -dx,-dy], [dx, dy, -dx,-dy], [dx, dy, -dx,-dy], [dx, dy, -dx,-dy]])
    a1=dx*r1[i,0]*np.cos(r1[i,2])+dy*r1[i,1]*np.sin(r1[i,2])
    a2=-dx*r2[i,0]*np.cos(r2[i,2])-dy*r2[i,1]*np.sin(r2[i,2])
    #Modelo ampliado
    #c=cos(r1[i,2])
    #s=np.sin(r1[i,2])
    #t1=
    G=2*np.array([a1,0,a2,0])
    #h = gamma*(d1**2-Ds**2)*np.array([1., 1., 1.,1.]).reshape((4,))
    h = np.array([gamma*(d1**2-Ds**2)])
    problem = Problem(P, q, G, h)
    solution = solve_problem(problem, solver="quadprog")
    u=solution.x
    v_opt[i,:]=u
    v_star[i,0]=v1
    v_star[i,1]=w1
    v_star[i,2]=v2
    v_star[i,3]=w2
    v1=u[0]
    w1=u[1]
    v2=u[2]
    w2=u[3]
    sdot=mbm.unicycle(v1,w1,r1[i,2])
    r1[i+1,0]=r1[i,0]+dt*sdot[0]
    r1[i+1,1]=r1[i,1]+dt*sdot[1]
    r1[i+1,2]=r1[i,2]+dt*sdot[2]
    #Robot 2
    sdot=mbm.unicycle(v2,w2,r2[i,2])
    r2[i+1,0]=r2[i,0]+dt*sdot[0]
    r2[i+1,1]=r2[i,1]+dt*sdot[1]
    r2[i+1,2]=r2[i,2]+dt*sdot[2]
    
    i=i+1
    t=t+dt
plt.title("With CBF")

#Animacion
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()

plt.plot(r1[:,0], r1[:,1],'r')
plt.plot(r2[:,0], r2[:,1],'b')
plt.plot(r1_star[0],r1_star[1],'rx')
plt.plot(r2_star[0],r2_star[1],'bx')
plt.plot(r1[0,0],r1[0,1],'rs')
plt.plot(r2[0,0],r2[0,1],'bs')
red_robot, = plt.plot([], [], 'ro', markersize = 10)
cyan_robot,= plt.plot([], [], 'bo', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')

with writer.saving(fig, "cbf_1.mp4", 100):
    for i in range(n):
        x0 = r1[i,0]
        y0 = r1[i,1]
        red_robot.set_data(x0, y0)
        x0 = r2[i,0]
        y0 = r2[i,1]
        cyan_robot.set_data(x0, y0)
        writer.grab_frame()

t=np.linspace(0, tf,n)
plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t, v_opt[:,0],'r',label='v_real') 
ax1.plot(t,v_star[:,0],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v1(m/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 2)
ax1.plot(t, v_opt[:,1],'r',label='v_real') 
ax1.plot(t,v_star[:,1],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('w1(rad/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 3)
ax1.plot(t, v_opt[:,2],'r',label='v_real') 
ax1.plot(t,v_star[:,2],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('v2(m/s)')
plt.legend()

ax1 = plt.subplot(2, 2, 4)
ax1.plot(t, v_opt[:,3],'r',label='v_real') 
ax1.plot(t,v_star[:,3],'c:',label='v_control')
plt.xlabel('t(s)')
plt.ylabel('w2(rad/s)')
plt.legend()


