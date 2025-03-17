import numpy as np
import matplotlib.pyplot as plt 
from ldl import

def phi1(x,w):
    f1 = x - 15*np.sin(2*w)
    return f1
def phi2(y,w):
    f2 = y - 30*np.sin(w)*np.sqrt(0.5*(1-0.5*np.sin(w)**2))
    return f2
def phi3(z,w): 
    f3 = z - (3+5*np.cos(2*w))
    return f3 

def dphi1(w): 
    f1 = -30*np.cos(2*w)
    return f1
def dphi2(w): 
    f2 = -(15*np.cos(w)*np.sqrt(np.cos(w)**2+1))
    return f2
def dphi3(w): 
    f3 = 10 * np.sin(2*w) 
    return f3


puntos = np.arange(-25,26,12)

valores = np.linspace(-4,4,100)

[xm,ym,zm] = np.meshgrid(puntos,puntos,np.arange(-10,11,5))
K1 = 20
K2 = 20
K3 = 20

for w in np.arange(0,np.pi,np.pi/4):
    vx = phi1(xm,w)
    vy = phi2(ym,w)
    vz = phi3(zm,w)
    dx = dphi1(w)
    dy = dphi2(w)
    dz = dphi3(w)

    #m = np.sqrt((-K1*vx-dx)**2+(-K2*vy-dy)**3+(-K3*vz-dz)**2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #nuestra curva 
    ax.plot(15*np.sin(2*valores),30*np.sin(valores)*np.sqrt(0.5*(1-0.5*np.sin(valores)**2)),3+5*np.cos(2*valores),color="royalblue",linewidth = 2.5,zorder = 2)
    ax.scatter(15*np.sin(2*w),30*np.sin(w)*np.sqrt(0.5*(1-0.5*np.sin(w)**2)),3+5*np.cos(2*w),color = "red",linewidth = 2.5, zorder = 3)
    #campo vectorial - gradiente 
    ax.quiver(xm,ym,zm,(-K1*vx-dx),(-K2*vy-dy),(-K3*vz-dz) ,color = "y", length=5, normalize=True,zorder = 1)
    ax.set_title('w = '+str(w),  fontsize = 15)
    ax.set_xlabel("x", fontsize = 15)
    ax.set_ylabel("y", fontsize = 15)
    ax.set_zlabel("z", fontsize = 15)
