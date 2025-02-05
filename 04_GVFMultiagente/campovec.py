import numpy as np
import matplotlib.pyplot as plt 
def phi1(x,w):
    f1 = x - 15*np.sin(2*w)
    return f1
def phi2(y,w):
    f2 = y - 30*np.sin(w)*np.sqrt(0.5*(1-0.5*np.sin(w)**2))
    return f2

def dphi1(w): 
    f1 = -30*np.cos(2*w)
    return f1
def dphi2(w): 
    f2 = -(15*np.cos(w)*np.sqrt(np.cos(w)**2+1))
    return f2

puntos = np.arange(-30,30,3)

valores = np.linspace(-4,4,100)
[xm,ym] = np.meshgrid(puntos,puntos)
K1 = 10
K2 = 10
for w in np.arange(0,np.pi,np.pi/4):
    vx = phi1(xm,w)
    vy = phi2(ym,w)
    dx = dphi1(w)
    dy = dphi2(w)
    plt.figure()
    #nuestra curva 
    plt.plot(15*np.sin(2*valores),30*np.sin(valores)*np.sqrt(0.5*(1-0.5*np.sin(valores)**2)),color="royalblue",linewidth = 2.5,zorder = 2)
    plt.scatter(15*np.sin(2*w),30*np.sin(w)*np.sqrt(0.5*(1-0.5*np.sin(w)**2)),color = "red",linewidth = 2.5, zorder = 3)
    #campo vectorial - gradiente 
    plt.quiver(puntos,puntos,-K1*vx+dx,-K2*vy+dy, color = "olive", zorder = 1)
    plt.title('w = '+str(w))