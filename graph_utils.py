# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:43:39 2024

@author: liaga
"""
import numpy as np

def matriz_incidencia(Z,num_nodos,num_arcos):
    B=np.zeros([num_arcos,num_nodos])
    for i in range(num_arcos):
        head=Z[i,0]-1
        tail=Z[i,1]-1
        B[i,head]=1
        B[i,tail]=-1
    return B

def matriz_laplaciana(Z,pesos,num_nodos,num_arcos):
    L=np.zeros([num_nodos,num_nodos])
    for k in range(num_arcos):
        head=Z[k,0]-1
        tail=Z[k,1]-1
        L[head,tail]=-pesos[head,tail]
        L[tail,head]=-pesos[tail,head]
    for i in range(num_nodos):
        for k in range(num_nodos):
            if k==i:
                continue
            L[i,i]=L[i,i]-L[i,k]
    return L

def unstack(ps,m):
    n =len(ps)
    npuntos=int(n/m)
    p=np.zeros([npuntos,m])
    i=0
    k=0
    while i<n:
        for j in range(m):
            p[k,j]=ps[i+j]

        k=k+1
        i=i+m
    return p

def stack(p):
    n,m=np.shape(p)
    ps=np.zeros([n*m,1])
    j=0
    for i in range(n):
        for k in range(m):
            ps[j,0]=p[i,k]
            j+=1
    return ps