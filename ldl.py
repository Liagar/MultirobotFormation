#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:10:11 2025

@author: lia
"""

import numpy as np

    

def ldlt_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for i in range(n):
        sum_ldl = sum(L[i, k] ** 2 * D[k] for k in range(i))
        D[i] = A[i, i] - sum_ldl

        for j in range(i + 1, n):
            sum_l = sum(L[j, k] * L[i, k] * D[k] for k in range(i))
            L[j, i] = (A[j, i] - sum_l) / D[i]

    return L, D        
    

def diag_sys(A,b):
    [f,c]=np.shape(A)
    x=np.zeros([f,1])
    for i in range(f):
        x[i]=b[i]/A[i,i]
    return x

def progressive(A,b):
    ''' This function computes the solution of a lower equation system using 
    progressive substitution. It receives the coefficient matriz A and the 
    independent term vector b. It returns the vector solution x
    '''
    # Coefficient matrix size. Return error if not square
    [f,c]=np.shape(A)
    if f!=c:
        print("A is not square")
        return
    # To build the solution vector x
    x=np.zeros(f)
    x=b.copy()
    for i in range(f):
        '''The inner block subtracts to the independent term the previous solution
         multiplied by the correspondent coefficient'''
        for j in range(0,i):
            x[i]-=x[j]*A[i,j]
        '''Finaly divide the result by the diagonal coefficient'''
        x[i]=(x[i])/A[i,i]
    return x

def regressive(A,b):
    ''' This function computes the solution of a upper equation system using 
    regressive substitution. It receives the coefficient matriz A and the 
    independent term vector b. It returns the vector solution x
    '''
    # Coefficient matrix size. Return error if not square
    [f,c]=np.shape(A)
    if f!=c:
        print("A is not square")
        return
    # To build the solution vector x
    x=np.zeros(f)
    x=b.copy()

    for i in range(f-1,-1,-1):
        '''The inner block subtracts to the independent term the previous solution
         multiplied by the correspondent coefficient'''
        for j in range(i+1,f,1):
            x[i]-=x[j]*A[i,j]
            
        '''Finaly divide the result by the diagonal coefficient'''
        x[i]=(x[i])/A[i,i]
    return x

def inverseUsingLDLT(L,D):
    n,nc=np.shape(L)
    e=np.zeros(n)
    inv=np.zeros([n,n])
    for i in range(n):
        #Vector unitario
        for j in range(n):
            if i==j:
                e[j]=1.0
        y=progressive(L, e)
        z=diag_sys(np.diag(D), y)
        x=regressive(L.T, z)
        for j in range(n):
            inv[j,i]=x[j]
    return inv

def inverse_using_ldlt(L, D):
    n = L.shape[0]
    inv_A = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1

        # Forward substitution: L * y = e
        y=progressive(L, e)
        
        # Solve diagonal system: D * z = y
        z = y / D
        
        z=diag_sys(np.diag(D), y)
        

        # Backward substitution: Lᵀ * x = z
        x=regressive(L.T,z)
        
        inv_A[:, i] = x[:,0]

    return inv_A

A=np.array([[2,4.0,-2.,2],[4.0,9,-1,6],[-2,-1,14,13.],[2.,6,13,35]])
L,D=ldlt_decomposition(A)
print(L)
print(D)
Ia=inverse_using_ldlt(L, D)

print(Ia)
print(np.linalg.inv(A))