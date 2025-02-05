#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:46:07 2024

@author: lia
"""

import numpy as np
from qpsolvers import Problem, solve_problem

def qp_solve(P,q,G,h,A,b,lb,ub):
    ''' Resuelve el problema de minimización cuadrática
    min 0.5*x'Px+q'x
    sujeto a Gx<h
    Ax=b
    lb<=x<=ub
    '''
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = solve_problem(problem, solver="quadprog")
    return solution.x




M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = M.T.dot(M)  # quick way to build a symmetric matrix
q = np.array([3., 2., 3.]).dot(M).reshape((3,))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))
A = np.array([1., 1., 1.])
b = np.array([1.])
lb = -0.6 * np.ones(3)
ub = +0.7 * np.ones(3)


'''
print(f"Primal: x = {solution.x}")
print(f"Dual (Gx <= h): z = {solution.z}")
print(f"Dual (Ax == b): y = {solution.y}")
print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")
'''

M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = -np.dot(M.T, np.array([3., 2., 3.]))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))
problem = Problem(P, q, G, h)
solution = solve_problem(problem, solver="quadprog")


M = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
P = np.dot(M.T, M)
q = -np.dot(M.T, np.array([1., 0.5, 1.,0.5]))
G = np.array([[1., 0., 0.,0.], [0., 1., 0.,0.], [0., 0., 1.,0.], [0., 0., 0.,1.]])
h = np.array([3., 2., 2.,2.]).reshape((4,))
problem = Problem(P, q, G, h)
solution = solve_problem(problem, solver="quadprog")

print(f"Primal: x = {solution.x}")
print(f"Dual (Gx <= h): z = {solution.z}")
print(f"Dual (Ax == b): y = {solution.y}")
print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")