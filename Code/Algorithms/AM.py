# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:18:40 2020

@author: ameer
"""

import numpy as np

def aRWM(target, beta, init, M, B):
    dim = len(init)
    samps = np.zeros((M+B,dim))
    accepts = 0
    cur = init 
    
    propmean = np.zeros(dim)
    propcov = (0.1**2)*np.eye(dim)/dim
    
    for i in range(M+B):
        prop = cur + np.random.multivariate_normal(propmean, propcov)
        U = np.random.rand(1)[0]
        alpha = target(prop)/target(cur)
        if alpha > U:
            cur = prop
            accepts += 1
        samps[i,:] = cur
        if i <= 2*dim:
            propcov = (0.1**2)*np.eye(dim)/dim
        else:
            propcov = (1-beta)*(2.38**2)*np.cov(samps[:(i+1),:], rowvar=False)/dim + beta*(0.1**2)*np.eye(dim)/dim
    
    print(accepts/(M+B))
    print(propcov)
    return samps[(B+1):,:]