# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:19:12 2020

@author: ameer
"""

import numpy as np

def RWM(target, sigma, init, M, B):
    dim = len(init)
    samps = np.zeros((M+B,dim))
    accepts = 0
    cur = init 
    
    propmean = np.zeros(dim)
    propcov = sigma*np.eye(dim)
    
    for i in range(M+B):
        prop = cur + np.random.multivariate_normal(propmean, propcov)
        U = np.random.rand(1)[0]
        alpha = target(prop)/target(cur)
        if alpha > U:
            cur = prop
            accepts += 1
        samps[i,:] = cur
    
    print(accepts/(M+B))
    return samps[(B+1):,:]

