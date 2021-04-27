# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:48:45 2020

@author: ameer
"""

import numpy as np

def PTk(target, init, M, B, sigma = 1, k=2, maxtau=0.1):
    dim = len(init)
    tau = np.linspace(1, maxtau, k)
    #Output
    samps = np.zeros((M+B, dim))
    
    accepts = 0
    cur =  np.repeat(init.reshape((1,dim)), k, axis=0)
    print(cur)
    propmean = np.zeros(dim)
    propcov = sigma*np.eye(dim)
    
    for i in range(M+B):
        epsilon = np.random.multivariate_normal(propmean, propcov, size=k)
        prop = cur + epsilon
        U = np.random.rand(k)
        
        for j in range(k):
            curdensity = target(cur[j,:])
            propdensity = target(prop[j,:])
            alpha = (propdensity/curdensity)**tau[j]
            if alpha > U[j]:
                cur[j,:] = prop[j,:]
                if j == 0:
                    accepts += 1
        
        #Consider swapping positions between the two chains
        #Swap is random in keeping with traditional parallel tempering
        swap = np.random.rand(1)[0]
        chains = np.random.choice(k, 2, replace=False)
        alpha3 = ((target(cur[chains[1],:])/target(cur[chains[0],:]))**tau[chains[0]])*((target(cur[chains[0],:])/target(cur[chains[1],:]))**tau[chains[1]])
        if swap < alpha3:
            temp = cur[chains[0],:]
            cur[chains[0],:] = cur[chains[1],:]
            cur[chains[1],:] = temp

        
        samps[i,:] = cur[0,:]
    
    print(accepts/(M+B))
    
    return (samps[(B+1):,:]) # change to not a list
    