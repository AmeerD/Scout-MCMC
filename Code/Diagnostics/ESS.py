# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:45:10 2020

@author: ameer
"""

from statsmodels.tsa.stattools import acf
import numpy as np

#uESS computes the ESS for univariate chains
def uESS(chain, lags=50, increment = 10, threshold = 0.05):
    n = len(chain)
    autocor = acf(chain, nlags=lags, fft=False)
    #Set a check to see if enough lags have been computed
    while np.any(autocor[-increment:] > threshold):
        lags += increment
        autocor = acf(chain, nlags=lags, fft=False)
    cumacf = 2*np.sum(autocor[abs(autocor) > threshold]) - 1
    return n/cumacf

#mESS computes the multivariate ESS for multivariate chains in accordance with
# https://arxiv.org/pdf/1512.07713.pdf
#chain is a numpy array of size nxd where n is the number of samples
# and d is the dimension of the target function
def mESS(chain):
    n = chain.shape[0]
    d = chain.shape[1]
    
    #b is the batch size
    #a is the number of batches
    #If the chain length isn't a perfect square, the final batch will be 
    # shorter
    b = int(np.floor(np.sqrt(n)))
    a = int(np.ceil(n/b))
    
    #Expected Value of the chain
    est = np.mean(chain, axis=0)
    
    alpha = np.cov(chain, rowvar=False)
    sigma = np.zeros((d,d))
    
    for i in range(a):
        #Compute the batch mean and variance contribution
        bmean = np.mean(chain[(i*b):((i+1)*b),:], axis=0)
        bias = bmean - est
        varcont = np.outer(bias,bias)
        sigma += varcont
        
    sigma = sigma * b/(a-1)
    
    return n*((np.linalg.det(alpha)/np.linalg.det(sigma))**(1/d))

