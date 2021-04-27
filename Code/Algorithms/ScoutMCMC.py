# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 19:25:33 2020

@author: ameer
"""

import autograd.numpy as np
from autograd import grad
import autograd.scipy.stats as ss

def DMSampler(target, beta, init, M, B, gamma, sigma, n=10):
    dim = len(init)
    stepmax = 10
    threshold = stepmax/gamma
    
    #Output
    samps = np.zeros((M+B,dim))
    cov = np.zeros((M+B,dim,dim))
    
    accepts = 0
    cur = init 
    tgrad = grad(target)
    
    origin = np.zeros(dim)
    identity = np.eye(dim)
    chol = sigma*np.eye(dim)
    #maxdet = 0
    for i in range(M+B):
        #Generate proposal
        epsilon = np.random.multivariate_normal(origin, identity)
        prop = cur + np.matmul(chol, epsilon)
               
        #Compute quantities for acceptance and adaption
        U = np.random.rand(1)[0]
        curdensity = target(cur)
        propdensity = target(prop)
        #propgrad = tgrad(prop)
               
        #Compute the gradient used to adapt the Cholesky factor
        gradchol = np.diag(beta/np.diag(chol))
        
        for j in range(n):
            eps = np.random.multivariate_normal(origin, identity)
            if j == 0:
                eps = epsilon
            y = cur + np.matmul(chol, eps)
            ydens = target(y)
            ygrad = tgrad(y)
            if (ydens != 0) and (not np.isinf(1/ydens)):
                if (np.log(target(y)) - np.log(curdensity)) < 0:
                    gradchol += ((beta+1)/(n*ydens))*np.outer(ygrad,eps)
                else:
                    gradchol += (beta/(n*ydens))*np.outer(ygrad,eps)
                        
        #Remove upper element of Cholesky factor gradients to preserve triangular structure
        gradchol = np.tril(gradchol)
        
        #gcur = gamma
        if np.amax(gradchol) > threshold:
            gradchol = np.minimum(gradchol, threshold * np.ones(dim, dim))
        if np.amin(gradchol) < -threshold:
            gradchol = np.maximum(gradchol, -threshold * np.ones(dim, dim))
        #if np.linalg.det(gradchol) > detmax:
        #    gcur = (detmax/np.linalg.det(gradchol)) ** (1/dim)
        
        if np.linalg.det(chol + gamma*gradchol) == 0:
            print("Current:", cur)
            print("Cholesky:", chol)
            print("Proposal:", prop)
            print("Density:", target(prop))
            print("Inverse Density:", 1/target(prop))
            print("Gradient:", tgrad(prop))
            print("CGradient:", gradchol)
            print("Epsilon:", epsilon)
            break   

        #Accept proposal 
        # Note: This is done after computing gradients as the M-H rule depends on the gradient.
        # Also, the gradients are computed regardless of whether the point was accepted or not. Thus,
        # we can learn from accepts and rejects.
        alpha = (np.log(propdensity))-(np.log(curdensity))
        #print(gcur)
        #print(gcur*gradchol)
        #print(chol + gcur*gradchol)
        if alpha > np.log(U):
            cur = prop
            accepts += 1
        
        #Update parameters (step size controlled by gradient determinant)
        chol += gamma*gradchol
            
        samps[i,:] = cur
        cov[i,:,:] = np.matmul(chol,chol.T)
    #print(cur)
    #print(np.matmul(chol,chol.T))
    #print(maxdet)
    print(accepts/(M+B))
    return [samps[(B+1):,:], cov[(B+1):,:,:]]

def ScoutMCMC(target, beta, init, M, B, gamma, sigma, ssigma = 1, tau=0.1, n=10, snum=20):
    dim = len(init)
    stepmax = 10
    threshold = stepmax/gamma
    
    #Output
    samps = np.zeros((M+B,dim))
    scout = np.zeros((M+B,dim))
    cov = np.zeros((M+B,dim,dim))
    
    accepts = 0
    cur = init 
    scur = init
    tgrad = grad(target)
    
    origin = np.zeros(dim)
    identity = np.eye(dim)
    scov = ssigma*identity
    chol = sigma*np.eye(dim)
    
    for i in range(M+B):
        #Main chain proposal
        epsilon = np.random.multivariate_normal(origin, identity)
        prop = cur + np.matmul(chol, epsilon)
        #Scout chain proposal
        sprop = np.random.multivariate_normal(scur, scov)
        
        #Compute quantities for acceptance and adaption
        U = np.random.rand(2)
        curdensity = target(cur)
        propdensity = target(prop)
        
        #Compute the gradient used to adapt the Cholesky factor
        gradchol = np.diag(beta/np.diag(chol))
        
        for j in range(n):
            eps = np.random.multivariate_normal(origin, identity)
            if j == 0:
                eps = epsilon
            y = cur + np.matmul(chol, eps)
            ydens = target(y)
            ygrad = tgrad(y)
            if (ydens != 0) and (not np.isinf(1/ydens)):
                if (np.log(target(y)) - np.log(curdensity)) < 0:
                    gradchol += ((beta+1)/(n*ydens))*np.outer(ygrad,eps)
                else:
                    gradchol += (beta/(n*ydens))*np.outer(ygrad,eps)
                        
        #Remove upper element of Cholesky factor gradients to preserve triangular structure
        gradchol = np.tril(gradchol)
        
        #gcur = gamma
        if np.amax(gradchol) > threshold:
            gradchol = np.minimum(gradchol, threshold * np.ones((dim, dim)))
        if np.amin(gradchol) < -threshold:
            gradchol = np.maximum(gradchol, -threshold * np.ones(dim, dim))
        #if np.linalg.det(gradchol) > detmax:
        #    gcur = (detmax/np.linalg.det(gradchol)) ** (1/dim)
        
        if np.sum(np.isinf(gradchol)) != 0:
            print("Current:", cur)
            print("Cholesky:", chol)
            print("Proposal:", prop)
            print("Density:", target(prop))
            print("Inverse Density:", 1/target(prop))
            print("Gradient:", tgrad(prop))
            print("CGradient:", gradchol)
            print("Epsilon:", epsilon)
            break   
            
        #Accept proposals
        # Note: This is done after computing gradients as the M-H rule depends on the gradient.
        # Also, the gradients are computed regardless of whether the point was accepted or not. Thus,
        # we can learn from accepts and rejects.
        #forward = ss.multivariate_normal.logpdf(prop, cur, np.matmul(chol, chol.T), allow_singular = True)
        #backward = ss.multivariate_normal.logpdf(cur, prop, np.matmul(chol + gamma*gradchol, (chol + gamma*gradchol).T), allow_singular = True)
        alpha1 = (np.log(propdensity))-(np.log(curdensity))
        alpha2 = tau*(np.log(target(sprop))-np.log(target(scur)))
        
        if alpha1 > np.log(U[0]):
            cur = prop
            accepts += 1
            
        if alpha2 > np.log(U[1]):
            scur = sprop
            
        #Update parameters
        chol += gamma*gradchol
        
        #Consider swapping positions between the two chains every snum iterations
        #Swap is random in keeping with traditional parallel tempering
        if (i % snum == 0):
            swap = np.random.rand(1)[0]
            alpha3 = (target(scur)/target(cur))*((target(cur)/target(scur))**tau)
            if swap < alpha3:
                temp = cur
                cur = scur
                scur = temp
        
        samps[i,:] = cur
        scout[i,:] = scur
        cov[i,:,:] = np.matmul(chol,chol.T)
    #print(cur)
    #print(np.matmul(chol,chol.T))
    print(accepts/(M+B))
    return [samps[(B+1):,:], cov[(B+1):,:,:], scout[(B+1):,:]]

def DMfinite(target, adsamps, adcov, init, F):
    dim = len(init)
    n = adsamps.shape[0]
    
    #Output
    samps = np.zeros((F,dim))
    
    accepts = 0
    cur = init 
    advecs = adsamps - cur
    dists = np.zeros(n)
    for j in range(n):
        dists[j] = np.linalg.norm(advecs[j,:])
    idx = np.argwhere(dists == np.amin(dists))[-1,0]
    
    for i in range(F):
        prop = np.random.multivariate_normal(cur, adcov[idx,:,:])
    
        advecs = adsamps - prop
        dists2 = np.zeros(n)
        for j in range(n):
            dists2[j] = np.linalg.norm(advecs[j,:])
        idx2 = np.argwhere(dists == np.amin(dists))[-1,0]
    
        U = np.random.rand(1)[0]
        curdensity = target(cur)
        propdensity = target(prop)
        
        forward = ss.multivariate_normal.logpdf(prop, cur, adcov[idx,:,:], allow_singular = True)
        backward = ss.multivariate_normal.logpdf(cur, prop, adcov[idx2,:,:], allow_singular = True)
        alpha = (np.log(propdensity)+backward)-(np.log(curdensity)+forward)
        
        if alpha > np.log(U):
            cur = prop
            accepts += 1
            idx = idx2
            dists = dists2
            
        samps[i,:] = cur
    print(accepts/F)
    return samps
      
def Scoutfinite(target, adsamps, adcov, init, F, ssigma = 1, tau=0.1, snum=20):
    dim = len(init)
    n = adsamps.shape[0]
    
    #Output
    samps = np.zeros((F,dim))
    scout = np.zeros((F,dim))
    accepts = 0
    cur = init 
    scur = init
    
    identity = np.eye(dim)
    scov = ssigma*identity
    
    advecs = adsamps - cur
    dists = np.zeros(n)
    for j in range(n):
        dists[j] = np.linalg.norm(advecs[j,:])
    idx = np.argwhere(dists == np.amin(dists))[-1,0]
    
    for i in range(F):
        #Main chain proposal
        prop = np.random.multivariate_normal(cur, adcov[idx,:,:])
        #Scout chain proposal
        sprop = np.random.multivariate_normal(scur, scov)
    
        advecs = adsamps - prop
        dists2 = np.zeros(n)
        for j in range(n):
            dists2[j] = np.linalg.norm(advecs[j,:])
        idx2 = np.argwhere(dists == np.amin(dists))[-1,0]
    
        U = np.random.rand(2)
        curdensity = target(cur)
        propdensity = target(prop)
        
        forward = ss.multivariate_normal.logpdf(prop, cur, adcov[idx,:,:], allow_singular = True)
        backward = ss.multivariate_normal.logpdf(cur, prop, adcov[idx2,:,:], allow_singular = True)
        alpha1 = (np.log(propdensity)+backward)-(np.log(curdensity)+forward)
        alpha2 = tau*(np.log(target(sprop))-np.log(target(scur)))
        
        if alpha1 > np.log(U[0]):
            cur = prop
            accepts += 1
            idx = idx2
            dists = dists2
                    
        if alpha2 > np.log(U[1]):
            scur = sprop
            
        #Consider swapping positions between the two chains every snum iterations
        #Swap is random in keeping with traditional parallel tempering
        if (i % snum == 0):
            swap = np.random.rand(1)[0]
            alpha3 = (target(scur)/target(cur))*((target(cur)/target(scur))**tau)
            if swap < alpha3:
                temp = cur
                cur = scur
                scur = temp
        
        samps[i,:] = cur
        scout[i,:] = scur
    print(accepts/F)
    return samps  