# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:45:10 2020

@author: ameer
"""

import autograd.numpy as np
import autograd.scipy.stats as ss

def u3(x, ban=1):
    xnew = np.array(x, copy=True).reshape((-1,2))
    newcol = xnew[:,1] + ban*xnew[:,0]**2 - ban
    xnew2 = np.hstack((xnew[:,0].reshape(-1,1),newcol.reshape(-1,1)))
    return ss.multivariate_normal.pdf(xnew2, mean=[0,0], cov=np.diag([9,4]))

def u4(x, ban=1):
    xnew = np.array(x, copy=True).reshape((-1,2))
    newcol = xnew[:,1] + ban*xnew[:,0]**2 - ban
    newcol2 = xnew[:,1] - ban*xnew[:,0]**2 + ban
    xnew2 = np.hstack((xnew[:,0].reshape(-1,1),newcol.reshape(-1,1)))
    xnew3 = np.hstack((xnew[:,0].reshape(-1,1),newcol2.reshape(-1,1)))
    return (ss.multivariate_normal.pdf(xnew2, mean=[0,0], cov=np.diag([9,4])) +
            ss.multivariate_normal.pdf(xnew3, mean=[0,-50], cov=np.diag([9,4])))

def u5(x):
    k = 10
    return(ss.multivariate_normal.pdf(x, mean=[k,0,0,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[-k,0,0,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,k,0,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,-k,0,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,0,k,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,0,-k,0], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,0,0,k], cov=np.diag([1,1,1,1])) +
            ss.multivariate_normal.pdf(x, mean=[0,0,0,-k], cov=np.diag([1,1,1,1])))

def u7(x, ban=1):
    xnew = np.array(x, copy=True).reshape((-1,2))
    newcol = xnew[:,1] + ban*xnew[:,0]**2 - ban
    newcol2 = xnew[:,1] - ban*xnew[:,0]**2 + ban
    newcol3 = xnew[:,0] + ban*xnew[:,1]**2 - ban
    newcol4 = xnew[:,0] - ban*xnew[:,1]**2 + ban
    xnew2 = np.hstack((xnew[:,0].reshape(-1,1),newcol.reshape(-1,1)))
    xnew3 = np.hstack((xnew[:,0].reshape(-1,1),newcol2.reshape(-1,1)))
    xnew4 = np.hstack((newcol3.reshape(-1,1), xnew[:,1].reshape(-1,1)))
    xnew5 = np.hstack((newcol4.reshape(-1,1), xnew[:,1].reshape(-1,1)))
    return (ss.multivariate_normal.pdf(xnew2, mean=[0,40], cov=np.diag([9,4])) +
            ss.multivariate_normal.pdf(xnew3, mean=[0,-40], cov=np.diag([9,4])) +
            ss.multivariate_normal.pdf(xnew4, mean=[40,0], cov=np.diag([4,9])) +
            ss.multivariate_normal.pdf(xnew5, mean=[-40,0], cov=np.diag([4,9])))

def u8(x, ban=1):
    xnew = np.array(x, copy=True).reshape((-1,3))
    
    newcol1 = xnew[:,0] + ban*xnew[:,1]**2 - ban
    newcol2 = xnew[:,0] + ban*xnew[:,2]**2 - ban
    xnew1 = np.hstack((newcol1.reshape(-1,1), xnew[:,1].reshape(-1,1), xnew[:,2].reshape(-1,1)))
    xnew2 = np.hstack((newcol2.reshape(-1,1), xnew[:,1].reshape(-1,1), xnew[:,2].reshape(-1,1)))
    newcol3 = xnew[:,0] - ban*xnew[:,1]**2 + ban
    newcol4 = xnew[:,0] - ban*xnew[:,2]**2 + ban
    xnew3 = np.hstack((newcol3.reshape(-1,1), xnew[:,1].reshape(-1,1), xnew[:,2].reshape(-1,1)))
    xnew4 = np.hstack((newcol4.reshape(-1,1), xnew[:,1].reshape(-1,1), xnew[:,2].reshape(-1,1)))

    newcol5 = xnew[:,1] + ban*xnew[:,0]**2 - ban
    newcol6 = xnew[:,1] + ban*xnew[:,2]**2 - ban
    xnew5 = np.hstack((xnew[:,0].reshape(-1,1), newcol5.reshape(-1,1), xnew[:,2].reshape(-1,1)))
    xnew6 = np.hstack((xnew[:,0].reshape(-1,1), newcol6.reshape(-1,1), xnew[:,2].reshape(-1,1)))
    newcol7 = xnew[:,1] - ban*xnew[:,0]**2 + ban
    newcol8 = xnew[:,1] - ban*xnew[:,2]**2 + ban
    xnew7 = np.hstack((xnew[:,0].reshape(-1,1), newcol7.reshape(-1,1), xnew[:,2].reshape(-1,1)))
    xnew8 = np.hstack((xnew[:,0].reshape(-1,1), newcol8.reshape(-1,1), xnew[:,2].reshape(-1,1)))

    newcol9 = xnew[:,2] + ban*xnew[:,0]**2 - ban
    newcol10 = xnew[:,2] + ban*xnew[:,1]**2 - ban
    xnew9 = np.hstack((xnew[:,0].reshape(-1,1), xnew[:,1].reshape(-1,1), newcol9.reshape(-1,1)))
    xnew10 = np.hstack((xnew[:,0].reshape(-1,1), xnew[:,1].reshape(-1,1), newcol10.reshape(-1,1)))
    newcol11 = xnew[:,2] - ban*xnew[:,0]**2 + ban
    newcol12 = xnew[:,2] - ban*xnew[:,1]**2 + ban
    xnew11 = np.hstack((xnew[:,0].reshape(-1,1), xnew[:,1].reshape(-1,1), newcol11.reshape(-1,1)))
    xnew12 = np.hstack((xnew[:,0].reshape(-1,1), xnew[:,1].reshape(-1,1), newcol12.reshape(-1,1)))
    
    return (ss.multivariate_normal.pdf(xnew1, mean=[40,0,0], cov=np.diag([4,9,4])) +
            ss.multivariate_normal.pdf(xnew2, mean=[40,0,0], cov=np.diag([4,4,9])) +
            ss.multivariate_normal.pdf(xnew3, mean=[-40,0,0], cov=np.diag([4,9,4])) +
            ss.multivariate_normal.pdf(xnew4, mean=[-40,0,0], cov=np.diag([4,4,9])) +
            
            ss.multivariate_normal.pdf(xnew5, mean=[0,40,0], cov=np.diag([9,4,4])) +
            ss.multivariate_normal.pdf(xnew6, mean=[0,40,0], cov=np.diag([4,4,9])) +
            ss.multivariate_normal.pdf(xnew7, mean=[0,-40,0], cov=np.diag([9,4,4])) +
            ss.multivariate_normal.pdf(xnew8, mean=[0,-40,0], cov=np.diag([4,4,9])) +
            
            ss.multivariate_normal.pdf(xnew9, mean=[0,0,40], cov=np.diag([9,4,4])) +
            ss.multivariate_normal.pdf(xnew10, mean=[0,0,40], cov=np.diag([4,9,4])) +
            ss.multivariate_normal.pdf(xnew11, mean=[0,0,-40], cov=np.diag([9,4,4])) +
            ss.multivariate_normal.pdf(xnew12, mean=[0,0,-40], cov=np.diag([4,9,4])))

