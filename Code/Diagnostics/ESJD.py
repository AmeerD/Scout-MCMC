# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:11:59 2020

@author: ameer
"""

import numpy as np

def esjd(samps):
    n = samps.shape[0]
    ans = 0
    for i in range(1,n):
        dist = np.linalg.norm(samps[i,:]-samps[i-1,:])
        ans += dist**2
    return ans/(n-1)