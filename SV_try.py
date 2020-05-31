# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:43:52 2020

@author: Daniel
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from pandas_datareader import data as ub
import matplotlib.pyplot as plt
from scipy.special import gammaln
from numpy.linalg import inv
from numpy.linalg import det
from numpy.random import rand, seed
from sim_m_SV import sim_m_SV



dfY = pd.read_csv('DAX_sep_index.csv')
dfY['pct_change']=dfY.price.pct_change()
dfY['log_ret']=np.log(dfY.price) - np.log(dfY.price.shift(1))
returnssep=dfY['log_ret']
returnssep=returnssep.dropna()
returnssep.reset_index(inplace=True,drop=True)
returnssep.to_frame()

# Calculate sample moments
T = len(returnssep)
returnssepa = abs(returnssep)
corr = returnssep.shift(1).corr(returnssep)
sample_m = np.array([np.var(returnssep),scipy.stats.kurtosis(returnssep), corr])
sample_m = sample_m.transpose()
#%%
seed(1234)
H=50*T
epsilon = rand(H)
eta = rand(H)
e = np.array([epsilon,eta])
e = e.T
#%%

## 2. Optimization Options

options ={'eps':1e-09, 'disp': True,'maxiter':200}

## 4. Initial Parameter Values

beta_ini = 0.9
omega_ini = 0.1
sig2f_ini = 0.1

theta_ini = np.array([omega_ini, beta_ini, sig2f_ini])


## 5. Optimize Log Likelihood Criterion

#  optim input:
# (1) negative log likelihood function: llik_fun_GARCH() note the minus when calculating the mean
# (2) initial parameter: theta_ini
# (3) parameter space bounds: lb & ub
# (4) optimization setup: control
#  Note: a number of parameter restriction are left empty with []

#  optim output:
# (1) parameter estimates: par
# (2) negative mean log likelihood value at theta_hat: value
# (3) exit flag indicating (no) convergence: convergence

results_SV = scipy.optimize.minimize(sim_m_SV, theta_ini, args=(returnssep),
                                  options = options,
                                  method='SLSQP',
                                  bounds=( (0,  1),
                                          (0.001,0.999), (0.001,0.999))
                                  ) #restrictions in parameter space



## 7. Print Output

print('parameter estimates:')
print(results_SV.x)

print('log likelihood value:')
print(results_SV.fun)

print('exit flag:')
print(results_SV.success)





