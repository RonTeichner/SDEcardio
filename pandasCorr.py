import pandas as pd
import numpy as np

def R_L(cj, cm, L):
    N = cj.shape[0]

    i0 = np.max((L, 0))
    i1 = N - 1 + np.min((0, L))

    cj, cm = cj[i0:i1+1], cm[i0-L:i1+1-L]

    cj_ValidIndexes, cm_ValidIndexes = np.logical_not(np.isnan(cj)), np.logical_not(np.isnan(cm))
    validIndexes = np.logical_and(cj_ValidIndexes, cm_ValidIndexes)

    cj_mean, cm_mean = cj[validIndexes].mean(), cm[validIndexes].mean()

    effectiveNorm_cj, effectiveNorm_cm = np.sqrt(np.power(cj[validIndexes] - cj_mean, 2).sum()), np.sqrt(np.power(cm[validIndexes] - cm_mean, 2).sum())
    dotProduct = np.dot(cj[validIndexes] - cj_mean, cm[validIndexes] - cm_mean)

    return dotProduct/(effectiveNorm_cm*effectiveNorm_cj)

x = np.array([0.25, 0.5, np.nan, -0.05])

s = pd.Series(x)
print(f'series mean manual calc: {(0.25+0.5-0.05)/3}, series pandas calc: {s.mean()}')
print(f'series var manual calc: {(np.power(0.25-s.mean(),2) + np.power(0.5-s.mean(),2) + np.power(-0.05-s.mean(),2))/2}, series pandas calc: {s.var()}')

N = 100
x = np.random.rand(N)
nanIndexes = [10,20,30,40,50,60,70,80,90]
#x[nanIndexes] = np.nan
lagValues = np.arange(-(N-2),N-1)
#lagValues = np.array([N-1-1])
maxDiff = 0.0
for lag in lagValues:
    autocorr_pd = pd.Series(x).autocorr(lag=lag)
    autoCorr_self = R_L(x, x, lag)  # This calculates R(L,j,m) with support for nan values.
    #print(f'my autocorr with lag = {lag} is {autoCorr_self}, pd autocorr is {autocorr_pd}')
    maxDiff = np.max((np.abs(autocorr_pd - autoCorr_self), maxDiff))
print(f'maxDiff between pandas autocorr and self calculated: {maxDiff}')




