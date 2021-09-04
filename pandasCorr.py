import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def N_C(cj, cm, L):
    N = cj.shape[0]

    i0 = np.max((L, 0))
    i1 = N - 1 + np.min((0, L))

    cj, cm = cj[i0:i1+1], cm[i0-L:i1+1-L]

    cj_ValidIndexes, cm_ValidIndexes = np.logical_not(np.isnan(cj)), np.logical_not(np.isnan(cm))
    validIndexes = np.logical_and(cj_ValidIndexes, cm_ValidIndexes)

    effectiveNorm_cj, effectiveNorm_cm = np.sqrt(np.power(cj[validIndexes], 2).sum()), np.sqrt(np.power(cm[validIndexes], 2).sum())
    dotProduct = np.dot(cj[validIndexes], cm[validIndexes])

    return dotProduct/(effectiveNorm_cm*effectiveNorm_cj)

x = np.array([0.25, 0.5, np.nan, -0.05])

s = pd.Series(x)
print(f'series mean manual calc: {(0.25+0.5-0.05)/3}, series pandas calc: {s.mean()}')
print(f'series var manual calc: {(np.power(0.25-s.mean(),2) + np.power(0.5-s.mean(),2) + np.power(-0.05-s.mean(),2))/2}, series pandas calc: {s.var()}')

N = 100
x = np.random.rand(N)
nanIndexes = [10,20,30,40,50,60,70,80,90]
x[nanIndexes] = np.nan
lagValues = np.arange(-(N-2),N-1)
#lagValues = np.array([N-1-1])
maxDiffAutoCorr, maxDiffNormalizedCorr = 0.0, 0.0
for lag in lagValues:
    autocorr_pd = pd.Series(x).autocorr(lag=lag)
    if np.isnan(x).any() and np.logical_not(np.isnan(autocorr_pd)):
        print('nan values in signal but autocorr res is not nan')
    autoCorr_self = R_L(x, x, lag)  # This calculates R(L,j,m) with support for nan values.
    #print(f'my autocorr with lag = {lag} is {autoCorr_self}, pd autocorr is {autocorr_pd}')
    maxDiffAutoCorr = np.max((np.abs(autocorr_pd - autoCorr_self), maxDiffAutoCorr))

    normalizedCorr_self = N_C(x, x, lag)
    #normalizedCorr_pd = N_C_pd(x, x, lag)
    #maxDiffNormalizedCorr = np.max((np.abs(normalizedCorr_pd - normalizedCorr_self), maxDiffNormalizedCorr))
print(f'maxDiff between pandas autocorr and self calculated: {maxDiffAutoCorr}')
#print(f'maxDiff between pandas corr and self calculated: {maxDiffNormalizedCorr}')

x = np.random.randn(100, 5, 3)
x[:,:,1] = x[:,:,1]*1000
x[:,:,2] = x[:,:,2]*1000
nanIndexes = [10,20,30,40,50,60,70,80,90]
x[nanIndexes] = np.nan

#hist, binEdges = np.histogram(x.reshape((100, -1))[:, 2], bins=100, density=True)
#cdf = np.cumsum(hist)

plt.figure()
n_bins = 1000
feature0, feature1 = x.reshape((100, -1))[:, 2], x.reshape((100, -1))[:, 1]
feature0, feature1 = feature0[np.logical_not(np.isnan(feature0))], feature1[np.logical_not(np.isnan(feature1))]
n, bins, _ = plt.hist([feature0, feature1], n_bins, histtype='step', density=True, cumulative=True, label='hist')
plt.clf() # eliminates the matplotlib plot
bins = bins[:-1]

plt.xlabel(r'$\sigma_e^2$ [dbW]')
plt.grid()

# A text mixing normal text and math text.
msg = (r"Normal Text. $Text\ in\ math\ mode:\ "r"\int_{0}^{\infty } x^2 dx$")

# Set the text in the plot.
plt.plot(bins, n[0, :], label='me_0')
plt.plot(bins, n[1, :], label='me_1')
plt.text(0.1, 0.9, 'matplotlib')#, horizontalalignment='center', verticalalignment='center')#, transform=ax.transAxes)
plt.legend()
plt.show()




