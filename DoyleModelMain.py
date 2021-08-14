import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde
import sys
import scipy
import pickle
import black_box as bb
from cardio_func import *


simDuration = 400  # sec
batch_size = 1
enablePinnedSearch = True
enableInputWorkload = True

dfs = 100  # [hz]
dVal = 53.3  # [watt]
d_tVec = torch.tensor(np.arange(0, np.ceil(simDuration*dfs))/dfs)
d = torch.zeros(d_tVec.shape[0], batch_size, 1, dtype=torch.float)
if enableInputWorkload:
    workStartTimes = np.array([33.17, 100, 225])  # [sec]
    workStopTimes = np.array([74.4, 160, 310])  # [sec]
    workStartIndexes = np.round(dfs*workStartTimes)
    workStopIndexes = np.round(dfs*workStopTimes)
    for i, startIndex in enumerate(workStartIndexes):
        stopIndex = workStopIndexes[i]
        d[int(startIndex):int(stopIndex)+1, :, 0] = dVal

    # Pinned values from figure S14:
    workPinnedTimes = np.array([28.43, 28.9, 29.383, 29.857, 33.17])
    workPinnedValues = np.array([0, 9.765, 28.544, 40.1877, 53.33])
    workStartIndexes = np.round(dfs*workPinnedTimes)
    workStopTimes = workPinnedTimes[1:]
    workStopIndexes = np.round(dfs*workStopTimes)
    for i, startIndex in enumerate(workStartIndexes):
        if i == 4:
            stopIndex = workStopIndexes[i-1]
            d[int(startIndex):int(stopIndex) + 1, :, 0] = torch.linspace(workPinnedValues[i], workPinnedValues[i], int(stopIndex) - int(startIndex) + 1)[:, None]
        else:
            stopIndex = workStopIndexes[i]
            d[int(startIndex):int(stopIndex) + 1, :, 0] = torch.linspace(workPinnedValues[i], workPinnedValues[i+1], int(stopIndex) - int(startIndex) + 1)[:, None]

# pinned state and control values from figure S14:
pinned_Times = torch.tensor([29.857, 40.758, 53.0805, 63.033, 72.51], dtype=torch.float)
pinned_HR = torch.tensor([57.464, 69.859, 80.375, 85.258, 86.76], dtype=torch.float)
pinned_Pas = torch.tensor([78.873, 95.774, 110.047, 117.183, 120.938], dtype=torch.float)
pinned_O2 = torch.tensor([150.61, 136.7136, 126.197, 120.5633, 117.5586], dtype=torch.float)

# subject 5:
# DoylePatientLow = DoyleSDE(d,     d_tVec, u_L=55,  d_L=0,   q_as=40, q_o2=1e5, q_H=1,  c_l=0.03, c_r=0.05)
# DoylePatientHigh =DoyleSDE(d+100, d_tVec, u_L=100, d_L=100, q_as=80, q_o2=1e5, q_H=40, c_l=0.03, c_r=0.05)

#  subject 3:
#DoylePatientLow = DoyleSDE(d,    d_tVec, u_L=45.8, d_L=0,  q_as=850, q_o2=2.3076e+06, q_H=1,  c_l=0.02, c_r=0.04)
DoylePatientLow = DoyleSDE(d,    d_tVec, u_L=45.8, d_L=0,  q_as=40, q_o2=1e5, q_H=5,  c_l=0.02, c_r=0.04)
DoylePatientHigh =DoyleSDE(d+55, d_tVec, u_L=90,   d_L=55, q_as=65, q_o2=1e5, q_H=15, c_l=0.02, c_r=0.04)

#DoylePatient.create_figure_S4()
state_size = DoylePatientLow.state_size

fs = DoylePatientLow.paramsDict["displayParamsDict"]["fs"]

#x_0 = torch.full((batch_size, state_size), 0.1)
x_0_Low = DoylePatientLow.referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0]
x_0_High = DoylePatientHigh.referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0]
tVec = torch.tensor(np.arange(0, np.ceil(simDuration*fs))/fs, dtype=torch.float)  # [sec]
simDuration = tVec.shape[0]/fs  # [sec]

sys.setrecursionlimit(10000) # this is to enable a long simulation
# Initial state x0, the SDE is solved over the interval [tVec[0], tVec[-1]].
# x_k will have shape (tVec.shape[0], batch_size, state_size)

#DoylePatientLow.enableController = False
#DoylePatientHigh.enableController = False

with torch.no_grad():
    x_k_Low = DoylePatientLow.runSolveIvp(x_0_Low, simDuration)
    u_k_Low = DoylePatientLow.reCalcControl(x_k_Low)
    tVec_Low = DoylePatientLow.getTvec(x_k_Low)
    x_k_High = DoylePatientHigh.runSolveIvp(x_0_High, simDuration)
    #x_k = torchsde.sdeint(DoylePatient, x_0, tVec)

DoylePatientLow.plot(x_k_Low)
DoylePatientHigh.plot(x_k_High)

plt.show()

if enablePinnedSearch:
    # searching:
    DoylePatient = DoyleSDE(d, d_tVec, u_L=45.8, d_L=0, q_as=7.9951e+02, q_o2=1e5, q_H=1, c_l=0.02, c_r=0.04, pinnedList=[pinned_Times, pinned_HR, pinned_Pas, pinned_O2, simDuration])
    DoylePatient.hyperParamSearchCounter = 0

    best_params = bb.search_min(f = DoylePatient.controlParamsOptimize,  # given function
                                domain = [  # ranges of each parameter
                                    [0, 1000], # q_as
                                    [1e5/5, 1e7],  # q_o2, and q_H=1
                                    [0.02, 0.02],  # c_l
                                    [0.04, 0.04]  #  c_r
                                    ],
                                budget = 100,  # total number of function calls available
                                batch = 12,  # number of calls that will be evaluated in parallel
                                resfile = 'output.csv')  # text file where results will be saved

    #best_params_scipy = scipy.optimize.minimize(fun=DoylePatient.controlParamsOptimizeSciPy, x0=np.array([40.0, 1e5]))

'''
q_as_vals = np.linspace(0, 100, 101)
q_o2_vals = np.linspace(1e4, 1e8, 100)
q_H_vals = np.linspace(1, 100, 100)

scoresList = list()
scoresParams = list()
for q_as in q_as_vals:
    for q_o2 in q_o2_vals:
        print('start q_o2')
        for q_H in q_H_vals:
            DoylePatient = DoyleSDE(d, d_tVec, u_L=45.8, d_L=0, q_as=q_as, q_o2=q_o2, q_H=q_H, c_l=0.02, c_r=0.04, pinnedList=[pinned_Times, pinned_HR, pinned_Pas, pinned_O2, simDuration])
            with torch.no_grad():
                x_k = DoylePatient.runSolveIvp(x_0_Low, simDuration)
                u_k = DoylePatient.reCalcControl(x_k)
                tVec = DoylePatient.getTvec(x_k)
            score = calcScore(x_k, u_k, tVec, pinned_Times, pinned_HR, pinned_Pas, pinned_O2)
            scoresList.append(score)
            scoresParams.append((q_as, q_o2, q_H))

pickle.dump( (scoresList, scoresParams), open( "saveScores.pt", "wb" ) )
scoresTupple = pickle.load( open( "saveScores.pt", "rb" ) )
'''

