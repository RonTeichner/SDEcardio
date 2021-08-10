import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde
import sys
from cardio_func import *


simDuration = 400  # sec
batch_size = 1

enableInputWorkload = True

dfs = 1  # [hz]
dVal = 50  # [watt]
d_tVec = torch.tensor(np.arange(0, np.ceil(simDuration*dfs))/dfs)
d = torch.zeros(d_tVec.shape[0], batch_size, 1, dtype=torch.float)
if enableInputWorkload:
    workStartTimes = [22.5, 100, 225]  # [sec]
    workStopTimes = [65, 160, 310]  # [sec]
    workStartIndexes = np.round(dfs*workStartTimes)
    workStopIndexes = np.round(dfs*workStopTimes)
    for i, startIndex in enumerate(workStartIndexes):
        stopIndex = workStopIndexes[i]
        d[int(startIndex):int(stopIndex)+1, :, 0] = dVal

DoylePatient = DoyleSDE(d, d_tVec)
#DoylePatient.create_figure_S4()
state_size = DoylePatient.state_size

fs = DoylePatient.paramsDict["displayParamsDict"]["fs"]

#x_0 = torch.full((batch_size, state_size), 0.1)
x_0 = DoylePatient.referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0]
tVec = torch.tensor(np.arange(0, np.ceil(simDuration*fs))/fs, dtype=torch.float)  # [sec]
simDuration = tVec.shape[0]/fs  # [sec]

sys.setrecursionlimit(10000) # this is to enable a long simulation
# Initial state x0, the SDE is solved over the interval [tVec[0], tVec[-1]].
# x_k will have shape (tVec.shape[0], batch_size, state_size)


with torch.no_grad():
    x_k = DoylePatient.runSolveIvp(x_0, simDuration)
    #x_k = torchsde.sdeint(DoylePatient, x_0, tVec)

DoylePatient.plot(x_k)

plt.show()