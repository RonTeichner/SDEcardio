import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde
import sys
import scipy
import pickle
import black_box as bb
from cardio_func import *

# simulation parameters:
simDuration = 400  # sec
batch_size = 1
enableInputWorkload = True

# creating two profiles of the same patient, one for each workload:
DoylePatientLow = DoyleSDE(u_L=45.8, d_L=0,  q_as=40, q_o2=1e5, q_H=5,  c_l=0.02, c_r=0.04)
DoylePatientHigh =DoyleSDE(u_L=90,   d_L=55, q_as=65, q_o2=1e5, q_H=15, c_l=0.02, c_r=0.04)

# creating the starting point to be a fixed point of the system in order to match the figures in the article:
x_0_Low = DoylePatientLow.referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0]
x_0_High = DoylePatientHigh.referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0]
workRefLow = DoylePatientLow.referenceValues["d_L"][None, :, :].repeat(batch_size, 1, 1)
workRefHigh = DoylePatientHigh.referenceValues["d_L"][None, :, :].repeat(batch_size, 1, 1)

# creating the workload profile:
d, d_tVec = generate_workload_profile(batch_size, simDuration, enableInputWorkload)

# Initial state x0, the SDE is solved over the interval [tVec[0], tVec[-1]].
# x_k will have shape (tVec.shape[0], batch_size, state_size)

# running the simulation and obtaining the trajectories:
with torch.no_grad():
    x_k_Low = DoylePatientLow.runSolveIvp(x_0_Low, d + workRefLow, d_tVec, simDuration)
    #x_k_High = DoylePatientHigh.runSolveIvp(x_0_High, d + workRefHigh, d_tVec, simDuration)

    tVec = DoylePatientLow.getTvec(x_k_Low)

    x_k_Low_sde = runSdeint(DoylePatientLow, x_0_Low, d + workRefLow, d_tVec, simDuration)
    #sys.setrecursionlimit(10000)  # this is to enable a long simulation in torchsde
    #x_k_Low_sde = torchsde.sdeint(DoylePatientLow, x_0_Low, tVec)

# plotting the trajectories:
#DoylePatientLow.plot(x_k_Low)
#DoylePatientHigh.plot(x_k_High)

DoylePatientLow.plot(x_k_Low_sde)
plt.show()

