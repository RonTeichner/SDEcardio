import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde
import sys
import scipy
import pickle
import datetime
import black_box as bb
from cardio_func import *

# simulation parameters:
simDuration = 60*10  # sec
batch_size = 1
enableInputWorkload = True

# create a dataset of patients, each patient has a low and high workload profile and a trajectory at each profile:
fileName = 'DoylePatientsHugeDataset.pt'
enableSaveFile = True
nPatients = 100

disableNoise = False
disableController = False
disableWorkloadVar = True

DoylePatients = list()

for p in range(nPatients):
    print(f'starting creating patient {p+1} out of {nPatients}')
    print(datetime.datetime.now())
    # creating two profiles of the same patient, one for each workload:
    DoylePatient = createRandomPatient()

    if disableNoise:
        for i in range(len(DoylePatient)):
            DoylePatient[i].noiseStd = torch.zeros_like(DoylePatient[i].noiseStd)

    if disableController:
        for i in range(len(DoylePatient)):
            DoylePatient[i].enableController = False

    # creating the starting point to be a fixed point of the system in order to match the figures in the article:
    x_0 = list()
    x_0.append(DoylePatient[0].referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0])
    x_0.append(DoylePatient[1].referenceValues["x_L"][None, :, :].repeat(batch_size, 1, 1)[:, :, 0])


    # creating the workload profile:
    workRef = list()
    workRef.append(DoylePatient[0].referenceValues["d_L"][None, :, :].repeat(batch_size, 1, 1))
    workRef.append(DoylePatient[1].referenceValues["d_L"][None, :, :].repeat(batch_size, 1, 1))
    d, d_tVec = generate_workload_profile(batch_size, simDuration, workRef, enableInputWorkload, disableWorkloadVar)

    # Initial state x0, the SDE is solved over the interval [tVec[0], tVec[-1]].
    # x_k will have shape (tVec.shape[0], batch_size, state_size)

    # running the simulation and obtaining the trajectories:

    with torch.no_grad():
        device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # failed to transform to GPU, not working on it anymore...
        DoylePatient[0].to(device)
        DoylePatient[1].to(device)

        if disableNoise:
            runsolveIvpRun(DoylePatient[0], x_0[0].to(device), d[0].to(device), d_tVec.to(device), simDuration)
            runsolveIvpRun(DoylePatient[1], x_0[1].to(device), d[1].to(device), d_tVec.to(device), simDuration)
        else:
            runSdeint(DoylePatient[0], x_0[0].to(device), d[0].to(device), d_tVec.to(device), simDuration)
            runSdeint(DoylePatient[1], x_0[1].to(device), d[1].to(device), d_tVec.to(device), simDuration)



    DoylePatients.append(DoylePatient)
if enableSaveFile:
    pickle.dump(DoylePatients, open(fileName, "wb"))

'''
# plotting the trajectories:
DoylePatients = pickle.load(open(fileName, "rb"))
patientsIndex = 3
DoylePatient = DoylePatients[patientsIndex]
DoylePatient[0].plot(DoylePatient[0].x_k_sde)
DoylePatient[1].plot(DoylePatient[1].x_k_sde)
plt.show()
'''
