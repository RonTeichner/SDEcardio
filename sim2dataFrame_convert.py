import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from cardio_func import *

simFileNames = ["DoylePatientsDataset.pt", "DoylePatientsDataset_noNoise.pt", "DoylePatientsDataset_noNoise_noControl.pt"]

patientId = 0
for f, fileName in enumerate(simFileNames):
    print('starting with file ' + fileName)
    DoyleData = pickle.load(open(fileName, "rb"))
    for p, patient in enumerate(DoyleData):
        print(f'starting patient {p+1} out of {len(DoyleData)} patients')
        for w, workLevel in enumerate(patient):
            fs = workLevel.paramsDict["displayParamsDict"]["fs"]
            x_k = workLevel.x_k_sde

            nSamples, nBatches = x_k.shape[0], x_k.shape[1]

            tVec = workLevel.getTvec(x_k).numpy()
            u_k = workLevel.reCalcControl(x_k).numpy()
            x_k = x_k.numpy()
            d_tVec, d = workLevel.d_tVec.numpy(), workLevel.d.numpy()

            # converting d to the same time axis as other variables
            dnew = np.zeros((nSamples, nBatches, 1))
            for b in range(nBatches):
                f = interpolate.interp1d(d_tVec, d[:, b, 0])
                dnew[:, b, 0] = f(tVec)  # dnew is on the same time axis as all measurements

            # create c_k:
            q_as, q_o2, q_H = workLevel.paramsDict["controlParamsDict"]["q_as"], workLevel.paramsDict["controlParamsDict"]["q_o2"], workLevel.paramsDict["controlParamsDict"]["q_H"]
            Pas_set, O2_set, H_set = workLevel.referenceValues["x_L"][0].numpy(), workLevel.referenceValues["x_L"][3].numpy(), workLevel.referenceValues["u_L"].numpy()
            Pas_k, O2_k = x_k[:, :, 0:1], x_k[:, :, 3:4]
            c_k = np.power(q_as, 2)*np.power(Pas_k - Pas_set, 2) + np.power(q_o2, 2)*np.power(O2_k - O2_set, 2) + np.power(q_H, 2)*np.power(u_k - H_set, 2)
            Q_k = np.power(q_as, 2)*np.power(Pas_k - Pas_set, 2) + np.power(q_o2, 2)*np.power(O2_k - O2_set, 2)

            tilde_q_as, tilde_q_o2, tilde_q_H = q_as/Pas_set[0], q_o2/O2_set[0], q_H/H_set[0, 0]
            relativeDenominator = np.power(tilde_q_as, 2) + np.power(tilde_q_o2, 2) + np.power(tilde_q_H, 2)
            relative_q_as, relative_q_o2, relative_q_H = np.power(tilde_q_as, 2)/relativeDenominator , np.power(tilde_q_o2, 2)/relativeDenominator, np.power(tilde_q_H, 2)/relativeDenominator
            # stuck everything into an ndarray
            for b in range(nBatches):
                patientId = patientId + 1
                patientIdVec = patientId*np.ones_like(tVec)

                SigMatDataArraySingleBatch = np.concatenate((dnew[:, b], x_k[:, b, :], u_k[:, b], c_k[:, b], Q_k[:, b]), axis=1)[:, None, :]
                MetaDataDataArraySingleBatch = np.array([q_as, q_o2, q_H, relative_q_as, relative_q_o2, relative_q_H])[None, :]

                patientDataArraySingleBatch_originalFields = np.concatenate((patientIdVec[:, None], tVec[:, None], dnew[:, b], x_k[:, b, :], u_k[:, b], c_k[:, b], Q_k[:, b]), axis=1)
                # ID, time, workload, Pas, Pvs, Pap, O2, HR, C, Q

                patientDataArraySingleBatch_AseelsFields = np.concatenate((tVec[:, None], u_k[:, b], Q_k[:, b], x_k[:, b, 1:2], x_k[:, b, 0:1], x_k[:, b, 2:3], dnew[:, b], patientIdVec[:, None], c_k[:, b], x_k[:, b, 3:4]), axis=1)
                # 'Time', 'HR_electrical' <= HR, 'HR_mechanical' <= Q_k, 'DBP' <= Pvs, 'MBP' <= Pas, 'SBP' <= Pap, 'RR' <= workload, 'ID', 'PP' <= Combination, 'CO' <= O2

                # downsample to fs = 1:
                SigMatDataArraySingleBatch = SigMatDataArraySingleBatch[::fs]
                patientDataArraySingleBatch_originalFields, patientDataArraySingleBatch_AseelsFields = patientDataArraySingleBatch_originalFields[::fs], patientDataArraySingleBatch_AseelsFields[::fs]
                if b == 0:
                    patientDataArray_originalFields, patientDataArray_AseelsFields = patientDataArraySingleBatch_originalFields, patientDataArraySingleBatch_AseelsFields
                    SigMatDataArray = SigMatDataArraySingleBatch
                    MetaDataDataArray = MetaDataDataArraySingleBatch
                else:
                    patientDataArray_originalFields, patientDataArray_AseelsFields = np.concatenate((patientDataArray_originalFields, patientDataArraySingleBatch_originalFields), axis=0), np.concatenate((patientDataArray_AseelsFields, patientDataArraySingleBatch_AseelsFields), axis=0)
                    SigMatDataArray = np.concatenate((SigMatDataArray, SigMatDataArraySingleBatch), axis=1)
                    MetaDataDataArray = np.concatenate((MetaDataDataArray, MetaDataDataArraySingleBatch), axis=0)

            if w == 0:
                if p == 0:
                    patientsLowDataArray_originalFields, patientsLowDataArray_AseelsFields = patientDataArray_originalFields, patientDataArray_AseelsFields
                    SigMatLowDataArray = SigMatDataArray
                    MetaDataLowDataArray = MetaDataDataArray
                else:
                    patientsLowDataArray_originalFields, patientsLowDataArray_AseelsFields = np.concatenate((patientsLowDataArray_originalFields, patientDataArray_originalFields), axis=0), np.concatenate((patientsLowDataArray_AseelsFields, patientDataArray_AseelsFields), axis=0)
                    SigMatLowDataArray = np.concatenate((SigMatLowDataArray, SigMatDataArray), axis=1)
                    MetaDataLowDataArray = np.concatenate((MetaDataLowDataArray, MetaDataDataArray), axis=0)
            elif w == 1:
                if p == 0:
                    patientsHighDataArray_originalFields, patientsHighDataArray_AseelsFields = patientDataArray_originalFields, patientDataArray_AseelsFields
                    SigMatHighDataArray = SigMatDataArray
                    MetaDataHighDataArray = MetaDataDataArray
                else:
                    patientsHighDataArray_originalFields, patientsHighDataArray_AseelsFields = np.concatenate((patientsHighDataArray_originalFields, patientDataArray_originalFields), axis=0), np.concatenate((patientsHighDataArray_AseelsFields, patientDataArray_AseelsFields), axis=0)
                    SigMatHighDataArray = np.concatenate((SigMatHighDataArray, SigMatDataArray), axis=1)
                    MetaDataHighDataArray = np.concatenate((MetaDataHighDataArray, MetaDataDataArray), axis=0)

    dfLow_originalFields = pd.DataFrame(data=patientsLowDataArray_originalFields, columns=['ID', 'Time', 'Workload', 'Pas', 'Pvs', 'Pap', 'O2', 'HR', 'C', 'Q'])
    dfHigh_originalFields = pd.DataFrame(data=patientsHighDataArray_originalFields, columns=['ID', 'Time', 'Workload', 'Pas', 'Pvs', 'Pap', 'O2', 'HR', 'C', 'Q'])

    dfLow_AseelsFields = pd.DataFrame(data=patientsLowDataArray_AseelsFields, columns=['Time', 'HR_electrical', 'HR_mechanical', 'DBP', 'MBP', 'SBP', 'RR', 'ID', 'PP', 'CO'])
    dfHigh_AseelsFields = pd.DataFrame(data=patientsHighDataArray_AseelsFields, columns=['Time', 'HR_electrical', 'HR_mechanical', 'DBP', 'MBP', 'SBP', 'RR', 'ID', 'PP', 'CO'])

    SigMatFeatureNames = ['workload', 'Pas', 'Pvs', 'Pap', 'O2v', 'HR', 'c_k', 'Q_k']
    SigMatFeatureUnits = ['watt', 'mmHg', 'mmHg', 'mmHg', 'L O2 / L blood', 'beats@sec', '', '']
    MetaDataFeatureNames = ['q_as', 'q_o2', 'q_H', 'rel_q_as', 'rel_q_o2', 'rel_q_H']
    SigMatdataTuple = ([SigMatLowDataArray, MetaDataLowDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames], [SigMatHighDataArray, MetaDataHighDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames])
    pickle.dump(SigMatdataTuple, open('ndarrays_' + fileName, "wb"))

    pickle.dump((dfLow_originalFields, dfHigh_originalFields), open('dataFrame_originalFields_' + fileName, "wb"))
    pickle.dump((dfLow_AseelsFields, dfHigh_AseelsFields), open('dataFrame_AseelsFields_' + fileName, "wb"))

print('finished')
#asd = pickle.load(open('./asdDataFrame.pt', "rb"))
