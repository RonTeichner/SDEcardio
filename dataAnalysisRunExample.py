import numpy as np
from dataAnalysis_func import *

#np.random.seed(seed=2589629)

# create imaginary data set:
nTimePoints, nPatients, nFeatures = 100, 12, 3
nBatchesPerPatient = np.ndarray(nPatients, dtype=int)
for p in range(nPatients):
    nBatchesPerPatient[p] = np.random.randint(1,4,1)
nTotalPatientsBatches = int(nBatchesPerPatient.sum())

SigMat = np.random.randn(nTimePoints, nTotalPatientsBatches, nFeatures)
for f in range(SigMat.shape[2]):
    SigMat[:, :, f] = np.power(10, f) * SigMat[:, :, f]

for p in range(SigMat.shape[1]):
    nanIndexesTime = np.random.randint(0, nTimePoints, 5)
    #SigMat[nanIndexesTime] = np.nan

assert np.logical_not(np.isnan(SigMat)).any()

SigMatFeatureNames = ["bp", "hr", "rr"]
SigMatFeatureUnits = ["mmHg", "beats@min", "resp@min"]
PatientIds = np.ndarray(nTotalPatientsBatches, dtype=int)
startIndex = 0
for p in range(nPatients):
    PatientIds[startIndex:startIndex+nBatchesPerPatient[p]] = p
    startIndex = startIndex+nBatchesPerPatient[p]

PatientIds = PatientIds.tolist()
PatientClassification = ["control"]*nBatchesPerPatient[:int(nPatients/2)].sum() + ["cardio"]*nBatchesPerPatient[int(nPatients/2):].sum()

nMetaDataFeatures = 2
MetaData = 100*np.random.randn(SigMat.shape[1], nMetaDataFeatures)

MetaDataFeatureNames = ["age", "weight"]

fs = 2  # hz

patientMetaDataTextBox = ["age", "weight"]
slidingWindowSize = 11 # window size of autocorr
autoCorrMaxLag = 5
slidingWindowsWingap = int(slidingWindowSize/2)
figuresDirName = "exampleDataAnalysis"
#np.seterr(all='raise')
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)
plt.close()