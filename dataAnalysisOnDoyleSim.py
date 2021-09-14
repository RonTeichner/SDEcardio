import numpy as np
import pickle
from dataAnalysis_func import *

simFileNames = ["ndarrays_DoylePatientsDataset_noNoise","ndarrays_DoylePatientsDataset_noNoise_noControl","ndarrays_DoylePatientsDataset"]
#simFileNames = ["ndarrays_DoylePatientsHugeDataset_noNoise", "ndarrays_DoylePatientsHugeDataset_noNoise_noControl", "ndarrays_DoylePatientsHugeDataset", "ndarrays_DoylePatientsHugeDataset_noControl"]

fs = 1
slidingWindowSize = int(60*fs)
if np.mod(slidingWindowSize, 2) == 0:
    slidingWindowSize = slidingWindowSize + 1
autoCorrMaxLag = int(30*fs)
slidingWindowsWingap = 1  #int(slidingWindowSize/2)
Arlags = 120

for simFileName in simFileNames:
    ndarrays_DoylePatientsDataset = pickle.load(open(simFileName + '.pt', "rb"))
    SigMatLowDataArray, MetaDataLowDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames = ndarrays_DoylePatientsDataset[0]
    SigMatHighDataArray, MetaDataHighDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames = ndarrays_DoylePatientsDataset[1]

    dirName = simFileName

    ##################
    figuresDirName = dirName + '/lowWorkload'
    print('starting ' + figuresDirName)
    SigMat = SigMatLowDataArray
    SigMatFeatureNames = SigMatFeatureNames
    SigMatFeatureUnits = SigMatFeatureUnits
    P = SigMat.shape[1]
    PatientIds = np.arange(1, 1 + P).tolist()
    PatientClassification = ['Low'] * P
    MetaData = MetaDataLowDataArray
    MetaDataFeatureNames = MetaDataFeatureNames
    patientMetaDataTextBox = ''
    dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, Arlags, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

    ##################
    figuresDirName = dirName + '/highWorkload'
    print('starting ' + figuresDirName)
    SigMat = SigMatHighDataArray
    SigMatFeatureNames = SigMatFeatureNames
    SigMatFeatureUnits = SigMatFeatureUnits
    P = SigMat.shape[1]
    PatientIds = np.arange(1, 1 + P).tolist()
    PatientClassification = ['High'] * P
    MetaData = MetaDataHighDataArray
    MetaDataFeatureNames = MetaDataFeatureNames
    patientMetaDataTextBox = ''
    dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, Arlags, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)
