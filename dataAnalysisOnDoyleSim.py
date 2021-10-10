import numpy as np
import pickle
from dataAnalysis_func import *

simFileNames = ["ndarrays_DoylePatientsDataset_noNoise","ndarrays_DoylePatientsDataset_noNoise_noControl","ndarrays_DoylePatientsDataset"]
#simFileNames = ["ndarrays_DoylePatientsHugeDataset", "ndarrays_DoylePatientsHugeDataset_noControl", "ndarrays_DoylePatientsHugeDataset_noNoise", "ndarrays_DoylePatientsHugeDataset_noNoise_noControl"]

fs = 1
slidingWindowSize = 60
if np.mod(slidingWindowSize, 2) == 0:
    slidingWindowSize = slidingWindowSize + 1
autoCorrMaxLag = 30  # sec
slidingWindowsWingap = 1  #int(slidingWindowSize/2)
Arlags = 120
paramsDict = {"fs": fs, "slidingWindowSize": slidingWindowSize, "slidingWindowsWingap": slidingWindowsWingap, "autoCorrMaxLag": autoCorrMaxLag, "Arlags": Arlags, "analysisStartTime": 100, "analysisEndTime": np.inf}

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
    nBatchesPerPatient = np.ones(P)
    PatientClassification = ['Low'] * P
    MetaData = MetaDataLowDataArray
    MetaDataFeatureNames = MetaDataFeatureNames
    patientMetaDataTextBox = ''
    patientsDf = SigMat2Df(SigMat, fs, SigMatFeatureNames, PatientIds, nBatchesPerPatient)
    metaDataDf = MetaData2Df(MetaData, MetaDataFeatureNames, PatientClassification, PatientIds)
    dataAnalysis(paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, figuresDirName)

    ##################
    figuresDirName = dirName + '/highWorkload'
    print('starting ' + figuresDirName)
    SigMat = SigMatHighDataArray
    SigMatFeatureNames = SigMatFeatureNames
    SigMatFeatureUnits = SigMatFeatureUnits
    P = SigMat.shape[1]
    PatientIds = np.arange(1, 1 + P).tolist()
    nBatchesPerPatient = np.ones(P)
    PatientClassification = ['High'] * P
    MetaData = MetaDataHighDataArray
    MetaDataFeatureNames = MetaDataFeatureNames
    patientMetaDataTextBox = ''
    patientsDf = SigMat2Df(SigMat, fs, SigMatFeatureNames, PatientIds, nBatchesPerPatient)
    metaDataDf = MetaData2Df(MetaData, MetaDataFeatureNames, PatientClassification, PatientIds)
    dataAnalysis(paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, figuresDirName)
