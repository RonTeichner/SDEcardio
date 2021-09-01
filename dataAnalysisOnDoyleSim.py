import numpy as np
import pickle
from dataAnalysis_func import *

ndarrays_DoylePatientsDataset_noNoise = pickle.load(open('ndarrays_DoylePatientsDataset_noNoise.pt', "rb"))
ndarrays_DoylePatientsDataset_noNoise_noControl = pickle.load(open('ndarrays_DoylePatientsDataset_noNoise_noControl.pt', "rb"))

SigMatLowDataArray_noNoise, MetaDataLowDataArray_noNoise, SigMatFeatureNames_noNoise, SigMatFeatureUnits_noNoise, MetaDataFeatureNames_noNoise = ndarrays_DoylePatientsDataset_noNoise[0]
SigMatHighDataArray_noNoise, MetaDataHighDataArray_noNoise, SigMatFeatureNames_noNoise, SigMatFeatureUnits_noNoise, MetaDataFeatureNames_noNoise = ndarrays_DoylePatientsDataset_noNoise[1]

SigMatLowDataArray_noNoise_noControl, MetaDataLowDataArray_noNoise_noControl, SigMatFeatureNames_noNoise_noControl, SigMatFeatureUnits_noNoise_noControl, MetaDataFeatureNames_noNoise_noControl = ndarrays_DoylePatientsDataset_noNoise_noControl[0]
SigMatHighDataArray_noNoise_noControl, MetaDataHighDataArray_noNoise_noControl, SigMatFeatureNames_noNoise_noControl, SigMatFeatureUnits_noNoise_noControl, MetaDataFeatureNames_noNoise_noControl = ndarrays_DoylePatientsDataset_noNoise_noControl[1]

fs = 1
slidingWindowSize = int(30*fs)
if np.mod(slidingWindowSize, 2) == 0:
    slidingWindowSize = slidingWindowSize + 1
autoCorrMaxLag = int(30*fs)
slidingWindowsWingap = int(slidingWindowSize/2)

dirName = 'DoyleSimAnalysis'

##################3
figuresDirName = dirName + '/lowWorkload_noNoise'
print('starting ' + figuresDirName)
SigMat = SigMatLowDataArray_noNoise
SigMatFeatureNames = SigMatFeatureNames_noNoise
SigMatFeatureUnits = SigMatFeatureUnits_noNoise
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low']*P
MetaData = MetaDataLowDataArray_noNoise
MetaDataFeatureNames = MetaDataFeatureNames_noNoise
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/highWorkload_noNoise'
print('starting ' + figuresDirName)
SigMat = SigMatHighDataArray_noNoise
MetaData = MetaDataHighDataArray_noNoise
SigMatFeatureNames = SigMatFeatureNames_noNoise
SigMatFeatureUnits = SigMatFeatureUnits_noNoise
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['High']*P
MetaDataFeatureNames = MetaDataFeatureNames_noNoise
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/lowWorkload_noNoise_noControl'
print('starting ' + figuresDirName)
SigMat = SigMatLowDataArray_noNoise
MetaData = MetaDataLowDataArray_noNoise_noControl
SigMatFeatureNames = SigMatLowDataArray_noNoise_noControl
SigMatFeatureUnits = SigMatFeatureNames_noNoise_noControl
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low_noControl']*P
MetaDataFeatureNames = MetaDataFeatureNames_noNoise_noControl
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + 'highWorkload_noNoise_noControl'
print('starting ' + figuresDirName)
SigMat = SigMatHighDataArray_noNoise_noControl
MetaData = MetaDataHighDataArray_noNoise_noControl
SigMatFeatureNames = SigMatFeatureNames_noNoise_noControl
SigMatFeatureUnits = SigMatFeatureUnits_noNoise_noControl
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['High_noControl']*P
MetaDataFeatureNames = MetaDataFeatureNames_noNoise_noControl
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)


