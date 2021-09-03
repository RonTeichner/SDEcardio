import numpy as np
import pickle
from dataAnalysis_func import *

ndarrays_DoylePatientsDataset = pickle.load(open('ndarrays_DoylePatientsDataset.pt', "rb"))
ndarrays_DoylePatientsDataset_noNoise = pickle.load(open('ndarrays_DoylePatientsDataset_noNoise.pt', "rb"))
ndarrays_DoylePatientsDataset_noNoise_noControl = pickle.load(open('ndarrays_DoylePatientsDataset_noNoise_noControl.pt', "rb"))

ndarrays_DoylePatientsHugeDataset_noNoise = pickle.load(open('ndarrays_DoylePatientsHugeDataset_noNoise.pt', "rb"))
ndarrays_DoylePatientsHugeDataset_noNoise_noController = pickle.load(open('ndarrays_DoylePatientsHugeDataset_noNoise_noController.pt', "rb"))


SigMatLowDataArray, MetaDataLowDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames = ndarrays_DoylePatientsDataset[0]
SigMatHighDataArray, MetaDataHighDataArray, SigMatFeatureNames, SigMatFeatureUnits, MetaDataFeatureNames = ndarrays_DoylePatientsDataset[1]

SigMatLowDataArray_noNoise, MetaDataLowDataArray_noNoise, SigMatFeatureNames_noNoise, SigMatFeatureUnits_noNoise, MetaDataFeatureNames_noNoise = ndarrays_DoylePatientsDataset_noNoise[0]
SigMatHighDataArray_noNoise, MetaDataHighDataArray_noNoise, SigMatFeatureNames_noNoise, SigMatFeatureUnits_noNoise, MetaDataFeatureNames_noNoise = ndarrays_DoylePatientsDataset_noNoise[1]

SigMatLowDataArray_noNoise_noControl, MetaDataLowDataArray_noNoise_noControl, SigMatFeatureNames_noNoise_noControl, SigMatFeatureUnits_noNoise_noControl, MetaDataFeatureNames_noNoise_noControl = ndarrays_DoylePatientsDataset_noNoise_noControl[0]
SigMatHighDataArray_noNoise_noControl, MetaDataHighDataArray_noNoise_noControl, SigMatFeatureNames_noNoise_noControl, SigMatFeatureUnits_noNoise_noControl, MetaDataFeatureNames_noNoise_noControl = ndarrays_DoylePatientsDataset_noNoise_noControl[1]

SigMatLowHugeDataArray_noNoise, MetaDataLowHugeDataArray_noNoise, SigMatHugeFeatureNames_noNoise, SigMatHugeFeatureUnits_noNoise, MetaDataHugeFeatureNames_noNoise = ndarrays_DoylePatientsHugeDataset_noNoise[0]
SigMatHighHugeDataArray_noNoise, MetaDataHighHugeDataArray_noNoise, SigMatHugeFeatureNames_noNoise, SigMatHugeFeatureUnits_noNoise, MetaDataHugeFeatureNames_noNoise = ndarrays_DoylePatientsHugeDataset_noNoise[1]

SigMatLowHugeDataArray_noNoise_noControl, MetaDataLowHugeDataArray_noNoise_noControl, SigMatHugeFeatureNames_noNoise_noControl, SigMatHugeFeatureUnits_noNoise_noControl, MetaDataHugeFeatureNames_noNoise_noControl = ndarrays_DoylePatientsHugeDataset_noNoise_noController[0]
SigMatHighHugeDataArray_noNoise_noControl, MetaDataHighHugeDataArray_noNoise_noControl, SigMatHugeFeatureNames_noNoise_noControl, SigMatHugeFeatureUnits_noNoise_noControl, MetaDataHugeFeatureNames_noNoise_noControl = ndarrays_DoylePatientsHugeDataset_noNoise_noController[1]

fs = 1
slidingWindowSize = int(60*fs)
if np.mod(slidingWindowSize, 2) == 0:
    slidingWindowSize = slidingWindowSize + 1
autoCorrMaxLag = int(30*fs)
slidingWindowsWingap = 1  #int(slidingWindowSize/2)

dirName = 'DoyleSimAnalysisHuge'

##################3
figuresDirName = dirName + '/lowWorkload_noNoise'
print('starting ' + figuresDirName)
SigMat = SigMatLowHugeDataArray_noNoise
SigMatFeatureNames = SigMatHugeFeatureNames_noNoise
SigMatFeatureUnits = SigMatHugeFeatureUnits_noNoise
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low']*P
MetaData = MetaDataLowHugeDataArray_noNoise
MetaDataFeatureNames = MetaDataHugeFeatureNames_noNoise
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/lowWorkload_noNoise_noControl'
print('starting ' + figuresDirName)
SigMat = SigMatLowHugeDataArray_noNoise_noControl
MetaData = MetaDataLowHugeDataArray_noNoise_noControl
SigMatFeatureNames = SigMatHugeFeatureNames_noNoise_noControl
SigMatFeatureUnits = SigMatHugeFeatureUnits_noNoise_noControl
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low_noControl']*P
MetaDataFeatureNames = MetaDataHugeFeatureNames_noNoise_noControl
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/highWorkload_noNoise'
print('starting ' + figuresDirName)
SigMat = SigMatHighHugeDataArray_noNoise
MetaData = MetaDataHighHugeDataArray_noNoise
SigMatFeatureNames = SigMatHugeFeatureNames_noNoise
SigMatFeatureUnits = SigMatHugeFeatureUnits_noNoise
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['High']*P
MetaDataFeatureNames = MetaDataHugeFeatureNames_noNoise
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/highWorkload_noNoise_noControl'
print('starting ' + figuresDirName)
SigMat = SigMatHighHugeDataArray_noNoise_noControl
MetaData = MetaDataHighHugeDataArray_noNoise_noControl
SigMatFeatureNames = SigMatHugeFeatureNames_noNoise_noControl
SigMatFeatureUnits = SigMatHugeFeatureUnits_noNoise_noControl
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['High_noControl']*P
MetaDataFeatureNames = MetaDataHugeFeatureNames_noNoise_noControl
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

#############################################3

'''
dirName = 'DoyleSimAnalysis_SW1'

##################3
figuresDirName = dirName + '/lowWorkload'
print('starting ' + figuresDirName)
SigMat = SigMatLowDataArray
SigMatFeatureNames = SigMatFeatureNames
SigMatFeatureUnits = SigMatFeatureUnits
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low']*P
MetaData = MetaDataLowDataArray
MetaDataFeatureNames = MetaDataFeatureNames
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)

##################3
figuresDirName = dirName + '/highWorkload'
print('starting ' + figuresDirName)
SigMat = SigMatHighDataArray
SigMatFeatureNames = SigMatFeatureNames
SigMatFeatureUnits = SigMatFeatureUnits
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low']*P
MetaData = MetaDataHighDataArray
MetaDataFeatureNames = MetaDataFeatureNames
patientMetaDataTextBox = ''
dataAnalysis(slidingWindowSize, slidingWindowsWingap, autoCorrMaxLag, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName)
'''
'''
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
figuresDirName = dirName + '/lowWorkload_noNoise_noControl'
print('starting ' + figuresDirName)
SigMat = SigMatLowDataArray_noNoise_noControl
MetaData = MetaDataLowDataArray_noNoise_noControl
SigMatFeatureNames = SigMatFeatureNames_noNoise_noControl
SigMatFeatureUnits = SigMatFeatureUnits_noNoise_noControl
P = SigMat.shape[1]
PatientIds = np.arange(1,1+P).tolist()
PatientClassification = ['Low_noControl']*P
MetaDataFeatureNames = MetaDataFeatureNames_noNoise_noControl
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
figuresDirName = dirName + '/highWorkload_noNoise_noControl'
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


'''