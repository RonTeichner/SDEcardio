import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataAnalysis(SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName, enableSave=True):
    if not(os.path.isdir("./" + figuresDirName)): os.makedirs("./" + figuresDirName)

    rawDataFiguresDirName = figuresDirName + "/rawData"
    if not (os.path.isdir("./" + rawDataFiguresDirName)): os.makedirs("./" + rawDataFiguresDirName)
    singleMatAnalysis("rawData", SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, rawDataFiguresDirName, featuresShareUnits=False, enableSave=enableSave)


def singleMatAnalysis(matrixName, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):
    N, P, F = SigMat.shape

    # analysis on all patients:
    populationName = 'all'
    allPatientsFigureDirName = figuresDirName + "/allPatientsUnion"
    if not (os.path.isdir("./" + allPatientsFigureDirName)): os.makedirs("./" + allPatientsFigureDirName)
    allPatientsAnalysis(matrixName, populationName, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, allPatientsFigureDirName, featuresShareUnits, enableSave)

    # analysis per population:
    populations = list(set(PatientClassification))
    for population in populations:
        specificPopulationFigureDirName = figuresDirName + "/" + population
        if not (os.path.isdir("./" + specificPopulationFigureDirName)): os.makedirs("./" + specificPopulationFigureDirName)
        populationIndexes = stringListCompare(PatientClassification, population)
        SigMatSignlePopulation = SigMat[:, populationIndexes, :]
        allPatientsAnalysis(matrixName, population, SigMatSignlePopulation, SigMatFeatureNames, SigMatFeatureUnits, PatientIds, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, specificPopulationFigureDirName, featuresShareUnits, enableSave)

    # analysis per patient:
    for population in populations:
        specificPopulationFigureDirName = figuresDirName + "/" + population
        populationIndexes = stringListCompare(PatientClassification, population)
        SigMatSignlePopulation = SigMat[:, populationIndexes, :]
        PatientIdSinglePopulation = np.array(PatientIds)[populationIndexes]
        patients = np.unique(PatientIdSinglePopulation)
        CvVecPopulation = np.zeros((patients.shape[0], F))
        for p, patient in enumerate(patients):
            patientId = np.array2string(patient)
            specificPopulationSpecificPatientFigureDirName = specificPopulationFigureDirName + "/" + "patient_" + patientId
            if not (os.path.isdir("./" + specificPopulationSpecificPatientFigureDirName)): os.makedirs("./" + specificPopulationSpecificPatientFigureDirName)
            patientIndexes = PatientIdSinglePopulation == patient
            SigMatSignlePopulationSinglePatient = SigMatSignlePopulation[:, patientIndexes, :]
            CvVecPopulation[p], _ = singlePatientAnalysis(False, matrixName, population, SigMatSignlePopulationSinglePatient, SigMatFeatureNames, SigMatFeatureUnits, patientId, '', PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, specificPopulationSpecificPatientFigureDirName, featuresShareUnits, enableSave)

        # plot CDF of Cv values for population:
        cdfPlot(CvVecPopulation[None, :, :], featuresShareUnits, matrixName, population, 'CV', '', SigMatFeatureNames, SigMatFeatureUnits, specificPopulationFigureDirName, enableSave)

    # analysis per batch of patient:
    for population in populations:
        specificPopulationFigureDirName = figuresDirName + "/" + population
        populationIndexes = stringListCompare(PatientClassification, population)
        SigMatSignlePopulation = SigMat[:, populationIndexes, :]
        PatientIdSinglePopulation = np.array(PatientIds)[populationIndexes]
        patients = np.unique(PatientIdSinglePopulation)
        AcVecPopulation = np.zeros((SigMatSignlePopulation.shape[1], F))
        AcVecIndex = -1
        for p, patient in enumerate(patients):
            patientId = np.array2string(patient)
            specificPopulationSpecificPatientFigureDirName = specificPopulationFigureDirName + "/" + "patient_" + patientId
            patientIndexes = PatientIdSinglePopulation == patient
            SigMatSignlePopulationSinglePatient = SigMatSignlePopulation[:, patientIndexes, :]
            nBatches = SigMatSignlePopulationSinglePatient.shape[1]
            for b in range(nBatches):
                batchId = str(b)
                specificPopulationSpecificPatientBatchFigureDirName = specificPopulationSpecificPatientFigureDirName + "batch_" + batchId
                if not (os.path.isdir("./" + specificPopulationSpecificPatientBatchFigureDirName)): os.makedirs("./" + specificPopulationSpecificPatientBatchFigureDirName)
                SigMatSignlePopulationSinglePatientSingleBatch = SigMatSignlePopulationSinglePatient[:, b, :][:, None, :]
                AcVecIndex = AcVecIndex + 1
                _, AcVecPopulation[AcVecIndex] = singlePatientAnalysis(True, matrixName, population, SigMatSignlePopulationSinglePatientSingleBatch, SigMatFeatureNames, SigMatFeatureUnits, patientId, batchId, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, specificPopulationSpecificPatientBatchFigureDirName, featuresShareUnits, enableSave)

        # plot CDF of Ac values for population:
        cdfPlot(AcVecPopulation[None, :, :], featuresShareUnits, matrixName, population, 'AC', '', SigMatFeatureNames, SigMatFeatureUnits, specificPopulationFigureDirName, enableSave)


def singlePatientAnalysis(singleBatch, matrixName, populationName, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientId, batchId, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):
    N, P, F = SigMat.shape

    # CDF plot:
    cdfPlot(SigMat, featuresShareUnits, matrixName, populationName, PatientId, batchId, SigMatFeatureNames, SigMatFeatureUnits, figuresDirName, enableSave)

    # Cv:
    CvVec = CoefVar(SigMat.reshape(-1, F)[:, None, :]) # union of all batches from patient
    title = matrixName + "_" + populationName + "_" + PatientId + "_Cv(set-points)"
    myBarPlot(SigMatFeatureNames, CvVec[0], title)
    if enableSave: plt.savefig("./" + figuresDirName + "/" + title + ".png")

    if singleBatch:
        assert P == 1
        AcVec = TotalAutoCorr(SigMat)
    else:
        AcVec = np.nan

    return CvVec, AcVec


def allPatientsAnalysis(matrixName, populationName, SigMat, SigMatFeatureNames, SigMatFeatureUnits, PatientId, PatientClassification, MetaData, MetaDataFeatureNames, fs, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):
    N, P, F = SigMat.shape

    # CDF plot:
    patientId = '' # all patients
    cdfPlot(SigMat, featuresShareUnits, matrixName, populationName, patientId, '', SigMatFeatureNames, SigMatFeatureUnits, figuresDirName, enableSave)

    # normalized variance:
    NvVec = NormalizedVariacne(SigMat)
    plt.figure()
    title = matrixName + "_" + populationName + "_normalizedVar"
    myBarPlot(SigMatFeatureNames, NvVec, title)
    if enableSave: plt.savefig("./" + figuresDirName + "/" + title + ".png")

    # CV(set-points)
    CvOfSetPoints = CoefVarOfSetPoints(SigMat)
    plt.figure()
    title = matrixName + "_" + populationName + "_Cv(set-points)"
    myBarPlot(SigMatFeatureNames, CvOfSetPoints, title)
    if enableSave: plt.savefig("./" + figuresDirName + "/" + title + ".png")

def TotalAutoCorr(SigMat):
    N, P, F = SigMat.shape
    AcVec = np.zeros((P, F))
    for p in range(P):
        for f in range(F):
            summed = 0
            for L in range(-(N-2), N-1):
                AcLagVec = AutoCorrSpecificLag(SigMat[:, p, f][:, None, None], L)
                i0 = np.max((L, 0))
                i1 = N - 1 + np.min((0, L))
                weight = (1 + i1 - i0)/np.power(N, 2)
                summed = summed + weight*np.abs(AcLagVec)
            AcVec[p, f] = summed
    return AcVec

def AutoCorrSpecificLag(SigMat, L):
    N, P, F = SigMat.shape
    AcLagVec = np.zeros((P, F))
    for p in range(P):
        for f in range(F):
            AcLagVec[p, f] = pd.Series(SigMat[:, p, f]).autocorr(lag=L)
            if np.isnan(AcLagVec[p, f]): AcLagVec[p, f] = 0.0
    return AcLagVec

def CoefVar(SigMat):
    N, P, F = SigMat.shape
    MeanVec, VarVec = np.zeros((P, F)), np.zeros((P, F))
    for p in range(P):
        for f in range(F):
            MeanVec[p, f] = pd.Series(SigMat[:, p, f]).mean()
            VarVec[p, f] = pd.Series(SigMat[:, p, f]).var()
    CvVec = np.sqrt(np.divide(VarVec, np.power(MeanVec, 2)))
    return CvVec

def CoefVarOfSetPoints(SigMat):
    N, P, F = SigMat.shape
    MeanVec, VarOfMeansVec, MeansOfMeansVec = np.zeros((P, F)), np.zeros((F)), np.zeros((F))
    for p in range(P):
        for f in range(F):
            MeanVec[p, f] = pd.Series(SigMat[:, p, f]).mean()

    for f in range(F):
        MeansOfMeansVec[f] = pd.Series(MeanVec[:, f]).mean()
        VarOfMeansVec[f] = pd.Series(MeanVec[:, f]).var()

    CvOfSetPoints = np.sqrt(np.divide(VarOfMeansVec, np.power(MeansOfMeansVec, 2)))
    return CvOfSetPoints

def NormalizedVariacne(SigMat):
    N, P, F = SigMat.shape
    MeanVec, VarOfMeansVec, TotalVar = np.zeros((P, F)), np.zeros((F)), np.zeros((F))
    for p in range(P):
        for f in range(F):
            MeanVec[p, f] = pd.Series(SigMat[:, p, f]).mean()

    for f in range(F):
        VarOfMeansVec[f] = pd.Series(MeanVec[:, f]).var()
        TotalVar[f] = pd.Series(SigMat.reshape(-1, F)[:, f]).var()

    NvVec = np.divide(VarOfMeansVec, TotalVar)
    return NvVec


def CalcCDF(SigMat):
    N, P, F = SigMat.shape
    n_bins = 1000
    CdfMat, binsMat = np.ndarray((n_bins, P, F)), np.ndarray((P, F, n_bins))
    for p in range(P):
        for f in range(F):
            singlePatientSingleFeature = SigMat[:, p, f]
            n, bins, _ = plt.hist(singlePatientSingleFeature, n_bins, histtype='step', density=True, cumulative=True, label='hist')
            plt.close()  # eliminates the matplotlib plot
            CdfMat[:, p, f], binsMat[p, f, :] = n, bins[:-1]
    return binsMat, CdfMat

def stringListCompare(listOfStrings, str):
    compareList = list()
    for string in listOfStrings:
        compareList.append((string == str))
    return compareList

def cdfPlot(SigMat, featuresShareUnits, matrixName, populationName, patientId, batchId, SigMatFeatureNames, SigMatFeatureUnits, figuresDirName, enableSave):
    N, P, F = SigMat.shape
    binsMat, CdfMat = CalcCDF(SigMat.reshape(-1, F)[:, None, :])

    if featuresShareUnits:  # plot all cdf curves in the same figure
        plt.figure()
        p = 0  # due to union of all patients
        title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_CDF"
        for f in range(F):
            myPlot(binsMat[p, f, :], CdfMat[:, p, f], label=SigMatFeatureNames[f], title=title)
        if enableSave: plt.savefig("./" + figuresDirName + "/" + title + ".png")

    else:  # plot each cdf curve in a new figure
        p = 0  # due to union of all patients
        for f in range(F):
            plt.figure()
            title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_CDF_" + SigMatFeatureNames[f]
            myPlot(binsMat[p, f, :], CdfMat[:, p, f], label=SigMatFeatureNames[f], title=title,
                   xlabel=SigMatFeatureUnits[f])
            if enableSave: plt.savefig("./" + figuresDirName + "/" + title + ".png")


def myPlot(x, y, label='', title='', xlabel=''):
    plt.plot(x, y, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.title(title)


def myBarPlot(names, values, title):
    plt.bar(names, values)
    plt.title(title)


'''
x = np.zeros((4,3,2), dtype=int)

x[:,0,0] = np.array([1,2,3,4])
x[:,1,0] = 10*np.array([1,2,3,4])
x[:,2,0] = 100*np.array([1,2,3,4])

x[:,0,1] = np.array([5,6,7,8])
x[:,1,1] = 10*np.array([5,6,7,8])
x[:,2,1] = 100*np.array([5,6,7,8])
'''