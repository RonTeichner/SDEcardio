import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import interpolate
import matplotlib.pyplot as plt

def dataAnalysis(paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, figuresDirName, enableSave=True):
    enableAcSw = False

    paramsDict["slidingWindowSize"], paramsDict["slidingWindowsWingap"], paramsDict["autoCorrMaxLag"], paramsDict["Arlags"] = int(paramsDict["slidingWindowSize"]), int(paramsDict["slidingWindowsWingap"]), int(paramsDict["autoCorrMaxLag"]), int(paramsDict["Arlags"])
    paramsDict["nFeatures"] = len(SigMatFeatureUnits)
    paramsDict["nMetaDataFeatures"] = len(metaDataDf.columns) - 2

    # insert nans out of analysis times:
    features = patientsDf.columns.to_list()[-paramsDict["nFeatures"]:]
    patientsDf.loc[patientsDf["time"] < paramsDict["analysisStartTime"], features] = np.nan
    patientsDf.loc[patientsDf["time"] > paramsDict["analysisEndTime"], features] = np.nan

    if not(os.path.isdir("./" + figuresDirName)): os.makedirs("./" + figuresDirName)

    print('starting raw data analysis')
    rawDataFiguresDirName = figuresDirName + "/rawData"
    if not (os.path.isdir("./" + rawDataFiguresDirName)): os.makedirs("./" + rawDataFiguresDirName)
    singleMatAnalysis("rawData", paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, rawDataFiguresDirName, featuresShareUnits=False, enableSave=enableSave)

    print('starting Mahalanobis analysis')
    patientsMahDf = MahalanobisDistance(paramsDict, patientsDf)
    mahalanobisFiguresDirName = figuresDirName + "/mahalanobis"
    if not (os.path.isdir("./" + mahalanobisFiguresDirName)): os.makedirs("./" + mahalanobisFiguresDirName)
    MahMatFeatureUnits = ['']*len(SigMatFeatureUnits)
    singleMatAnalysis("Mahalanobis", paramsDict, patientsMahDf, metaDataDf, MahMatFeatureUnits, patientMetaDataTextBox, mahalanobisFiguresDirName, featuresShareUnits=True, enableSave=enableSave)

    if enableAcSw:
        print('starting auto-corr analysis')
        patientsAcSwDf = AutoCorrSw(paramsDict, patientsDf)
        paramsDictAcSw = paramsDict.copy()
        paramsDictAcSw["fs"] = paramsDict["fs"]/paramsDict["slidingWindowsWingap"]
        AcSwMatFiguresDirName = figuresDirName + "/autoCorrSw"
        if not (os.path.isdir("./" + AcSwMatFiguresDirName)): os.makedirs("./" + AcSwMatFiguresDirName)
        AcSwMatFeatureUnits = ['']*len(SigMatFeatureUnits)
        singleMatAnalysis("AutoCorrSw", paramsDictAcSw, patientsAcSwDf, metaDataDf, AcSwMatFeatureUnits, patientMetaDataTextBox, AcSwMatFiguresDirName, featuresShareUnits=True, enableSave=enableSave)

        print('starting Mahalanobis auto-corr analysis')
        patientsMahAcSwDf = MahalanobisDistance(patientsAcSwDf)
        MahAcSwMatFiguresDirName = figuresDirName + "/MahAutoCorrSw"
        if not (os.path.isdir("./" + MahAcSwMatFiguresDirName)): os.makedirs("./" + MahAcSwMatFiguresDirName)
        MahAcSwMatFeatureUnits = ['']*len(SigMatFeatureUnits)
        singleMatAnalysis("MahAutoCorrSw", paramsDictAcSw, patientsMahAcSwDf, metaDataDf, MahAcSwMatFeatureUnits, patientMetaDataTextBox, MahAcSwMatFiguresDirName, featuresShareUnits=True, enableSave=enableSave)

def singleMatAnalysis(matrixName, paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):
    F = paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]
    metaDataFeatures = metaDataDf.columns.to_list()[-paramsDict["nMetaDataFeatures"]:]

    # analysis on all patients:
    allPatientsFigureDirName = figuresDirName + "/allPatientsUnion"
    if not (os.path.isdir("./" + allPatientsFigureDirName)): os.makedirs("./" + allPatientsFigureDirName)
    _, _ = allPatientsAnalysis(matrixName, 'all', paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, allPatientsFigureDirName, featuresShareUnits, enableSave)

    # analysis per population:
    populations = metaDataDf["classification"].unique().tolist()
    NvVecPopulation, CvOfSetPointsPopulation = [dict()]*2
    for population in populations:
        specificPopulationFigureDirName = allPatientsFigureDirName + "/" + population
        if not (os.path.isdir("./" + specificPopulationFigureDirName)): os.makedirs("./" + specificPopulationFigureDirName)
        patients = metaDataDf[metaDataDf["classification"] == population][["Id"]].values[:, 0]
        patientsDfSinglePopulation = patientsDf[patientsDf["Id"].isin(patients)]
        NvVecPopulation[population], CvOfSetPointsPopulation[population] = allPatientsAnalysis(matrixName, population, paramsDict, patientsDfSinglePopulation, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, specificPopulationFigureDirName, featuresShareUnits, enableSave)

    # analysis per patient:
    for population in populations:
        specificPopulationFigureDirName = allPatientsFigureDirName + "/" + population
        metaDataSinglePopulationDf = metaDataDf[metaDataDf["classification"] == population]
        patients = metaDataSinglePopulationDf[["Id"]].values[:, 0]
        patientsDfSinglePopulation = patientsDf[patientsDf["Id"].isin(patients)]
        CvVecPopulationDf = pd.DataFrame(columns=features)  #np.zeros((patients.shape[0], F))
        for p, patient in enumerate(patients):
            patientId = np.array2string(patient)
            specificPopulationSpecificPatientFigureDirName = specificPopulationFigureDirName + "/" + "patient_" + patientId
            if not (os.path.isdir("./" + specificPopulationSpecificPatientFigureDirName)): os.makedirs("./" + specificPopulationSpecificPatientFigureDirName)
            patientsDfSinglePopulationSinglePatient = patientsDfSinglePopulation[patientsDfSinglePopulation["Id"] == patient]
            CvVecPopulation, _, _, _, _, _ = singlePatientAnalysis(False, matrixName, population, paramsDict, patientsDfSinglePopulationSinglePatient, metaDataDf, SigMatFeatureUnits, population, patientMetaDataTextBox, specificPopulationSpecificPatientFigureDirName, featuresShareUnits, enableSave)
            CvVecPopulationDf = CvVecPopulationDf.append(CvVecPopulation)

        # plot CDF of Cv values for population:
        cdfPlot(CvVecPopulationDf, featuresShareUnits, matrixName, population, 'CV', '', ['']*len(SigMatFeatureUnits), specificPopulationFigureDirName, enableSave)

    # analysis per batch of patient:
    for population in populations:
        specificPopulationFigureDirName = allPatientsFigureDirName + "/" + population
        metaDataSinglePopulationDf = metaDataDf[metaDataDf["classification"] == population]
        patients = metaDataSinglePopulationDf[["Id"]].values[:, 0]
        patientsDfSinglePopulation = patientsDf[patientsDf["Id"].isin(patients)]
        AcVecBatchPopulationDf, NcVecBatchPopulationDf, ArVecBatchPopulationDf, MeanVecBatchPopulationDf, VarVecBatchPopulationDf, CvVecBatchPopulationDf = [pd.DataFrame(columns=features)]*6
        MetaDataBatchPopulationDf = pd.DataFrame(columns=metaDataDf.columns)
        #AcVecIndex = -1
        for p, patient in enumerate(patients):
            patientId = np.array2string(patient)
            specificPopulationSpecificPatientFigureDirName = specificPopulationFigureDirName + "/" + "patient_" + patientId
            patientsDfSinglePopulationSinglePatient = patientsDfSinglePopulation[patientsDfSinglePopulation["Id"] == patient]
            metaDataSinglePopulationDfSinglePatient = metaDataSinglePopulationDf[metaDataSinglePopulationDf["Id"] == patient]
            batchesIds = patientsDfSinglePopulationSinglePatient["batch"].unique()
            for batchId in batchesIds:
                specificPopulationSpecificPatientBatchFigureDirName = specificPopulationSpecificPatientFigureDirName + "/" + "batch_" + str(batchId)
                if not (os.path.isdir("./" + specificPopulationSpecificPatientBatchFigureDirName)): os.makedirs("./" + specificPopulationSpecificPatientBatchFigureDirName)
                patientsDfSinglePopulationSinglePatientSingleBatch = patientsDfSinglePopulationSinglePatient[patientsDfSinglePopulationSinglePatient["batch"] == batchId]
                CvVec, MeanVec, VarVec, AcVecBatchPopulation, NcVecBatchPopulation, ArVecBatchPopulation = singlePatientAnalysis(True, matrixName, population, paramsDict, patientsDfSinglePopulationSinglePatientSingleBatch, metaDataDf, SigMatFeatureUnits, population, patientMetaDataTextBox, specificPopulationSpecificPatientBatchFigureDirName, featuresShareUnits, enableSave)
                CvVecBatchPopulationDf, MeanVecBatchPopulationDf, VarVecBatchPopulationDf, AcVecBatchPopulationDf, NcVecBatchPopulationDf, ArVecBatchPopulationDf = CvVecBatchPopulationDf.append(CvVec), MeanVecBatchPopulationDf.append(MeanVec), VarVecBatchPopulationDf.append(VarVec), AcVecBatchPopulationDf.append(AcVecBatchPopulation), NcVecBatchPopulationDf.append(NcVecBatchPopulation), ArVecBatchPopulationDf.append(ArVecBatchPopulation)
                MetaDataBatchPopulationDf = MetaDataBatchPopulationDf.append(metaDataSinglePopulationDfSinglePatient)

        # plot CDF, mean, std  of Ac values for population:
        cdfPlot(AcVecBatchPopulationDf, True, matrixName, population, 'AC', '', ['']*len(SigMatFeatureUnits), specificPopulationFigureDirName, enableSave)

        # plot CDF, mean, std  of Nc values for population:
        cdfPlot(NcVecBatchPopulationDf, True, matrixName, population, 'NC', '', ['']*len(SigMatFeatureUnits), specificPopulationFigureDirName, enableSave)

        # plot CDF, mean, std of Cv values for population:
        cdfPlot(CvVecBatchPopulationDf, True, matrixName, population, 'Cv', '', ['']*len(SigMatFeatureUnits), specificPopulationFigureDirName, enableSave)

        # plot CDF, mean, std of Ar values for population:
        cdfPlot(ArVecBatchPopulationDf, True, matrixName, population, 'Ar', '', ['']*len(SigMatFeatureUnits), specificPopulationFigureDirName, enableSave)

        # 2dplot CvOfSetPoints vs means(Cv):
        title = matrixName + "_" + population + "_" + "_CvOfSetPoints_vs_means(Cv)"
        my2dPlot(CvVecBatchPopulationDf.mean(axis=0), CvOfSetPointsPopulation[population], title=title, enableDiagonal=True)
        if enableSave:
            plt.savefig("./" + specificPopulationFigureDirName + "/" + title + ".png")
            plt.close()

        # 2dplot NormalizedVariance vs means(Cv):
        title = matrixName + "_" + population + "_" + "_Nv_vs_means(Cv)"
        my2dPlot(CvVecBatchPopulationDf.mean(axis=0), NvVecPopulation[population], title=title)
        if enableSave:
            plt.savefig("./" + specificPopulationFigureDirName + "/" + title + ".png")
            plt.close()

        # plot scatter plots per metaDatafeature for population:
        specificPopulationScatterPlotsFigureDirName = specificPopulationFigureDirName + "/" + "ScatterPlots"
        if not (os.path.isdir("./" + specificPopulationScatterPlotsFigureDirName)): os.makedirs("./" + specificPopulationScatterPlotsFigureDirName)
        for m, metaDataFeature in enumerate(metaDataFeatures):
            for f, feature in enumerate(features):
                title = matrixName + "_" + population + "_" + "_scatter_" + "CV_" + feature + "_" + metaDataFeature
                myScatter(MetaDataBatchPopulationDf[metaDataFeature], CvVecBatchPopulationDf[feature], label='', title=title, xlabel=metaDataFeature, ylabel=feature+' '+SigMatFeatureUnits[f])
                if enableSave:
                    plt.savefig("./" + specificPopulationScatterPlotsFigureDirName + "/" + title + ".png")
                    plt.close()

                title = matrixName + "_" + population + "_" + "_scatter_" + "Ac_" + feature + "_" + metaDataFeature
                myScatter(MetaDataBatchPopulationDf[metaDataFeature], AcVecBatchPopulationDf[feature], label='', title=title, xlabel=metaDataFeature, ylabel=feature+' '+SigMatFeatureUnits[f])
                if enableSave:
                    plt.savefig("./" + specificPopulationScatterPlotsFigureDirName + "/" + title + ".png")
                    plt.close()

                title = matrixName + "_" + population + "_" + "_scatter_" + "Mean_" + feature + "_" + metaDataFeature
                myScatter(MetaDataBatchPopulationDf[metaDataFeature], MeanVecBatchPopulationDf[feature], label='', title=title, xlabel=metaDataFeature, ylabel=feature+' '+SigMatFeatureUnits[f])
                if enableSave:
                    plt.savefig("./" + specificPopulationScatterPlotsFigureDirName + "/" + title + ".png")
                    plt.close()

                if not(np.isnan(np.float_(VarVecBatchPopulationDf[feature])).all()):
                    title = matrixName + "_" + population + "_" + "_scatter_" + "Std_" + feature + "_" + metaDataFeature
                    myScatter(MetaDataBatchPopulationDf[metaDataFeature], np.sqrt(VarVecBatchPopulationDf[feature]), label='', title=title, xlabel=metaDataFeature, ylabel=feature+' '+SigMatFeatureUnits[f])
                    if enableSave:
                        plt.savefig("./" + specificPopulationScatterPlotsFigureDirName + "/" + title + ".png")
                        plt.close()

def singlePatientAnalysis(singleBatch, matrixName, populationName, paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, PatientClassification, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):
    assert len(patientsDf["Id"].unique()) == 1
    if singleBatch: assert len(patientsDf["batch"].unique()) == 1
    PatientId = str(patientsDf["Id"].unique()[0])
    F = paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]
    if singleBatch:
        batchId = str(patientsDf["batch"].unique()[0])
    else:
        batchId = ""

    patientsDfBatchUnion = patientsDf.copy()
    patientsDfBatchUnion["batch"] = 0

    # CDF plot: # union of all batches from patient
    cdfPlot(patientsDfBatchUnion, featuresShareUnits, matrixName, populationName, PatientId, batchId, SigMatFeatureUnits, figuresDirName, enableSave)

    # Cv:
    CvVec, MeanVec, VarVec = CoefVar(paramsDict, patientsDfBatchUnion) # union of all batches from patient
    if singleBatch:
        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Cv(set-points)"
    else:
        title = matrixName + "_" + populationName + "_" + PatientId + "_Cv(set-points)"
    myBarPlot(CvVec.squeeze(), title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

    # mean:
    if singleBatch:
        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Mean"
    else:
        title = matrixName + "_" + populationName + "_" + PatientId + "_Mean"
    myBarPlot(MeanVec.squeeze(), title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

        # std:
        if singleBatch:
            title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Std"
        else:
            title = matrixName + "_" + populationName + "_" + PatientId + "_Std"
        myBarPlot(np.sqrt(VarVec.squeeze()), title)
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()

    AcVec, NcVec, ArVec = [pd.DataFrame(columns=features)]*3
    if singleBatch:
        AcVec, NcVec, ArVec = TotalAutoCorr(paramsDict, patientsDf), TotalNormalizedCorr(paramsDict, patientsDf), ArPredictionLevel(paramsDict, patientsDf)
        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Ac"
        myBarPlot(AcVec.squeeze(), title)
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()

        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Nc"
        myBarPlot(NcVec.squeeze(), title)
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()

        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_Ar"
        myBarPlot(ArVec.squeeze(), title)
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()


        # plot trajectories:
        for f, feature in enumerate(features):
            title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_" + feature
            myPlot(patientsDf["time"].values, patientsDf[feature].values, label=feature, title=title, xlabel='sec', ylabel=SigMatFeatureUnits[f])
            if enableSave:
                plt.savefig("./" + figuresDirName + "/" + title + ".png")
                plt.close()

        # plot mean-normalized trajectories:
        title = matrixName + "_" + populationName + "_" + PatientId + "_" + batchId + "_" + "mean_normalized_trajectories"
        for f, feature in enumerate(features):
            values = patientsDf[feature].values / patientsDf[feature].mean()
            myPlot(patientsDf["time"].values, values, label=feature, title=title, xlabel='sec', ylabel=SigMatFeatureUnits[f])
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()

    return CvVec, MeanVec, VarVec, AcVec, NcVec, ArVec


def allPatientsAnalysis(matrixName, populationName, paramsDict, patientsDf, metaDataDf, SigMatFeatureUnits, patientMetaDataTextBox, figuresDirName, featuresShareUnits, enableSave):

    # CDF , mean, std plot:
    patientId = '' # all patients
    cdfPlot(patientsDf, featuresShareUnits, matrixName, populationName, patientId, '', SigMatFeatureUnits, figuresDirName, enableSave)

    # normalized variance:
    NvVec, TotalVar = NormalizedVariacne(paramsDict, patientsDf)
    title = matrixName + "_" + populationName + "_normalizedVar"
    myBarPlot(NvVec, title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

    # CV(set-points)
    CvOfSetPoints, _, MeansOfMeansVec = CoefVarOfSetPoints(paramsDict, patientsDf)
    title = matrixName + "_" + populationName + "_Cv(set-points)"
    myBarPlot(CvOfSetPoints, title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

    return NvVec, CvOfSetPoints


def MahalanobisDistance(paramsDict, patientsDf):
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]

    MeanVec, VarVec = np.zeros((B, F)), np.zeros((B, F))
    patientsMahDf = pd.DataFrame(columns=patientsDf.columns)
    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            MeanVec[b] = singleBatch[features].mean(axis=0).values
            VarVec[b] = singleBatch[features].var(axis=0).values

            numerator = np.power(singleBatch[features] - MeanVec[b], 2)
            denominator = VarVec[b]

            singleBatch[features] = np.sqrt(np.divide(numerator, denominator))
            patientsMahDf = patientsMahDf.append(singleBatch)

    return patientsMahDf


def ArPredictionLevel(paramsDict, patientsDf):
    lags = paramsDict["Arlags"]
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]

    ArVec = np.zeros((B, F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            for f, feature in enumerate(features):
                singleBatchSingleFeature = singleBatch[['time', feature]]
                singleBatchSingleFeatureResampled, fsNew = seriesResample(singleBatchSingleFeature)
                nLags = int(np.floor(lags * fsNew))
                ArVec[b, f] = ArPrediction(singleBatchSingleFeatureResampled[feature].values, nLags)[1]

    ArVec = pd.DataFrame(ArVec, columns=features)

    return ArVec

def seriesResample(patientDf):
    time = patientDf.columns.to_list()[0]
    feature = patientDf.columns.to_list()[-1]
    tVec, data = patientDf[time].values, patientDf[feature].values

    # we choose the new sampling frequency to be the most popular sampling frequency in the data
    # and we choose the sampling phase such that as many original samples are used in the resampled signal

    values, counts = np.unique(np.diff(tVec), return_counts=True)
    mostPopularTs = values[np.argmax(counts)]  # [sec]

    phases = np.mod(tVec, mostPopularTs)
    values, counts = np.unique(phases, return_counts=True)
    mostPopularPhase = values[np.argmax(counts)]

    duration = tVec[-1] - tVec[0]  # [sec]
    nSamplesNew = int(np.floor(duration/mostPopularTs + 1))
    tVecNew = tVec[0] + mostPopularTs * np.arange(0, nSamplesNew)
    tVecNew = tVecNew + (np.mod(tVecNew[0], mostPopularTs) - mostPopularPhase)
    tVecNew = tVecNew[np.logical_and(tVecNew >= tVec.min(), tVecNew <= tVec.max())]
    
    f = interpolate.interp1d(tVec, data)
    dataResampled = f(tVecNew)  # use interpolation function returned by `interp1d`

    patientDfResampled = pd.DataFrame(np.concatenate((tVecNew[:, None], dataResampled[:, None]), axis=1), columns=[time, feature])

    return patientDfResampled, 1/mostPopularTs

def TotalAutoCorr(paramsDict, patientsDf):
    autoCorrMaxLag = paramsDict["autoCorrMaxLag"]
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]

    AcVec = np.zeros((B, F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            for f, feature in enumerate(features):
                singleBatchSingleFeature = singleBatch[['time', feature]]
                singleBatchSingleFeatureResampled, fsNew = seriesResample(singleBatchSingleFeature)
                N = singleBatchSingleFeatureResampled.shape[0]
                nLagSampled = int(np.floor(autoCorrMaxLag*fsNew))
                minLag, maxLag = int(- np.min((N - 2, nLagSampled))), int(1 + np.min((N - 2, nLagSampled)))
                summed = 0
                for L in range(minLag, maxLag):
                    AcLagVec = AutoCorrSpecificLag(singleBatchSingleFeatureResampled[feature].values[:, None, None], L)
                    i0 = np.max((L, 0))
                    i1 = N - 1 + np.min((0, L))
                    weight = (1 + i1 - i0) / np.power(N, 2)
                    summed = summed + np.abs(AcLagVec)
                AcVec[b, f] = summed
    AcVec = pd.DataFrame(AcVec, columns=features)

    return AcVec

def TotalNormalizedCorr(paramsDict, patientsDf):
    normalizedCorrMaxLag = paramsDict["autoCorrMaxLag"]
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]

    NcVec = np.zeros((B, F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            for f, feature in enumerate(features):
                singleBatchSingleFeature = singleBatch[['time', feature]]
                singleBatchSingleFeatureResampled, fsNew = seriesResample(singleBatchSingleFeature)
                N = singleBatchSingleFeatureResampled.shape[0]
                nLagSampled = int(np.floor(normalizedCorrMaxLag * fsNew))
                minLag, maxLag = int(- np.min((N - 2, nLagSampled))), int(1 + np.min((N - 2, nLagSampled)))
                summed = 0
                for L in range(minLag, maxLag):
                    NcLagVec = NormalizedCorrelationSpecificLag(singleBatchSingleFeatureResampled[feature].values[:, None, None], L)
                    i0 = np.max((L, 0))
                    i1 = N - 1 + np.min((0, L))
                    weight = (1 + i1 - i0) / np.power(N, 2)
                    summed = summed + np.abs(NcLagVec)
                NcVec[b, f] = summed
    NcVec = pd.DataFrame(NcVec, columns=features)

    return NcVec

def AutoCorrSpecificLag(SigMat, L):
    N, P, F = SigMat.shape
    AcLagVec = np.zeros((P, F))
    for p in range(P):
        for f in range(F):
            AcLagVec[p, f] = pd.Series(SigMat[:, p, f]).autocorr(lag=L)
            if np.isnan(AcLagVec[p, f]): AcLagVec[p, f] = 0.0
    return AcLagVec

def NormalizedCorrelationSpecificLag(SigMat, L):
    N, P, F = SigMat.shape
    NcLagVec = np.zeros((P, F))

    i0 = np.max((L, 0))
    i1 = N - 1 + np.min((0, L))

    LagMat = SigMat[i0-L:i1+1-L]
    SigMat = SigMat[i0:i1+1]

    for p in range(P):
        for f in range(F):
            s,l = SigMat[:, p, f], LagMat[:, p, f]
            SigMat_ValidIndexes, LagMat_ValidIndexes = np.logical_not(np.isnan(s)), np.logical_not(np.isnan(l))
            validIndexes = np.logical_and(SigMat_ValidIndexes, LagMat_ValidIndexes)

            effectiveNorm_SigMat, effectiveNorm_LagMat = np.sqrt(np.power(s[validIndexes], 2).sum()), np.sqrt(np.power(l[validIndexes], 2).sum())
            dotProduct = np.dot(s[validIndexes], l[validIndexes])

            NcLagVec[p, f] = dotProduct/(effectiveNorm_SigMat*effectiveNorm_LagMat)

    return NcLagVec

def AutoCorrSw(paramsDict, patientsDf):
    raise ValueError('not converter to dataframe')
    assert np.mod(windowSize, 2) == 1
    N, P, F = SigMat.shape
    h = int(0.5*(windowSize-1))
    paddedSigMat = np.concatenate((np.zeros((h, P, F)), SigMat, np.zeros((h, P, F))), axis=0)
    AcSwMat = np.zeros_like(SigMat[::wingap])

    for k in range(AcSwMat.shape[0]):
        tilde_k = k*wingap
        startIndex = tilde_k
        stopIndex = startIndex + 2*h + 1
        AcSwMat[k] = TotalAutoCorr(paddedSigMat[startIndex:stopIndex], autoCorrMaxLag=np.inf)
    return AcSwMat


def CoefVar(paramsDict, patientsDf):
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]

    MeanVec, VarVec = np.zeros((B, F)), np.zeros((B, F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            MeanVec[b] = singleBatch[features].mean(axis=0).values
            VarVec[b] = singleBatch[features].var(axis=0).values

    MeanVec = pd.DataFrame(MeanVec, columns=features)
    VarVec = pd.DataFrame(VarVec, columns=features)

    # assert np.power(MeanVec, 2).min() > 0
    CvVec = np.sqrt(np.divide(VarVec, np.power(MeanVec, 2)))
    return CvVec, MeanVec, VarVec

def CoefVarOfSetPoints(paramsDict, patientsDf):
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]
    MeanVec, VarOfMeansVec, MeansOfMeansVec = np.zeros((B, F)), np.zeros((F)), np.zeros((F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            MeanVec[b] = singleBatch[features].mean(axis=0).values

    MeanVec = pd.DataFrame(MeanVec, columns=features)
    MeansOfMeansVec = MeanVec.mean(axis=0)
    VarOfMeansVec = MeanVec.var(axis=0)

    # assert np.power(MeansOfMeansVec, 2).min() > 0
    CvOfSetPoints = np.sqrt(np.divide(VarOfMeansVec, np.power(MeansOfMeansVec, 2)))
    return CvOfSetPoints, MeanVec, MeansOfMeansVec

def getTotalNumberOfBatches(patientsDf):
    totalNumberOfBatches = 0
    for Id in patientsDf["Id"].unique():
        totalNumberOfBatches = totalNumberOfBatches + len(patientsDf[patientsDf["Id"] == Id]["batch"].unique())
    return totalNumberOfBatches

def NormalizedVariacne(paramsDict, patientsDf):
    B, F = getTotalNumberOfBatches(patientsDf), paramsDict["nFeatures"]
    features = patientsDf.columns.to_list()[-F:]
    MeanVec, VarOfMeansVec, TotalVar = np.zeros((B, F)), np.zeros((F)), np.zeros((F))

    b = -1
    for Id in patientsDf["Id"].unique():
        singlePatientDf = patientsDf[patientsDf["Id"] == Id]
        for batch in singlePatientDf["batch"].unique():
            singleBatch = singlePatientDf[singlePatientDf["batch"] == batch]
            b = b + 1
            MeanVec[b] = singleBatch[features].mean(axis=0).values

    MeanVec = pd.DataFrame(MeanVec, columns=features)
    VarOfMeansVec = MeanVec.var(axis=0)
    TotalVar = patientsDf[features].var(axis=0)

    # assert TotalVar.min() > 0
    NvVec = np.divide(VarOfMeansVec, TotalVar)

    return NvVec, TotalVar


def CalcCDF(SigMat):
    N, P, F = SigMat.shape
    n_bins = 1000
    CdfMat, binsMat = np.ndarray((n_bins, P, F)), np.ndarray((P, F, n_bins))
    for p in range(P):
        for f in range(F):
            singlePatientSingleFeature = SigMat[:, p, f]
            notNanIndexes = np.logical_not(np.isnan(np.float_(singlePatientSingleFeature)))
            n, bins, _ = plt.hist(singlePatientSingleFeature[notNanIndexes], n_bins, histtype='step', density=True, cumulative=True, label='hist')
            plt.close()  # eliminates the matplotlib plot
            CdfMat[:, p, f], binsMat[p, f, :] = n, bins[:-1]
    return binsMat, CdfMat

def stringListCompare(listOfStrings, str):
    compareList = list()
    for string in listOfStrings:
        compareList.append((string == str))
    return compareList

def cdfPlot(patientsDf, featuresShareUnits, matrixName, populationName, patientId, batchId, SigMatFeatureUnits, figuresDirName, enableSave):
    F = len(SigMatFeatureUnits)
    features = patientsDf.columns.to_list()[-F:]

    # plot Mean
    title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_Mean"
    myBarPlot(patientsDf[features].mean(axis=0), title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

    # plot std
    title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_Std"
    myBarPlot(patientsDf[features].std(axis=0), title)
    if enableSave:
        plt.savefig("./" + figuresDirName + "/" + title + ".png")
        plt.close()

    binsMat, CdfMat = CalcCDF(patientsDf[features].values[:, None, :])

    if featuresShareUnits:  # plot all cdf curves in the same figure
        plt.figure()
        p = 0  # due to union of all patients
        title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_CDF"
        for f in range(F):
            myPlot(binsMat[p, f, :], CdfMat[:, p, f], label=features[f], title=title)
        if enableSave:
            plt.savefig("./" + figuresDirName + "/" + title + ".png")
            plt.close()

    else:  # plot each cdf curve in a new figure
        p = 0  # due to union of all patients
        for f in range(F):
            title = matrixName + "_" + populationName + "_" + patientId + "_" + batchId + "_CDF_" + features[f]
            myPlot(binsMat[p, f, :], CdfMat[:, p, f], label=features[f], title=title, xlabel=SigMatFeatureUnits[f])
            if enableSave:
                plt.savefig("./" + figuresDirName + "/" + title + ".png")
                plt.close()

def ArPrediction(x, p):
    X = np.zeros((len(x) - p, p))
    y = x[p:]
    for r in range(p):
        X[:, r] = x[r:r + X.shape[0]]
    # remove nans:
    y_validIndexes = np.logical_not(np.isnan(y))
    X_validRows = np.logical_not(np.isnan(X)).all(axis=1)
    validRows = np.logical_and(y_validIndexes, X_validRows)

    X_noNans, y_noNans = X[validRows], y[validRows]

    if validRows.any():
        reg = LinearRegression().fit(X_noNans, y_noNans)
        predictionLevel = reg.score(X_noNans, y_noNans)
        coefs = reg.coef_[None, :]
        intercept = reg.intercept_

        predictions_noNans = reg.predict(X[X_validRows])
        predictions = np.full(len(y), np.nan)
        predictions[X_validRows] = predictions_noNans
        predictions = np.concatenate((np.full(p, np.nan), predictions))
    else:
        predictions, predictionLevel, intercept, coefs = np.full(len(y), np.nan), 0, np.nan, np.full(p, np.nan)

    return predictions, predictionLevel, intercept, coefs

def myPlot(x, y, label='', title='', xlabel='', ylabel=''):
    plt.plot(x, y, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def myScatter(x, y, label='', title='', xlabel='', ylabel=''):
    pearsonCorr = round(pd.Series(x).corr(pd.Series(y)), 2)
    title = title + ' p.c=' + str(pearsonCorr)
    plt.scatter(x, y, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def myBarPlot(values, title):
    #stringsSum = sumStrings(names)
    #factor = 0.1
    #plt.figure(figsize=(factor * stringsSum * 6.4, 4.8))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    fig.suptitle(title)
    scales = ['linear', 'log']
    for i, scale in enumerate(scales):
        axs[i].bar(values.index, values.values)
        #axs[i].set_title(scale)
        axs[i].grid()
        axs[i].set_yscale(scale)

def my2dPlot(x, y, title, enableDiagonal=False):
    features = x.index.to_list()
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    scales = ['linear', 'log']
    for i, scale in enumerate(scales):
        #plt.subplot(1, len(scales), i+1)
        if enableDiagonal:
            axs[i].plot(np.array([x.min(), x.max()]), np.array([x.min(), x.max()]), '--', label='diagonal')
        for feature in features:
            axs[i].plot(x[feature], y[feature], '+', label=feature)
        if i==0: axs[i].legend()
        #axs[i].set_title(scale)
        axs[i].grid()
        axs[i].set_xscale(scale)
        axs[i].set_yscale(scale)

def sumStrings(names):
    stringSum = 0
    for name in names:
        stringSum = stringSum + len(name)
    return stringSum


def R_L(cj, cm, L):
    N = cj.shape[0]

    i0 = np.max((L, 0))
    i1 = N - 1 + np.min((0, L))

    cj, cm = cj[i0:i1+1], cm[i0-L:i1+1-L]

    cj_ValidIndexes, cm_ValidIndexes = np.logical_not(np.isnan(cj)), np.logical_not(np.isnan(cm))
    validIndexes = np.logical_and(cj_ValidIndexes, cm_ValidIndexes)

    cj_mean, cm_mean = cj[validIndexes].mean(), cm[validIndexes].mean()

    effectiveNorm_cj, effectiveNorm_cm = np.sqrt(np.power(cj[validIndexes] - cj_mean, 2).sum()), np.sqrt(np.power(cm[validIndexes] - cm_mean, 2).sum())
    dotProduct = np.dot(cj[validIndexes] - cj_mean, cm[validIndexes] - cm_mean)

    return dotProduct/(effectiveNorm_cm*effectiveNorm_cj)

def SigMat2Df(SigMat, fs, SigMatFeatureNames, PatientIds, nBatchesPerPatient):
    N, P, F = SigMat.shape
    tVec = np.arange(0, N) / fs
    patientsDf = pd.DataFrame(columns=["time", "Id", "batch"] + SigMatFeatureNames)

    patientsDf["time"] = tVec.tolist() * P
    patientsDf["Id"] = [val for val in PatientIds for _ in range(N)]
    batchList = list()
    for p in range(len(nBatchesPerPatient)):
        for b in range(int(nBatchesPerPatient[p])):
            batchList = batchList + [b]*N

    patientsDf["batch"] = batchList
    for f, feature in enumerate(SigMatFeatureNames):
        patientsDf[feature] = np.transpose(SigMat[:, :, f]).reshape(-1)
    return patientsDf

def MetaData2Df(MetaData, MetaDataFeatureNames, PatientClassification, PatientIds):
    metaDataDf = pd.DataFrame(columns=["Id", "classification"] + MetaDataFeatureNames)
    metaDataDf["Id"] = np.unique(PatientIds)

    patientFirstIndexes = np.zeros(len(metaDataDf["Id"]), dtype=int)
    PatientClassificationList = list()
    for p, Id in enumerate(metaDataDf["Id"]):
        for pii, pi in enumerate(PatientIds):
            if pi == Id:
                patientFirstIndexes[p] = pii
                PatientClassificationList.append(PatientClassification[patientFirstIndexes[p]])
                break

    metaDataDf["classification"] = PatientClassificationList

    for f, feature in enumerate(MetaDataFeatureNames):
        metaDataDf[feature] = MetaData[patientFirstIndexes, f]

    return metaDataDf

'''
x = np.zeros((4,3,2), dtype=int)

x[:,0,0] = np.array([1,2,3,4])
x[:,1,0] = 10*np.array([1,2,3,4])
x[:,2,0] = 100*np.array([1,2,3,4])

x[:,0,1] = np.array([5,6,7,8])
x[:,1,1] = 10*np.array([5,6,7,8])
x[:,2,1] = 100*np.array([5,6,7,8])
'''