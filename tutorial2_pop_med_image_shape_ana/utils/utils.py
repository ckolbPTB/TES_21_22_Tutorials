import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit

def trainAndPredictSVM(train_index, test_index, features, labels):

    lsvm = svm.LinearSVC()
    lsvm.fit(features[:, train_index].transpose(), labels[train_index])
    predictedTLabels = lsvm.predict(features[:, test_index].transpose())

    return predictedTLabels, labels[test_index]

def classifyMultipleTimes(part, nRandomSampling, features, labels):
    accuracyPerSampling = np.empty([nRandomSampling])
    sss = StratifiedShuffleSplit(n_splits=nRandomSampling, train_size=part)
    for j, [train_index, test_index] in enumerate(sss.split(features.transpose(), labels)):
        predictedTLabels, testLabels = trainAndPredictSVM(train_index, test_index, features, labels)
        accuracyPerSampling[j] = 1.0 - np.count_nonzero(testLabels - predictedTLabels)/predictedTLabels.shape[0]
        #if j % 50 == 0:
        #    print(f'Random sampling  {j}  ({nRandomSampling})')

    return accuracyPerSampling

def runClassification(nPartition, nRandomSamplings, features, labels):
    avgAccuracyPerPartition = np.empty([nPartition])
    stdDevPerPartition = np.empty([nPartition])
    for lp in range(1, nPartition + 1):
        accuracyPerSampling = classifyMultipleTimes(lp / (nPartition + 1), nRandomSamplings, features, labels)
        avgAccuracyPerPartition[lp - 1] = np.mean(accuracyPerSampling)
        stdDevPerPartition[lp - 1] = np.std(accuracyPerSampling)
        print(f'Part  {lp}  ({nPartition})')

    return avgAccuracyPerPartition, stdDevPerPartition