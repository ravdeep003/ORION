from scipy.io import loadmat, savemat
import numpy as np
import tensorly as tl
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from models import trainModelSVM, processModel
import matplotlib.pyplot as plt
import os


# Load data and labels - make sure data path is correct and change Xog and Y2d according to variable stored in the .mat(dataset) file
dataX = loadmat('../dataset/Indian_pines_corrected.mat')
dataY = loadmat('../dataset/Indian_pines_gt.mat')
Xog = dataX['indian_pines_corrected'] # 3D object
Y2d = dataY['indian_pines_gt']        # 2D object
folderName = 'IndianPines' # Make sure this is correct. Result folders will be created based on this.

# number of runs
runs = 10
# train and test split - if you want to use 80-20 split(80% training and 20% testing),
# assign testSize=0.2, if you want 30-70 split(30% training and 70% tesing) assign testSize=0.7
testSize = 0.2

# For Creating specific split folders, so that different splits don't overwrite the results.
splitName = int(testSize * 100)
splitName = str(100-splitName) + str(splitName)



# tensorly odject
X3d = tl.tensor(Xog, dtype='float64')
# Unfolidng 3D tensor in the 3rd dimension, so the spectral channels are features
X2d = tl.unfold(X3d, mode=2)
X2d = X2d.T
# print(type(X2d))
print('Unfolded Tensor: %s' %(X2d.shape,))
# Corresponding labels
Ygt = Y2d.flatten()
print('1-D labels: %s' %(Ygt.shape))

# Removing entries with class 0
idxs = np.nonzero(Ygt)
#print(idxs[0].shape)
X = X2d[idxs[0],:]
Y = Ygt[idxs[0]]
print(X.shape)
print(Y.shape)
###############################################################################
rpath = '../results/baselines/' + splitName + '/' + folderName + '/' # path to store the results
fig = rpath + 'fig/'  # path to store the figures

if os.path.isdir(rpath) == False:
    os.makedirs(rpath)
if os.path.isdir(fig) == False:
    os.mkdir(fig)

rpath =  rpath + folderName + '.mat' # result file



# Seed for random number generator - Set to number only for testing.
# Should be set to None
randomState = None
count = 0
# decision function one vs one
decisionFunc = 'ovo'
numClasses = len(np.unique(Y))
ssSplit = StratifiedShuffleSplit(n_splits=runs, test_size=testSize, random_state=randomState)

linearAcc = []
polyAcc = []
rbfAcc = []

classLinear = np.zeros((runs, numClasses))
classPoly = np.zeros((runs, numClasses))
classRBF = np.zeros((runs, numClasses))

prfLinear = np.zeros((runs, 3))
prfPoly = np.zeros((runs, 3))
prfRBF = np.zeros((runs, 3))

linearBestParams = []
polyBestParams = []
rbfBestParams = []

###############################################################################


for trainIdx, testIdx in ssSplit.split(X,Y):
    linearTitle = 'Confusion Matrix for Linear SVM - ' + str(count + 1)
    polyTitle = 'Confusion Matrix for Polynomial SVM - ' + str(count + 1)
    rbfTitle = 'Confusion Matrix for RBF SVM - ' + str(count + 1)

    linearFig = fig + 'linearCM_' + str(count + 1)
    polyFig = fig + 'polyCM_' + str(count + 1)
    rbfFig = fig + 'rbfCM_' + str(count + 1)

    # Training and testing split
    trainX, testX = X[trainIdx], X[testIdx]
    trainY, testY = Y[trainIdx], Y[testIdx]
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    # print(trainX.shape, testX.shape)
    classes = np.unique(testY)

    # Linear SVM
    linearSVM = trainModelSVM(trainX, trainY, kernel='linear', decisionFunction=decisionFunc, randomState=randomState)
    accuracyX, ax, classWiseAccX, prfL = processModel(linearSVM, testX, testY, classes, title=linearTitle)
    print('Linear Accuracy:', accuracyX)
    linearAcc.append(accuracyX)
    classLinear[count, :] = classWiseAccX
    prfLinear[count, :] = prfL
    linearBestParams.append(linearSVM.best_params_)
    # print(prf)
    plt.savefig(linearFig)
    plt.clf()
    plt.close()

    # Poly SVM
    polySVM = trainModelSVM(trainX, trainY, kernel='poly', decisionFunction=decisionFunc, randomState=randomState)
    accuracyX, ax, classWiseAccX, prfP = processModel(polySVM, testX, testY, classes, title=polyTitle)
    print('Polynomial Accuracy:', accuracyX)
    polyAcc.append(accuracyX)
    classPoly[count, :] = classWiseAccX
    prfPoly[count, :] = prfP
    polyBestParams.append(polySVM.best_params_)
    plt.savefig(polyFig)
    plt.clf()
    plt.close()


    # RBF SVM
    rbfSVM = trainModelSVM(trainX, trainY, kernel='rbf', decisionFunction=decisionFunc, randomState=randomState)
    accuracyX, ax, classWiseAccX, prfR = processModel(rbfSVM, testX, testY, classes, title=rbfTitle)
    print('RBF Accuracy:', accuracyX)
    rbfAcc.append(accuracyX)
    classRBF[count, :] = classWiseAccX
    prfRBF[count, :] = prfR
    rbfBestParams.append(rbfSVM.best_params_)
    plt.savefig(rbfFig)
    plt.clf()
    plt.close()

    # should be at the end of the loop
    count += 1
###########################################################################
savemat(rpath, {'linearAcc': linearAcc, 'polyAcc': polyAcc, 'rbfAcc': rbfAcc, 'classLinear': classLinear, 'classPoly': classPoly, 'classRBF': classRBF, 'prfLinear': prfLinear, 'prfPoly': prfPoly, 'prfRBF': prfRBF, 'linearBestParams': linearBestParams, 'polyBestParams': polyBestParams, 'rbfBestParams': rbfBestParams})