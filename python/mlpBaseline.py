from scipy.io import loadmat, savemat
import numpy as np
import tensorly as tl
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from models import trainModelSVM, processModel, trainNN
import matplotlib.pyplot as plt
import os


# Load data and labels - make sure data path is correct and change Xog and Y2d according to variable stored in the .mat(dataset) file
dataX = loadmat('../dataset/Indian_pines_corrected.mat')
dataY = loadmat('../dataset/Indian_pines_gt.mat')
Xog = dataX['indian_pines_corrected'] # 3D object
Y2d = dataY['indian_pines_gt']        # 2D object
folderName = 'IndianPines' # Make sure this is correct. Result folders will be created based on this.

runs = 10
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
##############################################################################
rpath = '../results/baselines/' + splitName + '/' + folderName + '/' # path to store the results
fig = rpath + 'fig/'  # path to store the figures

if os.path.isdir(rpath) == False:
    os.makedirs(rpath)
if os.path.isdir(fig) == False:
    os.mkdir(fig)

rpath = rpath + 'MLP' + folderName + '.mat'


randomState = None
count = 0
decisionFunc = 'ovo'
numClasses = len(np.unique(Y))
ssSplit = StratifiedShuffleSplit(n_splits=runs, test_size=testSize, random_state=randomState)


mlpAcc = []
mlpBestParams = []
classMLP = np.zeros((runs, numClasses))
prfMLP = np.zeros((runs, 3))

###############################################################################

for trainIdx, testIdx in ssSplit.split(X,Y):
    mlpTitle = 'Confusion Matrix for MLP - ' + str(count + 1)
    mlpFig = fig + 'MLPCM_' + str(count + 1)

    # Training and testing split
    trainX, testX = X[trainIdx], X[testIdx]
    trainY, testY = Y[trainIdx], Y[testIdx]
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    # print(trainX.shape, testX.shape)
    classes = np.unique(testY)

    mlp = trainNN(trainX, trainY)
    accuracyX, ax, classWiseAccX, prfR = processModel(mlp, testX, testY, classes, title=mlpTitle)
    print('MLP:', accuracyX)
    mlpAcc.append(accuracyX)
    classMLP[count, :] = classWiseAccX
    prfMLP[count, :] = prfR
    mlpBestParams.append(mlp.best_params_)
    plt.savefig(mlpFig)
    plt.clf()
    plt.close()

    count += 1
###############################################################################

savemat(rpath, {'mlpAcc': mlpAcc, 'classMLP': classMLP, 'prfMLP': prfMLP, 'mlpBestParams':mlpBestParams})