import tensorly as tl
import numpy as np
from sklearn.preprocessing import StandardScaler
#from scipy import sparse
from statistics import mean
import matplotlib.pyplot as plt
# from scipy.io import loadmat, savemat
import scipy.io as sio
import matplotlib as mpl
import os
from models import *

# Change the datapath here
dataPath = '../tensorDataset/IndianPines/'

folderName = dataPath.split('/')[-2]
if folderName == '' or folderName == 'tensorDataset':
    print("ERROR: dataPath not set correctly")
    exit()




# resultPath = '../results/SalinasA/result/'
# fig = '../results/SalinasA/result/fig/'
# For testing assign specific number otherwise use None
randomState = None
# one versus one
decFunc = 'ovo'

matFiles = [f for f in os.listdir(dataPath) if f.endswith('.mat')]
#resFiles = [f for f in os.listdir(resultPath) if f.endswith('.mat')]
for file in matFiles:
    # if file in resFiles:
    #     print(file)
    #     continue

    fpath = dataPath + file

    #print(file)
    # print(rpath)
    # Loading data
    data = sio.loadmat(fpath)
    testSize = data['testSize']
    splitName = int(testSize * 100)
    splitName = str(100-splitName) + str(splitName)
    resultPath = '../results/orion/' + splitName + '/' + folderName + '/'
    fig = resultPath + 'fig/'
    if os.path.isdir(resultPath) == False:
        os.makedirs(resultPath)
    if os.path.isdir(fig) == False:
        os.mkdir(fig)

    rpath = resultPath + file
    figPath = fig + file.split('.')[0]


    rank = data['rank']
    err = data['err']
    rmse = data['rmse']
    A = data['A']
    B = data['B']
    C = data['C']
    lam = data['lambda']
    #kr = data['data']
    Y = data['Y']
    testI = data['testI']
    testJ = data['testJ']
    trainI = data['trainI']
    trainJ = data['trainJ']
    #print(lam)
    # Khatri Rao Product
    krp = tl.tenalg.khatri_rao([A,B]) @ np.diag(lam.T[0])
    #print(np.diag(lam.T[0]))
    # krp = tl.tenalg.khatri_rao([A,B])
    # print(krp.shape)
    # diff = krp-kr
    # print(np.linalg.norm(diff))

    # Changing index from Matlab 1 based indexing to python 0 based indexing
    testIs = testI - 1
    testJs = testJ - 1
    trainIs = trainI - 1
    trainJs = trainJ - 1

    # Test data
    labels = Y.flatten()
    arr = np.array([testIs, testJs])
    idx = np.ravel_multi_index(arr, Y.shape)
    # print(idx.shape)
    testX = krp[idx.flatten()]
    testY = labels[idx.flatten()]

    # Training data
    tArr = np.array([trainIs, trainJs])
    tidx = np.ravel_multi_index(tArr, Y.shape)
    # print(tidx.shape)
    trainX = krp[tidx.flatten()]
    trainY = labels[tidx.flatten()]

    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)

    # SVM model with linear kernel
    classes = np.unique(testY)

    modelX = trainModelSVM(trainX, trainY, kernel='linear', decisionFunction=decFunc, randomState=randomState)
    # accuracyX, confusionMX, classWiseAccX = processModel(modelX, testX, testY)
    bestParams = modelX.best_params_
    accuracyX, ax, classWiseAccX, prf = processModel(modelX, testX, testY, classes, title=file)

    # Save figure
    plt.savefig(figPath)
    plt.clf()
    plt.close()
    # Saving data
    sio.savemat(rpath, {'err': err, 'rmse': rmse, 'rank': rank, 'accuracyX': accuracyX, 'classWiseAccX': classWiseAccX, 'prf': prf, 'bestParams': bestParams})