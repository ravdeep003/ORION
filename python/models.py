from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from utils import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

def trainModelSVM(trainX, trainY, kernel, decisionFunction, randomState=None):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 10]
    degrees = [2, 3, 4, 5]
    folds = 5
    # For testing
    # Cs = [1, 10]
    # gammas  = [0.1, 1]
    # degrees = [2, 3]
    # set njobs for running hyperparameter tuining in parallel. Depends on number of available cpus on the system
    njobs = 4
    scoring = 'f1_weighted'
    verbose = 3
    # scoring = 'accuracy'
    # scoring = 'balanced_accuracy'
    if kernel == 'linear':
        param_grid = {'C': Cs}
    elif kernel == 'poly':
        param_grid = {'C': Cs, 'degree' : degrees}
    elif kernel == 'rbf':
        param_grid = {'C': Cs, 'gamma' : gammas}
    else:
        print('Someother kernel function and hyperparameters not tuned')
    model = GridSearchCV(SVC(kernel=kernel, decision_function_shape=decisionFunction, random_state=randomState), param_grid, scoring=scoring, n_jobs=njobs, cv=folds, verbose=verbose)
    #model = SVC(gamma='scale', kernel=kernel, decision_function_shape=decisionFunction, random_state=randomState)
    model.fit(trainX, trainY)
    #print(model.cv_results_)
    #print(model.best_estimator_)
    print('Best Params:', model.best_params_)
    return model

def trainNN(trainX, trainY):
    folds = 5
    # set njobs for running hyperparameter tuining in parallel. Depends on number of available cpus on the system
    njobs = 15
    scoring = 'f1_weighted'
    # scoring = 'accuracy'

    param_grid = {
        'hidden_layer_sizes': [(50,100,50), (100, 100, 100), (150, 100, 150)],
        #'hidden_layer_sizes': [(50,100,50)],
        #'activation': ['tanh', 'relu'],
        # 'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant','adaptive'],
        'learning_rate_init': [0.0001, 0.001, 0.01]}

    # For testing
    # param_grid = {
    #     'hidden_layer_sizes': [(50,100,50)],
    #     #'activation': ['tanh', 'relu'],
    #     # 'solver': ['sgd', 'adam'],
    #     'alpha': [0.01],
    #     'learning_rate': ['constant'],
    #     'learning_rate_init': [0.01]}

    mpl = MLPClassifier(max_iter=500, activation='relu', solver='adam', verbose=True)
    model = GridSearchCV(mpl, param_grid, scoring=scoring, n_jobs=njobs, cv=folds, verbose=3)
    model.fit(trainX, trainY)
    print(model.best_params_)

    #print(model)
    return model

def processModel(model, testX, testY, classes, title):
    predY = model.predict(testX)
    accuracyScore = metrics.accuracy_score(testY, predY)
    ax , classWiseAcc = plot_confusion_matrix(testY, predY, classes, normalize=True, title=title)
    # prf = precision_recall_fscore_support(testY, predY, average='weighted')
    precision = metrics.precision_score(testY, predY, average='weighted')
    recall = metrics.recall_score(testY, predY, average='weighted')
    f1Score = metrics.f1_score(testY,predY, average='weighted')
    prf = [precision, recall, f1Score]
    print('PRF:', prf)
    return accuracyScore, ax, classWiseAcc, prf
    #return accuracyScore, ax, classWiseAcc