# ORION

## Pre-requisites
### Matlab
* [Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html)
* [Poblano Toolbox](https://sandialabs.github.io/poblano_toolbox/)

Download Tensor and Poblano Toolbox and add it to Matlab Path

### Python
* Scipy
* Sklearn
* Numpy
* Matplotlib
* [Tensorly](http://tensorly.org/stable/index.html)

If you are using Anaconda for Python Packages, you can use the following commands:
```
# Create a new environment, so that you don't mess up your exisiting environment
conda create --name <your new env name>
conda activate <your new env name>
conda install scikit-learn
conda install -c tensorly tensorly
conda install -c conda-forge matplotlib
```

## Datasets
* [Indian Pines](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)
* [Pavia University](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene)
* [Salinas Full](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene)
* [Salinas-A](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene)

## How to Run
1. Git clone or download this code 
2. Run `init.m` in matlab to create required folders like `dataset`, `results`, `tensorDataset`.
3. Download dataset(s) from the above provided links into the `dataset` folder.
4. In `runDataset.m` set below variables according to the dataset you are using. For example if you are running it for **IndianPines** dataset:
   ```
      datasetFname = 'dataset/Indian_pines_corrected.mat'
      datasetGt = 'dataset/Indian_pines_gt.mat'
      outFile= 'IndianPines'
      % IMPORTANT: Change X and Y according to variable stored in the .mat(dataset) file
      X = data.indian_pines_corrected;
      Y = gt. indian_pines_gt;
      
      testSize = 0.2
      % Number of datasets to be created
      numData = 10
      % Tensor decompostion rank
      ranks = [1000, 2000]
   ```
   After running `runDataset.m` using Matlab, it will create .mat files in `tensorDataset/IndianPines` folder if `outFile` variable was set accordingly.  
5. Now to run `ORION` method, navigate to python folder and set following variables in `orion.py` file.
```
    dataPath = '../tensorDataset/IndianPines/'
```
After running the `orion.py` file, it will generate results(figures and .mat files) in `results/orion/8020/IndianPines/`. Path of the result depends on the dataset being used and train-test split(in above example testSize was `0.2`, so 80-20 split).

6. We have also provided the code for baselines used in our paper, to run Linear, Polynomial and RBF SVM set follwing variables in `baselines.py` and to run Multi Layer perceptron set the same following variables in `mlpBaseline.py` file:
```
   dataX = loadmat('../dataset/Indian_pines_corrected.mat')
   dataY = loadmat('../dataset/Indian_pines_gt.mat')
   Xog = dataX['indian_pines_corrected'] # 3D object
   Y2d = dataY['indian_pines_gt']        # 2D object
   folderName = 'IndianPines' # Make sure this is correct. Result folders will be created based on this.
   
   # number of runs
   runs = 10
   
   testSize = 0.2   
```
7.(Optional) Our code uses `GridSearchCV` from `sklearn` for hyperparameter tuning, to make it run faster(in parallel) you can set `njobs` variable in trainModelSVM and trainNN in `models.py` according to your system configuration. For details refer to this [link](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

Above instructions to set variables is for IndianPines dataset, to use any other dataset follow instructions 3-6 and set the variables accordingly.
