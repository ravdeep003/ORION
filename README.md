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
2. Run `init.m` to create required folders like `dataset`, `results`, `tensorDataset`.
3. Download dataset(s) from the above provided links into the `dataset` folder.
4. In `runDataset.m` set below variables according to the dataset you are using. For example if you are running it for **IndianPines** dataset:
   ```datasetFname = 'dataset/Indian_pines_corrected.mat'
      datasetGt = 'dataset/Indian_pines_gt.mat'
      outFile= 'IndianPines'```

