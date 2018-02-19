# Exploring Floating-Point Trade-Offs in Machine Learning

The goal of the analysis is to study what is the impact of floating point precision on the accuracy of well known machine learning predictors: Perceptron (and Average) and SVM. 
The analysis is built upon numerical libraries: [MPFR](http://www.mpfr.org/mpfr-current/mpfr.html) and [SoftFloat](https://iis-git.ee.ethz.ch/pulp-sw/softFloat).

## Overview of the repository
### graphs
It contains the graphical representation of the results of the analysis. Each folder reports a different dataset: diabet,fourclass,heart,inosphere,and splice. In each folder you find the execution of Perceptron(P), Average Perceptron(AP), and SVM for the corresponding dataset. Moreover the folder ``` 1.Selected-Graphs``` contains most interesting graphs selected from the all datasets.
### realtime graphs
(Beta) In realtime graphs we rely on the web service ``` https://mybinder.org``` to reproduce graphs in realtime allowing the user to interact with them. Try it!
### src
the source code of the analyzer and useful tools.
### wax2018-ML
the work submitted to WAX 2018.

## Getting Started

Please follow this instruction to get a copy of the project on your local machine.

### Prerequisites
* Python: tested with Python 2.7.12;
* [gcc Compiler]() Tested with gcc (Ubuntu 5.4.0-6ubuntu1~16.04.5);
* [MPFR](http://www.mpfr.org/mpfr-current/mpfr.html#Installing-MPFR) - MPFR library;
* [SoftFloat](https://iis-git.ee.ethz.ch/pulp-sw/softFloat) - SoftFloat library (In particular FlexFloat);


### Installing
After all libraries have been installed in the machine, clone this repository.
To run the analysis, you need the following:
* ``` /src/FlexFloat-ML/* ``` Implementation of the analysis with FlexFloat.
* ``` /src/MPFR-ML/src/* ``` Implementation of the analysis with MPFR.
* ``` /src/execute.py ``` Run the analysis.

the only file you need to modify is ```src/execute.py```.
In particular:
* ```pathDatasets="(insert datasets paths)"``` ex: pathDatasets="/home/user/project/datasets/";
* ```mpfrcpp="(path to the project file /src/MPFR-ML/src/mpfrcpp.cpp")``` ex. mpfrcpp="../src/mpfr.cpp"
* ```softfloat="(path to SoftFloat library)"``` ex. softfloat="/home/user/softFloat/"
* ```flexfloatcpp=(path to the project file /src/FlexFloat-ML/flexfloat.cpp")``` ex.flexfloatcpp="../src/FlexFloat-ML/flexfloat.cpp"

The directory ``` src/tools/ ``` contains:

* ``` preprocessor.py ``` it converts the dataset from LIBSVM format to our standard.
* ``` surfacePlot.py ``` given in input the results of the analysis it outputs the graphs.

## Running the tests

### Datasets
The path ```pathDatasets``` contains the folder of each dataset we want to analyse.
In particular the dataset has to be splitted in four parts with names: ```part1.txt; part2.txt; part3.txt; part4.txt```

The directory tree has to look like the following:
```
pathDatasets/myDataset/part1.txt
pathDatasets/myDataset/part2.txt
pathDatasets/myDataset/part3.txt
pathDatasets/myDataset/part4.txt
```
The dataset has to be in the following format:

```value value value 1 label```

Inside the folder tools exists a file ```preprocessor.py``` that performs the following format conversion.

From the LIBSVM format (it can contains missing attributes):
```
label 1:value 2:value 3:value ... n:value
label 1:value 3:value ... n:value
label 1:value 2:value 3:value ... n:value
label 2:value 3:value ... n:value
```

To:

```
value value value ... bias label
value value value ... bias label
value value value ... bias label
value value value ... bias label
```
At the same time the ```preprocessor.py``` splits the dataset in four parts.

## Contributing
## Versioning
## Authors
## Acknowledgments
