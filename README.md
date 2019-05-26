#MATLAB code for Binary Particle Swarm Optimization.

##Data Preparation
Four mat files need to be created:  
* inputs.mat - contains the data to be used for training
* inputs_target.mat - contains one hot encoded class labels for training data
* test.mat - contains the data to be used for validation
* test_target.mat - contains one hot encoded class labels for validation data

##Parameters

The following 3 classifiers can be chosen using variable _ch_ in _classify.m_
* SVM
* MLP
* KNN

##Running the code
* Put all the data files in a folder name _Data_
* run file _pso.m_

Link for algorithm details: https://ieeexplore.ieee.org/abstract/document/637339/ - automatic!