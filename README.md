This repository contains the source code for the our paper "A normalizing flow density estimator for out-of-distribution detection". 

MLP-mixer is used as a backbone and a normalizing flow model is attached to the output of the last layer of the backbone. The normalizing flow model is trained to estimate the density of the training data. The model is then used to detect out-of-distribution samples.


The steps to train the models are 
1. Train mlp-mixer, or you can download a pretrained model from here.
1. Fine-tune the model on in-distribution data.
1. Extract in-distribution data features of the penultimate layer of the backbone.
1. Train a normalizing flow model on the extracted features.
2. Evaluate OOD detection performance.