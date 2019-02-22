# Plasma Reconstruction Summary

Code used for predicting continuous high-density laser ray-tracing results using deep learning techniques. To be used in VR modules for STEM outreach. 

## Collaborators: 

Isaac Whittaker, Jacob Braswell (@jocobtt), Tyki Wada (@tykiww)

## Data Preparation

- Full Factorial Design including noise (Cartesian Product with Gaussian blur in SAS/python)
- Latin Hypercube Design (pyDOE)
- Storage in hdf5 (240000 images)

## Computation

- BYU Supercomputer access
  - Pyspark executed on Slurm batches
 - Google Colab
  - Python3 on tensorflow GPU (TPU)

## Model

- Convolutional Neural Network (Regression output)
- Metrics: Customized loss function.


(Proprietary information (ie. packages) excluded)
