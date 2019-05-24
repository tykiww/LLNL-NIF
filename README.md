# Plasma Reconstruction Summary

Code used for predicting continuous high-density laser ray-tracing results using deep learning techniques. To be used in VR modules for Clean energy research and STEM outreach. 

![](https://raw.githubusercontent.com/tykiww/NIF-VR/master/images/poster-image.png)

## Collaborators: 

Isaac Whittaker, Jacob Braswell (@jocobtt), Tyki Wada (@tykiww)

## Data Preparation

- Full Factorial Design including noise (Cartesian Product with Gaussian blur in SAS/python)
- Latin Hypercube Design (pyDOE)
- Image Storage in hdf5 as numpy arrays (24000 images, 448MB)
- Labels as X input in CSV

## Computation

- BYU Supercomputer access
   - Pyspark executed on Slurm batches
  
- Google Colab
   - Python3 on tensorflow GPU (TPU)

## Models

- Deep Convolutional Generator (DCG)
- Regular Keras sequential model (forward fully-connected model)


(Proprietary information (ie. packages, models) excluded)
