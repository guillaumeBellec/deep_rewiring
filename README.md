# Introduction to Deep Rewiring in tensorflow
This repository provides basic implementations of Deep Rewiring in Tensorflow. The scripts are showing how to solve Mnist with networks that are kept very sparse all along the training (less than 2% of the connections are active), more advanced simulations and mathematical analyses are described in our paper. The four scripts are:  

- `script_mnist_deep_rewiring.py`: Basic implementation of DEEP R where the number of non-zero in each individual matrix is contrained
- `script_mnist_deep_rewiring_with_global_constraint.py`: Same with a constraint on the global number of connections 
- `script_mnist_deep_rewiring_with_sparse_matrices.py`: Same as the first script, but using the tensorflow sparse matrices
- `script_mnist_soft_deep_rewiring.py`: An implementation of the soft-DEEP R algorithm used as baseline in our paper



"Deep Rewiring: Training very sparse deep networks"  
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein  
ICLR 2018  
(https://arxiv.org/abs/1711.05136)
