# Core
Welcome to the core folder of the Mini-GP repository. This folder contains the fundamental components to build a GP (such as kernel functions and common computational operations) and GP models as callable class for easy use. We encourage to go to GPmodels_xxx to see more implementation details for your modification need.

This README provides an overview of the folder's structure, key files, and instructions on how to get started.
  - kernel.py: contains all the kernel functions
  - GP_CommonCalculation.py: contains all the common computation in GP, normalization functions and input warping functions. This essential to help you build your own GP model satisfying your needs. 
  - non_positive_definite_fixer.py: a set of tools to use when your kernel is non-positive definite, or you encounter NaN problem.

## Python scripts for the core models

  - autoGP.py: A GP model that automatically standardize the data and choose the kernel for you. It is a simple GP model that is easy to use for general users.
  - cigp_DeepKernel.py: A GP model that uses a deep kernel to model the data. It is a more complex GP model that is suitable for users who want to model complex data. [Original paper](https://arxiv.org/abs/1511.02222)
  - cigp_baseline.py: A basic GP model that is used as a baseline methods, and it is used for compare the performance of the other GP models.
  - hogp.py: A GP model that uses high-order output to model the data. It is suitable for users who want to model data with high-order output. [Original paper](https://proceedings.mlr.press/v89/zhe19a.html)
  - sgpr.py: A sparse GP model that uses variational inference to approximate the posterior distribution. It is suitable for users who want to model data with size between 1k to 10k. [Original paper](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)
  - svgp.py: A sparse GP model that uses stochastic variational inference that allow Mini-Batch gradient descent. It is suitable for users who want to model data with size over 10k. [Original paper](https://arxiv.org/abs/1411.2005)
  - ParametricGP.py: A seemly self-contradict idea, but a highly efficient and accurate GP model for large dataset. It is suitable for users who want to model data with size over 10k. 
  - inputWarpedGP.py: A GP model that uses input warping to model the data. It is suitable for users who want to model non-stationary data. 
  - gplvm.py: A GP model that uses latent variable to model the data. It is suitable for non-linear dimension reduction. [Original paper](https://papers.nips.cc/paper/2009/file/7c4d21b4b9f8f2b8d1e3e7d2e2e5b7f8-Paper.pdf)

### To understand how to use these callable classes, please refer to the test cases provided at the end of each model.py file.
```
