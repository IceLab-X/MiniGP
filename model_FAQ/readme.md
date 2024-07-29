# Model_FAQ
This folder contains the answer for frequently asked questions about Gaussian Process models--non positive definite kernel, NAN problem.
The first thing you should do when you encounter such problem is to set data type to double. If the problem still persists, you can refer to the following files:
- why_non_positive_definite: This file explains why the kernel become non-positive definite during the training. 
- Eigendeomposition: This file explains how to fix the non-positive definite kernel using Eigendecomposition.
- Parameter_reset: This file explains how to implement parameter reset to avoid NAN problem caused by gradient explosion.
- why_non_positive_definite (todo): This file explains why the kernel become non-positive definite during the training.
- remove_similar_data: This file demonstrate how to remove similar data and why this help to avoid non-positive definite problem.