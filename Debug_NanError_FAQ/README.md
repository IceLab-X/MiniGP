# Debug NanError FAQ

This folder contains the answers to frequently asked questions about Gaussian Process models, mainly, the NaN error caused by non-positive definite kernels.

The first thing you should do when you encounter such problems is to **set the data type to double**. If the problem still persists, you can refer to the following files for solutions that we have found so far:

- **why_non_positive_definite**: This file explains why the kernel becomes non-positive definite during training.
- **Eigendecomposition**: This file explains how to fix the non-positive definite kernel using Eigendecomposition.
- **Parameter_reset**: This file explains how to implement parameter reset to avoid NaN problems caused by gradient explosion.
- **why_non_positive_definite (todo)**: This file explains why the kernel becomes non-positive definite during training.
- **remove_similar_data**: This file demonstrates how to remove similar data and why this helps to avoid non-positive definite problems.
