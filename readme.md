# MiniGP
MiniGP is a minimalistic Gaussian Process (GP) library. It is designed to be simple and easy to understand. 

Despite that there are many successful GP libraries, such as GPy, GPflow, and Pyro, they are often too complex for beginners to understand. Things get worse when the user wants to customize the model. MiniGP is designed to be simple and easy to understand. It is a great tool for educational purposes. We also try to make it easy for anyone to use without a lot of background knowledge of GP.

## Installation
We do not have a pip package yet. You can install it by cloning the repository and rune the code for your own purpose. At this stage we think it is better to keep it simple and customizable. It servers as rather demo code than a library, with some useful functions to make computation easier.

## Structure
- Key components (in the root folder)
  - kernel.py: contains all the kernel functions
  - GP_CommonCalculation.py: contains all the common computation in GP
  - gp_transform.py: contains all the transformation functions for GP. NOT IMPLEMENTED YET.

- GPTutorial (folder): our signature tutorial for GP. It is a step-by-step tutorial to build a GP model from scratch. It is a great way to understand the basic of GP. Most of the tutorial is self-contained and can be used as a standalone code.
  - GPTutorial_basicGP: a basic GP model. 
  - GPTutorial_basicGP_chinese: a basic GP model in Chinese.
  - GPTutorial_hogp_chinese: high-oreder GP [(HOGP)](https://proceedings.mlr.press/v89/zhe19a.html) in Chinese.
  - GPTutorial_DynamicModel: GP dynamic model [(GPDM)](https://www.dgp.toronto.edu/~jmwang/gpdm/nips05final.pdf) in Chinese.
  - GPTutorial_sparseGP: sparse GP model [VSGP](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)


<!-- - Self-contained GP models.
  Several GP models that are self-contained and practical to use (we use them in our many of our research projects).  -->
  - GPmodels_Classic (folder): basic GP model and its variation, such as DeepKernel GP, InputWarp GP . It demonstrates how to build a GP model with the GP_CommonCalculation.
    - simpleGP, a basic GP model. It demonstrates how to build a GP model with the GP_CommonCalculation.
    - deepKernelGP, a GP model with deep kernel.[(DKL)](https://arxiv.org/abs/1511.02222) 
    - inputWarpGP, a GP model with input warping.[(IW)](https://arxiv.org/abs/1509.01168)
    - transformGP, a GP model with transformation. Contains all the transformation functions for GP. NOT IMPLEMENTED YET.
  - GPmodels_LatentVariable: provides tools for implementing Gaussian Process models with latent variables.
    - latentVariableGP, a GP model with latent variable. 
  - GPmodels_MultiOutput: provides tools for implementing Gaussian Process models with multiple outputs.
    - ConditionalIndependentGP, 
    - HighOrderGP, a GP model with high-order output.
    - IntrinsicModel,
  - GPmodels_Sparse: provides tools for implementing sparse Gaussian Process models.
    - sparseGP, a variational sparse GP model.
    - bigDataGP, a stochastic variational interference GP model that can allow mini-batch training. [(SVGP)](https://arxiv.org/abs/1411.2005)
    - SGP_parametric,
- Model_FAQ: a FAQ for GP model. It contains some techniques to solve NAN problem in GP model.
  - Eigendecomposition: a demo to show how to replace cholesky decomposition with eigendecomposition .
  - non_positive_definite_fixer: a collection of techniques to solve NAN problem in GP model.
  - non_positive_definite_fixer_implement: a demo to show how to use non_positive_definite_fixer when building up your own GP model.
  - parameter_reset: a demo to show how to reset the parameters when they explode.
  - remove_similar_data: a demo to show removing similar training data fix non positive definite issue in some circumstances.
  - why_non_positive_definite: a tutorial to show why your kernel is non-positive definite.
- TO BE ADDED.
  -
  - 
  - 
  - 

- Legacy: contains all the old GP models. They are deprecated and will be removed in the future.

## Contribution
We welcome any contribution to this project. Please feel free to open an issue or pull request. We are happy to discuss any ideas with you. 


<!-- # Todo list -->
<!-- - [ ] !Test different way of kernel inversion! -->
<!-- - [ ] add a testunit for kernel function -->
<!-- - [ ] add a testunit for GP model -->
<!-- - [ ] add GP_build which take a GP model and warp it with normalizer, optimizer, and loss function definition.
- [ ] add mean function to GP model
- [ ] redo previous GP models using the new architecture
- [ ] add GPU enabled GP
- [ ] Need HOGP -->






# Log

## 2024-06-28
- New structure for the project
- add SGP model
- add model_FAQ

## 2024-02-21
- Adding demo
- add gp_computation_pack to handle all common computation in GP
- add new gp models

## 2023-11-26
Code refactored with New architecture!
- using modual kernel. No more parameter passing. When a kernel is initialized, it will automatically create the parameters it needs. User can easily define their own way of creating kernels.
- new GP will use two key components: kernel and mean function. The likelihood tied to the GP and not spedified by the user. A GP model should have it own likelihood function.

## 2020-12-15
A painless GP implementation library for educational purpose.

Files explanation:
sgp.ipynb: Simple GP Implementation in an hour (GP tutorial-E02)
sgp_chs.ipynb: Simple GP Implementation in an hour in Chinese (GP tutorial-E02)

cigp_v10: conditional independent GP (cigp) for multivariate output.
cigp_dkl: CIGP with deep kernel learning (DKL), which pass the input X through a NN before reaching a normal GP.
cigp_v12: CIGP with a given mean function
cigp_v14: CIGP with a mixture of kernels

gpdm_v2: Gaussian Process Dynamical Model (Wang et al., 2005), which contains 2 GPs to handle dynamics and latent-to-observation mapping respectively. Version 2 contains a mixture of kernels.
