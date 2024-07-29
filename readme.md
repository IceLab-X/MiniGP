# MiniGP
MiniGP is a minimalistic Gaussian Process (GP) library. It is designed to be simple and easy to understand. 

Despite that there are many successful GP libraries, such as GPy, GPflow, and Pyro, they are often too complex for beginners to understand. Things get worse when the user wants to customize the model. MiniGP is designed to be simple and easy to understand. It is a great tool for educational purposes. We also try to make it easy for anyone to use without a lot of background knowledge of GP.

## Installation
We do not have a pip package yet. You can install it by cloning the repository and rune the code for your own purpose. At this stage we think it is better to keep it simple and customizable. It servers as rather demo code than a library, with some useful functions to make computation easier.

## Structure
- core (folder): contains all the core functions for GP. It is the backbone of the library.
  - kernel.py: contains all the kernel functions
  - GP_CommonCalculation.py: contains all the common computation in GP, normalization functions and input warping functions.

- GP_Tutorial (folder): our signature tutorial for GP. It is a step-by-step tutorial to build a GP model from scratch and related mathematical explanation. It is a great way to understand the basic of GP. Most of the tutorial is self-contained and can be used as a standalone code.
  - GPTutorial_basicGP: a basic GP model. 
  - GPTutorial_basicGP_chinese: a basic GP model in Chinese.
  - GPTutorial_hogp_chinese: high-oreder GP in Chinese. [Original paper](https://proceedings.mlr.press/v89/zhe19a.html) 
  - GPTutorial_DynamicModel: GP dynamic model in Chinese. [Original paper](https://www.dgp.toronto.edu/~jmwang/gpdm/nips05final.pdf) 
  - GPTutorial_sparseGP: sparse GP model. [Original paper](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)


<!-- - Self-contained GP models.
  Several GP models that are self-contained and practical to use (we use them in many of our research projects).  -->
  
  - GPmodels_Advance(folder): Advance GP models, including GP with GPU acceleration, and automatic GP.
    - GP&GPU: a GP model with GPU acceleration.
    - autoGP: a GP model with automatic kernel selection, data standardization, input warping and training.
   
  - GPmodels_Classic (folder): basic GP model and its variation, such as DeepKernel GP, InputWarp GP . It demonstrates how to build a GP model with the GP_CommonCalculation.
    - simpleGP, a basic GP model. It demonstrates how to build a GP model with the GP_CommonCalculation.
    - deepKernelGP, a GP model with deep kernel. Reference:[(DKL)](https://arxiv.org/abs/1511.02222) 
    - inputWarpGP, a GP model with input warping. Reference:[(IW)](https://proceedings.mlr.press/v32/snoek14.pdf)
    - neuralKernalGP, a GP model with neural kernel.
  - GPmodels_LatentVariable: provides tools for implementing Gaussian Process models with latent variables.
    - latentVariableGP, a GP model with latent variable. 
  - GPmodels_MultiOutput: provides tools for implementing Gaussian Process models with multiple outputs.
    - ConditionalIndependentGP, 
    - HighOrderGP, a GP model with high-order output.
    - IntrinsicModel,
  - GPmodels_Sparse: provides tools for implementing sparse Gaussian Process models.
    - sparseGP, a variational sparse GP model.
    - bigDataGP, a stochastic variational interference GP model that can allow mini-batch training. Reference: [(SVGP)](https://arxiv.org/abs/1411.2005)
- Model_FAQ: a FAQ for GP model. It contains some techniques to solve NAN problem in GP model.
  - Eigendecomposition: a demo to show how to replace Cholesky decomposition with Eigendecomposition .
  - non_positive_definite_fixer: a collection of techniques to solve NAN problem in GP model.
  - non_positive_definite_fixer_implement: a demo to show how to use non_positive_definite_fixer when building up your own GP model.
  - parameter_reset: a demo to show how to reset the parameters when they explode.
  - remove_similar_data: a demo to show removing similar training data fix non positive definite issue in some circumstances.
  - why_non_positive_definite: a tutorial to show why your kernel is non-positive definite.


## Contribution
We welcome any contribution to this project. Please feel free to open an issue or pull request. We are happy to discuss any ideas with you. 


[//]: # (<!-- # Todo list -->)

[//]: # (<!-- - [ ] !Test different way of kernel inversion! -->)

[//]: # (<!-- - [ ] add a testunit for kernel function -->)

[//]: # (<!-- - [ ] add a testunit for GP model -->)

[//]: # (<!-- - [ ] add GP_build which take a GP model and warp it with normalizer, optimizer, and loss function definition.)

[//]: # (- [ ] add mean function to GP model)

[//]: # (- [ ] redo previous GP models using the new architecture)

[//]: # (- [ ] add GPU enabled GP)

[//]: # (- [ ] Need HOGP -->)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (# Log)

[//]: # ()
[//]: # (## 2024-06-28)

[//]: # (- New structure for the project)

[//]: # (- add SGP model)

[//]: # (- add model_FAQ)

[//]: # ()
[//]: # (## 2024-02-21)

[//]: # (- Adding demo)

[//]: # (- add gp_computation_pack to handle all common computation in GP)

[//]: # (- add new gp models)

[//]: # ()
[//]: # (## 2023-11-26)

[//]: # (Code refactored with New architecture!)

[//]: # (- using modual kernel. No more parameter passing. When a kernel is initialized, it will automatically create the parameters it needs. User can easily define their own way of creating kernels.)

[//]: # (- new GP will use two key components: kernel and mean function. The likelihood tied to the GP and not spedified by the user. A GP model should have it own likelihood function.)

[//]: # ()
[//]: # (## 2020-12-15)

[//]: # (A painless GP implementation library for educational purpose.)

[//]: # ()
[//]: # (Files explanation:)

[//]: # (sgp.ipynb: Simple GP Implementation in an hour &#40;GP tutorial-E02&#41;)

[//]: # (sgp_chs.ipynb: Simple GP Implementation in an hour in Chinese &#40;GP tutorial-E02&#41;)

[//]: # ()
[//]: # (cigp_v10: conditional independent GP &#40;cigp&#41; for multivariate output.)

[//]: # (cigp_dkl: CIGP with deep kernel learning &#40;DKL&#41;, which pass the input X through a NN before reaching a normal GP.)

[//]: # (cigp_v12: CIGP with a given mean function)

[//]: # (cigp_v14: CIGP with a mixture of kernels)

[//]: # ()
[//]: # (gpdm_v2: Gaussian Process Dynamical Model &#40;Wang et al., 2005&#41;, which contains 2 GPs to handle dynamics and latent-to-observation mapping respectively. Version 2 contains a mixture of kernels.)
