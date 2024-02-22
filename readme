# MiniGP
MiniGP is a minimalistic Gaussian Process (GP) library. It is designed to be simple and easy to understand. 

Despite that there are many successful GP libraries, such as GPy, GPflow, and Pyro, they are often too complex for beginners to understand. Things get worse when the user wants to customize the model. MiniGP is designed to be simple and easy to understand. It is a great tool for educational purposes. We also try to make it easy for anyone to use without a lot of background knowledge of GP.

## Installation
We do not have a pip package yet. You can install it by cloning the repository and rune the code for your own purpose. At this stage we thing it is better to keep it simple and customizable. It servers as rather demo code than a library, with some useful functions to make computation easier.

## Structure
- GPTutorial: our signature tutorial for GP. It is a step-by-step tutorial to build a GP model from scratch. It is a great way to understand the basic of GP. Most of the tutorial is self-contained and can be used as a standalone code.
  - GPTutorial_basicGP: a basic GP model. 
  - GPTutorial_basicGP_chinese: a basic GP model in Chinese.
  - GPTutorial_hogp_chinese: high-oreder GP [(HOGP)](https://proceedings.mlr.press/v89/zhe19a.html) in Chinese.
  - GPTutorial_DynamicModel: GP dynamic model [(GPDM)](https://www.dgp.toronto.edu/~jmwang/gpdm/nips05final.pdf) in Chinese.

- Key components
  - kernel.py: contains all the kernel functions
  - gp_computation_pack.py: contains all the common computation in GP
  - gp_transform.py: contains all the transformation functions for GP. NOT IMPLEMENTED YET.
- Self-contained GP models.
  Several GP models that are self-contained and practical to use (we use them in our many of our research projects). 
    - gp_basic.py: basic GP model. It demonstrate how to build a GP model without using the gp_computation_pack.
    - cigp_v10.py: a practical GP model for multivariate output. It is a conditional independent GP (cigp) for multivariate output. v10 is the special version that work with many of our research projects. You can find more cigp_vxx variations in the legacy folder. The cigp_vxx is deprecated and will be removed in the future.
    - cigp_withMean.py: a simple yet efficient GP model for multivariate output. It is recommended to use this model for most problem. It is recommend to normalize the output before using this model.
  
- xxxGP_demo: a demo for a specific GP model. Normally there will be a xxxGP_demo.ipynb file and a xxxGP.py file. The xxxGP_demo.ipynb file is a tutorial for the GP model. The xxxGP.py file is the implementation of the GP model. 
- TO BE ADDED.
  - [ ] SparseGP_demo: a demo for sparse GP model
  - [ ] deepKernel_demo: a demo for deep kernel GP model
  - [ ] BigDataGP_demo: a demo for GP model that can handle big data
  - [ ] attentionGP_demo: a demo for attention GP model

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
