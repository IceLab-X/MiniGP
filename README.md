# MiniGP
MiniGP is a minimalistic Gaussian Process (GP) library. It is designed to be simple and easy to understand. 

Despite that there are many successful GP libraries, such as GPy, GPflow, and Pyro, they are often too complex for beginners to understand. Things get worse when the user wants to customize the model. MiniGP is designed to be simple and easy to understand. It is a great tool for educational purposes. We also try to make it easy for anyone to use without a lot of background knowledge of GP.

## Installation
We do not have a pip package yet. You can install it by cloning the repository and rune the code for your own purpose. At this stage we think it is better to keep it simple and customizable. It servers as rather demo code than a library, with some useful functions to make computation easier.

## Structure
- **core:** This folder contains all the core functions for Gaussian Processes (GP). It serves as the backbone of the library. Additionally, it includes Python scripts for GP models that are designed to be easy and quick to use for research and experimentation. The folder also contains a model comparison script and the corresponding results. More details can be found in the README file within the core folder.

<!-- - Self-contained GP models and signiture GPTutorials for educational purposes.
  Several GP models that are self-contained and practical to use (we use them in many of our research projects).  -->
  
  - **GPmodels_Advance:** Advance GP models, including GP with GPU acceleration, and automatic GP.
    - 01_GP&GPU: a GP model leverage GPU acceleration.
    - 01_GP&GPU_GPTutorial: Algorithms for GP leverage GPU acceleration.
    - 02_DynamicModel_GPTutorial: GP dynamic model in Chinese. [Original paper](https://www.dgp.toronto.edu/~jmwang/gpdm/nips05final.pdf) 
    
  - **GPmodels_Classic:** basic GP model and its variation, such as DeepKernel GP, InputWarp GP . It demonstrates how to build a GP model with the GP_CommonCalculation.
    - 01_simpleGP_GPTutorial: simple GP tutorial in both English and Chinese. This is a good starting point for beginners.
    - 01_simpleGP, a basic GP model. It demonstrates how to build a GP model with the GP_CommonCalculation.
    - 02_deepKernelGP, a GP model with deep kernel. [Original paper](https://arxiv.org/abs/1511.02222) 
    - 03_logTransformWarpGP, a GP model with log transform on the target values, this can improve the model performance when the noise does not follow Gaussian distribution.
    - 04_neuralKernelGP, a GP model with neural kernel.
    
  - **GPmodels_MultiOutput:** provides tools for implementing Gaussian Process models with multiple outputs.
    - 01_IntrinsicModel: Foundation work for multi-output GP.
    - 02_hogp_GPTutorial_Chinese: high-oreder GP in Chinese. [Original paper](https://proceedings.mlr.press/v89/zhe19a.html) 
  - **GPmodels_Sparse:** provides tools for implementing sparse Gaussian Process models.
    - 01_sgpr_GPTutorial: A detailed tutorial for Sparse Gaussian Process with variational learning inducing points. [Original paper](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)
    - 02_svgp, A demo for implementing mini-batch gradient descent on SVGP allows training a GP with 10k inputs in 2 seconds. 
    - 02_svgp_GPTutorial, A detailed tutorial for Stochastic Variational Interference GP model that can allow mini-batch training. [Original paper](https://arxiv.org/abs/1411.2005)
- **Debug_NaNError_FAQ:** Frequently asked questions for GP model. It contains some techniques to solve NaN problem in GP model. More details can be found in the README file in the Debug_NaNError_FAQ folder.
- **Bayesian_Optimization:** This folder contains useful tools for Bayesian optimization
    - acq: A Python scripy including several widely used acquisition functions. 
    - BO_demo: A demonstration of the process of Bayesian optimization.


    
## Contribution
We welcome any contribution to this project. Please feel free to open an issue or pull request. We are happy to discuss any ideas with you. 
