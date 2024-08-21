# MiniGP
MiniGP is a minimalistic Gaussian Process (GP) library focused on regression tasks. It is designed to be simple and easy to understand, super lightweight, and friendly to researchers and developers.

## Motivation
Despite that there are many successful GP libraries, such as GPy, GPflow, and GPyTorch we find them difficult to use for beginners and very time-consuming to customize. It will take a lot of time to understand the structure of the library and the GP model, by which time the user (young research student) may give up. 

Thus we want to create a simple and easy-to-use GP library that can be used by anyone. MiniGP is designed to be simple and easy to understand. It is a great tool for educational purposes. We also try to make it easy for anyone to use without a lot of background knowledge of GP.

## Useful and practical GP Models:
- [CIGP](https://github.com/IceLab-X/Mini-GP/blob/6899d3fb947293122d758fb6ef4dd4799a799eac/core/cigp.py): simple yet accurate multi-output regression model with complexity $O(n^3 d)$ for n training points with d outputs.
- [NeuralKernel](https://github.com/IceLab-X/Mini-GP/blob/64873663f7efb63de9a6f33d1de207e7a2db1f5d/GPmodels_Classic/04_neuralKernelGP.ipynb): automatic kernel learning with neural network structured kernel.
- [GPU&GP](https://github.com/IceLab-X/Mini-GP/blob/64873663f7efb63de9a6f33d1de207e7a2db1f5d/GPmodels_Advance/01_GP&GPU_GPTutorial.ipynb): Inference method that leverage GPU acceleration for GP model. This can be used in sparse GP model to speed up the computation for large inducing point number.
- [AutoGP](https://github.com/IceLab-X/Mini-GP/blob/64873663f7efb63de9a6f33d1de207e7a2db1f5d/core/autoGP.py): A powerful GP model that incorporates a learnable input warp module, a deep neural kernel, and Sparse Gaussian Processes.


## Installation
We do not have a pip package yet. You can install it by cloning the repository and rune the code for your own purpose. At this stage we think it is better to keep it simple and customizable. It servers as rather demo code than a library, with some useful functions to make computation easier.

To start using MiniGP, you can clone the repository by running the following command:
```bash
git clone
```
You can start by running the [Demo.ipynb](https://github.com/IceLab-X/Mini-GP/blob/bf66c980d55934d037992cd70625bd692ea02aaa/Demo.ipynb) to have a taste of the library. You can also check the tutorial in the GPmodels_xxx folder to learn how to use the library.
 
Most models have two version, the API version for direct call and the tutorial version for customized usage. The API version is in the 'core' folder, and the tutorial version is in the 'GPmodels_xxx' folder.




## Structure
- **core:** This folder contains all the core functions (computing likelihood, matrix inversion, kernels, etc.) for Gaussian Processes (GP). It serves as the backbone of the library. 
Additionally, it includes API GP models (.py) that are designed to be directly called for research and experimentation. 
More details can be found in the [README](https://github.com/IceLab-X/Mini-GP/blob/64873663f7efb63de9a6f33d1de207e7a2db1f5d/core/README.md) file within the core folder.

  
  - **GPmodels_Advance:** Advance GP models, including GP with GPU acceleration, and automatic GP.
    - 01_GP&GPU_GPTutorial: Algorithms for GP leverage GPU acceleration.
    - 01_GP&GPU: Algorithms for GP leverage GPU acceleration.
    - 02_DynamicModel_GPTutorial: GP dynamic model in Chinese. ([Gaussian Process Dynamical Models](https://www.dgp.toronto.edu/~jmwang/gpdm/nips05final.pdf) )
    
  - **GPmodels_Classic:** basic GP model and its variation, such as DeepKernel GP, InputWarp GP . It demonstrates how to build a GP model with the GP_CommonCalculation.
    - 01_simpleGP_GPTutorial: simple GP tutorial in both English and Chinese. This is a good starting point for beginners.
    - 01_simpleGP, a basic GP model. It demonstrates how to build a GP model with the GP_CommonCalculation.
    - 02_deepKernelGP, a GP model with deep kernel. ([Deep Kernel Learning](https://arxiv.org/abs/1511.02222) )
    - 03_logTransformWarpGP, a GP model with log transform on the target values, this can improve the model performance when the noise does not follow Gaussian distribution.
    - 04_neuralKernelGP, a GP model with neural kernel.
    
  - **GPmodels_MultiOutput:** provides tools for implementing Gaussian Process models with multiple outputs.
    - 01_IntrinsicModel: Foundation work for multi-output GP.
    - 02_hogp_GPTutorial_Chinese: high-oreder GP in Chinese. ([Scalable High-Order Gaussian Process Regression](https://proceedings.mlr.press/v89/zhe19a.html) )
  - **GPmodels_Sparse:** provides tools for implementing sparse Gaussian Process models.
    - 01_sgpr_GPTutorial: A detailed tutorial for Sparse Gaussian Process with variational learning inducing points. ([Variational Learning of Inducing Variables in Sparse Gaussian
Processes](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf))
    - 02_svgp, A demo for implementing mini-batch gradient descent on SVGP allows training a GP with 10k inputs in 2 seconds. 
    - 02_svgp_GPTutorial, A detailed tutorial for Stochastic Variational Interference GP model that can allow mini-batch training. ([Gaussian Process for Big Data](https://arxiv.org/abs/1411.2005))
- **Debug_NaNError_FAQ:** Frequently asked questions for GP model. It contains some techniques to solve NaN problem in GP model. More details can be found in the README file in the Debug_NaNError_FAQ folder.
- **Bayesian_Optimization:** This folder contains useful tools for Bayesian optimization
    - acq: A Python scripy including several widely used acquisition functions. 
    - BO_demo: A demonstration of the process of Bayesian optimization.
    <img src="https://github.com/IceLab-X/Mini-GP/blob/bf66c980d55934d037992cd70625bd692ea02aaa/asset/Bayesian_Optimization.png" />
- **experiment:** This folder contains the python scripts of the experiment.
  
- **asset:** This folder contains the result of the experiment in both .csv and .png format. 

    <img src="https://github.com/IceLab-X/Mini-GP/blob/29a021305924757376b25905c75b36bdbdfc5017/asset/Model_comparison_autoGP.png"/>

  - **result1.csv:** The result of the regression test for different training set sizes.

    <img src="https://github.com/IceLab-X/Mini-GP/blob/29a021305924757376b25905c75b36bdbdfc5017/asset/Model_comparison_result1.png"/>

  - **result2.csv:** The result of the regression test for different numbers of inducing points.

    <img src="https://github.com/IceLab-X/Mini-GP/blob/29a021305924757376b25905c75b36bdbdfc5017/asset/Model_comparison_result2.png"/>
    
## Contribution
We welcome any contribution to this project. Please feel free to open an issue or pull request. We are happy to discuss any ideas with you. 
