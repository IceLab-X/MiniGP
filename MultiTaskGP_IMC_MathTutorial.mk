Gaussaian process regression tutorial with structured data
=========================================================
<!-- add author -->
Wei W. Xing
2024-01-27


# Introduction
In this tutorial, we will discuss how to reduce te computational complexity when using Gaussian process to deal with structure data.

## Definiaion of structure data
Structure data we discussed here is the data with some kind of structure, such as spatial-temporal data. The key thing is that they have a repeat corresponding input pattern.

<!-- For example, let say we have $n$ sensors with location $\mathbf{z}_i$ for $i = 1, 2, \cdots, n$.
Each sensor measure some target value at time $t_j$ for $j = 1, 2, \cdots, m$, which results into observation data that can be written as a matrix $\mathbf{Y}$ of size $n \times m$. -->

For example, let say we have $m$ sensors with location $\mathbf{z}_j$ for $j = 1, 2, \cdots, m$, and each sensor measure some target value at time $t_i$ for $i = 1, 2, \cdots, n$, which results into observation data that can be written as a matrix $\mathbf{Y}$ of size $n \times m$.


Now for a simple GP implementation, we can simply vectorize the data and use a new input $\mathbf{x} = [\mathbf{z}, t]$ to represent the location and time to learn the input-output relationship. However, this will result in a huge covariance matrix $\mathbf{K}$ of size $nm \times nm$.

## Structure data and Kronecker product
The key to reduce the computational complexity is to use the Kronecker product. The Kronecker product of two matrices $\mathbf{A}$ and $\mathbf{B}$ is defined as
$$
\mathbf{A} \otimes \mathbf{B} = \begin{bmatrix}
a_{11} \mathbf{B} & a_{12} \mathbf{B} & \cdots & a_{1n} \mathbf{B} \\
a_{21} \mathbf{B} & a_{22} \mathbf{B} & \cdots & a_{2n} \mathbf{B} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} \mathbf{B} & a_{m2} \mathbf{B} & \cdots & a_{mn} \mathbf{B} \\
\end{bmatrix}
$$
where $\mathbf{A}$ is a $m \times n$ matrix and $\mathbf{B}$ is a $p \times q$ matrix. The Kronecker product of two matrices is a block matrix of size $mp \times nq$.

The Kronecker product has two important properties for matrix inversion, which is
$$
(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}
$$
<!-- kronecker product multiply vectorized y -->
$$
(\mathbf{A} \otimes \mathbf{B}) vec(\mathbf{Y}) = vec(\mathbf{B} \mathbf{Y} \mathbf{A}^T)
$$
<!-- $$
(\mathbf{A} \otimes \mathbf{B})^{-1} vec(\mathbf{Y}) = vec(\mathbf{A}^{-1} \mathbf{Y} \mathbf{B}^{-1})
$$ -->
where $\mathbf{A}$ and $\mathbf{B}$ are invertible matrices, $\mathbf{Y}$ is a matrix of size $m \times n$, and $vec(\mathbf{Y})$ is the vectorized version of $\mathbf{Y}$.



# Gauaaisn process likelihood
Let $\mathbf{y}$ denoted the vectorized observation data, $\mathbf{X}$ denoted the vectorized input data.

The likelihood of the Gaussian process is
$$
\mathcal{N}(vec(\mathbf{Y}) | \mathbf{0}, \mathbf{K} + \sigma^2 \mathbf{I})
$$
where $\mathbf{K}$ is the covariance matrix of the input data $\mathbf{X}$, and $\sigma^2$ is the noise variance.
When dealing with structured data like the spatial-temporal data mentioned earlier, we can exploit the structure to simplify the computation of the Gaussian Process likelihood. Instead of working with the large covariance matrix $\mathbf{K}$ of size $nm \times nm$, we can represent it more efficiently using the Kronecker product.

## Decomposing the Covariance Matrix

For spatial-temporal data, the covariance matrix $\mathbf{K}$ can be decomposed into two smaller matrices: $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$, representing the spatial and temporal components, respectively. The full covariance matrix can then be represented as:

$$
\mathbf{K} = \mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}}
$$

where $[\mathbf{K}_{\text{space}}]_{ij} = k_{\text{space}}(\mathbf{z}_i, \mathbf{z}_j)$ is a $m \times m$ matrix with $k_{\text{space}}(\mathbf{z}_i, \mathbf{z}_j)$ being the spatial kernel function. Similarly, $[\mathbf{K}_{\text{time}}]_{ij} = k_{\text{time}}(t_i, t_j)$ is a $n \times n$ matrix with $k_{\text{time}}(t_i, t_j)$ being the temporal kernel function.


## Efficient Computation of Likelihood

The likelihood function can be rewritten using this decomposed covariance matrix. This significantly reduces the computational complexity, especially when calculating the inverse and determinant of $\mathbf{K}$:

$$
p( vec(\mathbf{Y})) = \mathcal{N}(vec(\mathbf{Y}) | \mathbf{0}, \mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})
$$

```
[Important] The order of $K_{\text{space}}$ and $K_{\text{time}}$ in the Kronecker product is important. The above equation is correct. If we swap the order, the equation will be wrong. 
```

The likelihood computation can be further simplified by using the Kronecker product property:
$$ 
p( vec(\mathbf{Y})) = -\frac{1}{2} vec(\mathbf{Y})^T (\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} vec(\mathbf{Y})\\- \frac{1}{2} \log |\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I}| - \frac{nm}{2} \log 2 \pi 
$$

The first trick is to write $(\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1}$ as a Kronecker product of two smaller matrices.
Denote $\mathbf{K}_{\text{space}}^{-1}$ and $\mathbf{K}_{\text{time}}^{-1}$ as the inverse of $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$, respectively. 

Let's do eigendecomposition on $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$:
$$
\mathbf{K}_{\text{space}} = \mathbf{U}_{\text{space}} \mathbf{\Lambda}_{\text{space}} \mathbf{U}_{\text{space}}^T \\
\mathbf{K}_{\text{time}} = \mathbf{U}_{\text{time}} \mathbf{\Lambda}_{\text{time}} \mathbf{U}_{\text{time}}^T
$$
where $\mathbf{\Lambda}_{\text{space}}$ and $\mathbf{\Lambda}_{\text{time}}$ are diagonal matrices with eigenvalues of $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$ on the diagonal, respectively. $\mathbf{U}_{\text{space}}$ and $\mathbf{U}_{\text{time}}$ are the corresponding eigenvectors.
The computational complexity of eigendecomposition is $O(n^3)$ and $O(m^3)$ for $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$, respectively.


The inversion of $(\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1}$ can be simplified as:
$$
(\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} \\ = (\mathbf{U}_{\text{space}} \mathbf{\Lambda}_{\text{space}} \mathbf{U}_{\text{space}}^T \otimes \mathbf{U}_{\text{time}} \mathbf{\Lambda}_{\text{time}} \mathbf{U}_{\text{time}}^T + \sigma^2 \mathbf{I})^{-1} \\ = (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T
$$
The computational complexity of this inversion is $O(n^2 m^2)$ + $O(n^3)$ + $O(m^3)$ (for the eigendecomposition), which is much smaller than the $O(n^3 m^3)$ complexity of the original inversion.

Solving this inversion, the computation of the likelihood and predictive distribution can be done very efficiently.

# Saving the memory usage
The previous section shows how to reduce the computational complexity of the Gaussian Process likelihood. However, the memory complexity is still $O(n^2 m^2)$, which is still too large for large datasets.
Now putting the new inversion equation into the first term of the likelihood function, we have
$$
 vec(\mathbf{Y})^T (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T vec(\mathbf{Y}) \\
 = vec(\mathbf{A}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} vec(\mathbf{A})^T
$$
where $ \mathbf{A} =\mathbf{U}_{\text{time}}^T \mathbf{Y} \mathbf{U}_{\text{space}} $.

To derive this, remember the Kronecker product property at the beginning, $(\mathbf{A} \otimes \mathbf{B}) vec(\mathbf{Y}) = vec(\mathbf{B} \mathbf{Y} \mathbf{A}^T)$, we can rewrite 

$$
(\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T vec(\mathbf{Y}) \\ =
(\mathbf{U}_{\text{space}}^T \otimes \mathbf{U}_{\text{time}}^T) vec(\mathbf{Y}) \\ =
vec( (\mathbf{U}_{\text{time}}^T \mathbf{Y} \mathbf{U}_{\text{space}})
$$

Here $\mathbf{A}$ is a $n \times m$ matrix, and $vec(\mathbf{A})$ is a $nm \times 1$ vector. The memory complexity is reduced to $O(nm)$ instead of build the matrix $\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}$ with size $nm \times nm$.

For the log determinant term, we can use the Kronecker product property to simplify it as well:

$$
\log |\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I}| \\ =
\log |\mathbf{U}_{\text{space}} \mathbf{\Lambda}_{\text{space}} \mathbf{U}_{\text{space}}^T \otimes \mathbf{U}_{\text{time}} \mathbf{\Lambda}_{\text{time}} \mathbf{U}_{\text{time}}^T + \sigma^2 \mathbf{I}| \\ = \log |(\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I}) (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T|\\ = \log |(\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1}| \\
= \log | \prod_{i=1}^{nm} ([\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}}  ]_{ii} + \sigma^2)| \\ = \sum_{i=1}^{nm} \log ([\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}}  ]_{ii} + \sigma^2)
$$
where $[\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}}  ]_{ii}$ is the $i$-th diagonal element of $\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}}$ (which is a $nm \times nm$ diagonal matrix).

The simplification can be done because $\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}$ is an orthogonal matrix, which means $\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}} (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T = \mathbf{I}$, (because $\mathbf{U}_{\text{space}}$ and $\mathbf{U}_{\text{time}}$ are orthogonal matrices from the eigendecomposition of $\mathbf{K}_{\text{space}}$ and $\mathbf{K}_{\text{time}}$).

# Complexity in predictive posterior distribution (to be completed)
<!-- The predictive posterior distribution is -->
Let first consider prediction of a new time point $t_{*}$ at the same location $\mathbf{z}$ as the training data. The predictive mean vector is 
$$
\mathbf{m}_{*} = \mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^* (\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} vec(\mathbf{Y}) \\ = (\mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^*) (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T vec(\mathbf{Y}) \\ =
(\mathbf{K}_{\text{space}} \mathbf{U}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^* \mathbf{U}_{\text{time}} ) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{U}_{\text{space}} \otimes \mathbf{U}_{\text{time}})^T vec(\mathbf{Y}) \\ =
(\mathbf{K}_{\text{space}} \mathbf{U}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^* \mathbf{U}_{\text{time}}) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} vec(\mathbf{A}) \\ 
$$

where $\mathbf{k}_{\text{time}}^*$ is the vector of covariance between the new time point $t_{*}$ and all the training time points, and $\mathbf{A} = \mathbf{U}_{\text{time}}^T \mathbf{Y} \mathbf{U}_{\text{space}}$.
In computation, $(\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} vec(\mathbf{A})^T$ can be precomputed and stored in memory, which is a $nm \times 1$ vector. The memory complexity is $O(nm)$.


For computing the variance, we have
$$
\mathbf{v}_* =  \mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^{**} - (\mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^*) (\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^*)^T
$$
Let's discuss the computation of the second term first efficiently. We have
$$
(\mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^*) (\mathbf{K}_{\text{space}} \otimes \mathbf{K}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{K}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^*)^T \\= (\mathbf{K}_{\text{space}} \mathbf{U}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^* \mathbf{U}_{\text{time}} ) (\mathbf{\Lambda}_{\text{space}} \otimes \mathbf{\Lambda}_{\text{time}} + \sigma^2 \mathbf{I})^{-1} (\mathbf{K}_{\text{space}} \mathbf{U}_{\text{space}} \otimes \mathbf{k}_{\text{time}}^* \mathbf{U}_{\text{time}} )^T 
$$

