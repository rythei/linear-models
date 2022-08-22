---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Singular Value Decomposition

In the previous workbook, we saw that symmetric square matrices $\boldsymbol{A}$ have a special decomposition called an _eigenvalue_ decomposition: $\boldsymbol{A} = \boldsymbol{V\Lambda V}^\top$, where $\boldsymbol{V}$ is an orthogonal matrix satisfying $\boldsymbol{V^\top V} = \boldsymbol{VV}^\top = \boldsymbol{I}$, whose columns are _eigenvectors_, and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1,\dots, \lambda_n)$ is a diagonal matrix containing the _eigenvalues_ of $\boldsymbol{A}$.

In this section, we see that _all_ matrices -- even non-square and non-symmetric -- have a similar decomposition called the _singular value decomposition_, or SVD. Let's first remind ourselves why we can't eigenvalue decomposition doesn't make sense for non-square matrices. Suppose $\boldsymbol{A}$ is a $m\times n$ matrix. Then for any $\boldsymbol{v}\in \mathbb{R}^n$, $\boldsymbol{Av}$ is a vector in $\mathbb{R}^m$, and so the eigenvalue condition $\boldsymbol{Av} = \lambda \boldsymbol{v}$ does not make sense in this setting: the left-hand side is a $m$-dimensional vector, while $\lambda \boldsymbol{v}$ is an $n$-dimensional vector.

Instead, for $m\times n$ matrices $\boldsymbol{A}$, we consider instead a generalized version of the eigenvalue condition: vectors $\boldsymbol{v}\in \mathbb{R}^n$, $\boldsymbol{u}\in \mathbb{R}^m$ and a number $\sigma$ are called _right and left singular vectors_, and a _singular value_ if they satisfy:


$$
\begin{aligned}
\boldsymbol{Av} = \sigma \boldsymbol{u} && (1)\\
\boldsymbol{A^\top u} = \sigma \boldsymbol{v} && (2)
\end{aligned}
$$


Singular values and singular vectors are in fact closely related to eigenvalues and eigenvectors. Let's see why this is the case. Let's start with equation $(1)$, and multiply both sides by $\boldsymbol{A}^\top$:


$$
\boldsymbol{Av} = \sigma \boldsymbol{u} \implies \boldsymbol{A^\top A v} = \sigma \boldsymbol{A^\top u}.
$$


Now, let's plug in equation $(2)$, which says that $\boldsymbol{A^\top u} = \sigma \boldsymbol{v}$. We get:


$$
\boldsymbol{A^\top A v} = \sigma \boldsymbol{A^\top u} = \sigma^2 \boldsymbol{v}.
$$


This looks more like something we've seen before: if we set $\boldsymbol{B} = \boldsymbol{A^\top A}$ and $\lambda = \sigma^2$, this can be written as $\boldsymbol{Bv} = \lambda \boldsymbol{v}$. Therefore, the squared singular values and right singular vectors can be obtained by computing an eigenvalue decompostion of the symmetric matrix $\boldsymbol{A^\top A}$. Using a similar derivation, we can also show that


$$
\boldsymbol{AA^\top u} = \sigma^2 \boldsymbol{u}
$$


from which we see that $u$ is really an eigenvector of the symmetric matrix $AA^\top$.

**Remark:** In our discussion above, we saw that the eigenvalues of $\boldsymbol{AA}^\top$ and/or $\boldsymbol{A^\top A}$ correspond to the _squared_ singular values $\sigma^2$ of $\boldsymbol{A}$. This may seem odd, since we know that in general matrices may have positive or negative eigenvalues. However, this occurs specifically because $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ are always _positive semi-definite_, and therefore always have non-negative eigenvalues. To see why this is true, note that the smallest eigenvalue of $\boldsymbol{A^\top A}$ are the minimum of the quadratic form $Q(\boldsymbol{x}) = \boldsymbol{x^\top A^\top A x}$, over all unit vectors $\boldsymbol{x}$. Then:


$$
\lambda_{\text{min}} = \min_{\|\boldsymbol{x}\|_2 =1} \boldsymbol{x^\top A^\top A x} = \min_{\|\boldsymbol{x}\|_2 =1} (\boldsymbol{Ax})^\top \boldsymbol{Ax} = \min_{\|\boldsymbol{x}\|_2 =1}\|\boldsymbol{Ax}\|_2^2 \geq 0.
$$


A similar derivation shows that all the eigenvalues of $\boldsymbol{AA}^\top$ are non-negative.

How many singular values/vectors do we expect to get for a given $m\times n$ matrix $\boldsymbol{A}$? We know that the matrix $\boldsymbol{A^\top A}$ is $n\times n$, which gives us $n$ eigenvectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_n$ (corresponding to $n$ right singular vectors of $\boldsymbol{A}$), and $\boldsymbol{AA}^\top$ is $m\times m$, giving us $m$ eigenvectors $\boldsymbol{u}_1,\dots, \boldsymbol{u}_m$ (corresponding to $m$ left singular vectors of $\boldsymbol{A}$). The matrices $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ will of course not have the same number of eigenvalues, though they do always have the same _non-zero_ eigenvalues. The number $r$ of nonzero eigenvalues of $\boldsymbol{A^\top A}$ and/or $\boldsymbol{AA}^\top$  is exactly equal to the _rank_ of $\boldsymbol{A}$, and we always have that $r \leq \min(m,n)$.

Now let's collect the vectors $\boldsymbol{u}_1,\dots, \boldsymbol{u}_m$ into an $m\times m$ matrix $\boldsymbol{U} = \begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_m\end{bmatrix}$ and likewise with $\boldsymbol{v}_1,\dots, \boldsymbol{v}_n$ into the $n\times n$ matrix $\boldsymbol{V} = \begin{bmatrix}\boldsymbol{v}_1 &\cdots & \boldsymbol{v}_n\end{bmatrix}$. Note that since $\boldsymbol{U}$ and $\boldsymbol{V}$ come from the eigenvalue decompositions of the symmetric matrices $\boldsymbol{AA}^\top$ and $\boldsymbol{A^\top A}$, we have that $\boldsymbol{U}$ and $\boldsymbol{V}$ are always orthogonal, satisfying $\boldsymbol{U^\top U} = \boldsymbol{UU}^\top = \boldsymbol{I}$ and $\boldsymbol{V^\top V} = \boldsymbol{VV}^\top = \boldsymbol{I}$.

Then let's define the $m\times n$ matrix $\boldsymbol{\Sigma}$ as follows:


$$
\boldsymbol{\Sigma}_{ij} = \begin{cases}\sigma_i & \text{if } i=j\\ 0  & \text{if } i\neq j\end{cases}
$$


That is, $\boldsymbol{\Sigma}$ is a "rectangular diagonal" matrix, whose diagonal entries are the singular values of $\boldsymbol{A}$ -- i.e. the square roots of the eigenvalues of $\boldsymbol{A^\top A}$ or $\boldsymbol{AA}^\top$. For example, in the $2\times 3$ case $\boldsymbol{\Sigma}$ would generically look like


$$
\begin{bmatrix}\sigma_1 & 0 & 0 \\ 0 & \sigma_2 &0\end{bmatrix}
$$

and in the $3\times 2$ case it would look like


$$
\begin{bmatrix}\sigma_1 & 0  \\ 0 & \sigma_2 \\ 0 & 0\end{bmatrix}.
$$



Given the matrices $\boldsymbol{U}, \boldsymbol{\Sigma}$ and $\boldsymbol{V}$, we can finally write the full singular value decomposition of $A$:

$$
\boldsymbol{A} = \boldsymbol{U\Sigma V}^\top.
$$


This is one of the most important decompositions in linear algebra, especially as it relates to statistics, machine learning and data science.

**Remark:** Sometimes you may see a slightly different form of the SVD: the rank of $\boldsymbol{A}$ is $r\leq \min(n,m)$, we can actually remove the last $m-r$ columns of $\boldsymbol{U}$ and $n-r$ column of $\boldsymbol{V}$ (so that $\boldsymbol{U}$ is $m\times r$ and $\boldsymbol{V}$ is $n\times r$), and let $\boldsymbol{\Sigma}$ be the $r\times r$ diagonal matrix $\text{diag}(\sigma_1,\dots,\sigma_r)$. The two forms are totally equivalent, since the last $m-r$ columns of $\boldsymbol{U}$ are only multiplied by the $m-r$ zero rows at the bottom of $\boldsymbol{\Sigma}$ anyway. This form is sometimes called the "compact SVD". In this workbook, we'll assume we're working with the "standard" version, introduced above, though the compact version is sometimes better to work with in practice, especially when the matrix $\boldsymbol{A}$ is very low rank, with $r\ll m,n$.

## Computing the SVD in Python

Let's see some examples of computing the singular value decomposition in Python.

First, let's draw a random $m\times n$ matrix $\boldsymbol{A}$ to use.

```{code-cell}
import numpy as np
np.random.seed(1)

m = 5
n = 3

A = np.random.normal(size=(m,n))
```

Next, let's compute the eigenvalue decompositions of $\boldsymbol{A^\top A} = \boldsymbol{V\Lambda}_1 \boldsymbol{V}^\top$ and $\boldsymbol{AA^\top} = \boldsymbol{U\Lambda}_2 \boldsymbol{U}^\top$.

```{code-cell}
AAT = np.dot(A,A.T)
ATA = np.dot(A.T,A)

Lambda1, V = np.linalg.eig(ATA)
Lambda2, U = np.linalg.eig(AAT)
```

Of course, since $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ are of different dimensions, $\boldsymbol{\Lambda}_1$ and $\boldsymbol{\Lambda}_2$ will also be of different dimensions. However, as we mentioned above, $\boldsymbol{\Lambda}_1$ and $\boldsymbol{\Lambda}_2$ should have the same _non-zero_ entries. Let's check that this is true.

```{code-cell}
print(Lambda1.round(8))
print(Lambda2.round(8))
```

Indeed, we get the same non-zero eigenvalues, but $\boldsymbol{\Lambda}_2$ has 10 extra zero eigenvalues. Now let's form the matrix $\boldsymbol{\Sigma}$, which will be $m\times n$ matrix with $\boldsymbol{\Sigma}_{ii} = \sqrt{\lambda_i}$ and $\boldsymbol{\Sigma}_{ij} = 0$ for $i\neq j$.

```{code-cell}
Sigma = np.zeros((m,n))
for i in range(n):
    Sigma[i,i] = np.sqrt(Lambda1[i])

Sigma
```

Now we have our matrices $\boldsymbol{V},\boldsymbol{U}$ and $\boldsymbol{\Sigma}$; let's check that $\boldsymbol{A} = \boldsymbol{U\Sigma V}^\top$.

```{code-cell}
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Strangely, this doesn't give us the correct answer. The reason is that we have an issue with one of the signs of the eigenvectors: the eigenvalue is invariant to switching the signs of one of the eigenvectors (i.e. multiplying one of the columns of $V$ or $U$ by $-1$ ), but the SVD is not. Since we computed the eigenvalue decomposition of $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ separately, there was no guarantee that we would get the correct signs of the eigenvectors. It turns out in this case we can fix this by switching the sign of the third column of $\boldsymbol{V}$.

```{code-cell}
V[:,2] *= -1
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Now everything works! However, this issue is a bit annoying in practice -- fortunately, we can avoid it by simply using numpy's build in SVD function, `np.linalg.svd`. Let's see how this works.

```{code-cell}
U, S, VT = np.linalg.svd(A)

Sigma = np.zeros((m,n)) # make diagonal matrix
for i in range(n):
    Sigma[i,i] = S[i]

np.allclose(A, np.dot(U, np.dot(Sigma, VT)))
```
