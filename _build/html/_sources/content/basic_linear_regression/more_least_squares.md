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

# More on least squares

In the last section, we derived and implemented the solution to the least squares problem

$$
\min_{\boldsymbol{\beta}} \|\boldsymbol{y}-\boldsymbol{X\beta}\|_2^2 \hspace{10mm} (1)
$$

which we found (with some calculus) to be satisfied by any vector $\boldsymbol{\beta}$ satisfying the _normal equations_:

$$
\boldsymbol{X^\top X\beta} = \boldsymbol{X^\top y}. \hspace{10mm} (2)
$$

In this section, we investigate the properties of this solution a bit more, emphasizing the linear algebraic properties of this problem and its solution(s).

## The linear algebra of the least squares solution

In the simplest case, when the data matrix $\boldsymbol{X^\top X}$ is invertible, the least squares problem $(1)$ has a unique solution, obtained by multiplying either side of $(2)$ by $(\boldsymbol{X^\top X})^{-1}$:

$$
\hat{\boldsymbol{\beta}} = (\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top y}.
$$

In this case, we can easily find the fitted values $\hat{\boldsymbol{y}}$ by simply multiplying $\hat{\boldsymbol{\beta}}$ on the left by $\boldsymbol{X}$. This gives

$$
\hat{\boldsymbol{y}} = \boldsymbol{X}\hat{\boldsymbol{\beta}} = \boldsymbol{X}(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top y}.
$$

In the context of linear regression, the matrix $\boldsymbol{X}(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top}$ is sometimes called the "hat matrix", since it "puts the hat on" the vector $\boldsymbol{y}$. However, this matrix is also of more general interest in linear algebra: it is a special matrix which performs a very particular opertation. The matrix $\boldsymbol{P} = \boldsymbol{X}(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top}$ is called the projection onto the _column space_ of $\boldsymbol{X}$. To understand what this means, we must first define what we mean by the column space.

Given a set of vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_p$, the _span_ of $\boldsymbol{v}_1,\dots, \boldsymbol{v}_p$ is the vector space given by all linear combinations of these vectors:

  $$
\text{span}(\boldsymbol{v}_1,\dots,\boldsymbol{v}_p) = \left\{\boldsymbol{x} : \boldsymbol{x} = \sum_{j=1}^p \alpha_j \boldsymbol{v}_j,\; \text{for some } \alpha_1,\dots,\alpha_p \in \mathbb{R}\right\}
  $$

Then given a matrix $\boldsymbol{X}$ with column $\boldsymbol{x}_1,\dots, \boldsymbol{x}_p$, the column space of $\boldsymbol{X}$, denoted $\text{col}(\boldsymbol{X})$ is simply the $\boldsymbol{x}_1,\dots, \boldsymbol{x}_p$. The _projection_ onto the column space $\boldsymbol{X}$ is the matrix $\boldsymbol{P}$ which, when applied to a vector $\boldsymbol{y}$, returns the nearest vector in $\text{col}(\boldsymbol{X})$.

In 2-d, there is a simple picture which illustrates this concept. Imagine a single vector, say, $\boldsymbol{x} = \begin{bmatrix}1\\ 1\end{bmatrix}$, which we think of as a matrix with a single column. Then the column space of $\boldsymbol{x}$ is simply the set of all vectors of the form $\alpha \begin{bmatrix}1\\ 1\end{bmatrix} = \begin{bmatrix}\alpha \\ \alpha \end{bmatrix}$ for $\alpha \in \mathbb{R}$. Visually, this corresponds to a line through the origin with slope 1, i.e. the $y=x$ line. Let's visualize this in python.

```
import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(-2,2,10)

plt.plot(xx, xx, label='span([1,1])')
plt.legend()
plt.axis("equal")
plt.show()
```

Now let's consider another vector in $\mathbb{R}^2$, say $\boldsymbol{u}= \begin{bmatrix}-1\\ 1\end{bmatrix}$, which we can add to this plot as a point.

```
plt.scatter([-1], [1], color='red', label='[2,1]')
plt.plot(xx, xx, label='span([1,1])')
plt.legend()
plt.axis("equal")
plt.show()
```

The (orthogonal) projection of $\boldsymbol{u}$ onto $\text{span}(\boldsymbol{x})$ is defined to be the nearest $\boldsymbol{u}' \in \text{span}(\boldsymbol{x})$ to $\boldsymbol{u}$. Formally,

$$
\min_{\boldsymbol{u}' \in \text{span}(\boldsymbol{x})} \|\boldsymbol{u} - \boldsymbol{u}'\|_2^2 = \min_{\alpha \in \mathbb{R}} \|\boldsymbol{u} - \alpha \boldsymbol{x}\|_2^2.
$$

This now looks just like a toy version of our least squares problem! The solution is given by

$$
\hat{\boldsymbol{u}}' = \frac{\boldsymbol{xx}^\top}{\boldsymbol{x^\top x}}\boldsymbol{u}
$$

which corresponds to

$$
\hat{\alpha} = \frac{\boldsymbol{x^\top u}}{\boldsymbol{x^\top x}}.
$$

In our example, $\boldsymbol{x} = \begin{bmatrix}1\\ 1\end{bmatrix}$ and $\boldsymbol{u} = \begin{bmatrix}-1\\ 1\end{bmatrix}$, so

$$
\hat{\alpha} = 0
$$

and therefore the projection of $\boldsymbol{u}$ onto $\text{span}(\boldsymbol{x})$ is given by

$$
\hat{\boldsymbol{u}}' = \hat{\alpha}\boldsymbol{x} = \begin{bmatrix}0\\ 0\end{bmatrix}
$$

Let's visualize this on our plot.

```
plt.scatter([-1], [1], color='red', label='[-1,1]')
plt.scatter([0], [0], color='green', label='projection of [-1,1] onto span([1,1])')
plt.plot(xx, xx, label='span([1,1])')
plt.legend()
plt.axis("equal")
plt.show()
```

In the context of the least squares problem this means the following: the fitted values $\hat{\boldsymbol{y}}$ are the _projection_ of the response vector $\boldsymbol{y}$ onto the span of the columns of $\boldsymbol{X}$. This fact is important not only for mathematical intuition, but because it can help us design methods for solving least squares problems in practice. We discuss one such method next.

## Solving the least squares problem using the QR decomposition

The largest computational burden in performing least squares comes from computing the inverse of the matrix $\boldsymbol{X^\top X}$. In particular when the number of features $p$ is large, this can be expensive, numerically unstable or not otherwise not feasible. In these situations, we would like to have algorithms to find $\hat{\boldsymbol{\beta}}$ (or perhaps at least the fitted values $\hat{\boldsymbol{y})}$ without having to explicitly compute $(\boldsymbol{X^\top X})^{-1}$.

In this section, we show that the _QR decomposition_ can be used to find both $\hat{\boldsymbol{\beta}}$ and $\boldsymbol{y}$ without ever having to explicitly compute the matrix $(\boldsymbol{X^\top X})^{-1}$. The QR decomposition of an $n\times p$ matrix $\boldsymbol{X}$ is a decomposition of $\boldsymbol{X}$ into two matrices $\boldsymbol{Q}, \boldsymbol{R}$ such that $\boldsymbol{X} = \boldsymbol{QR}$ and $\boldsymbol{R}$ is an upper triangular matrix, and $\boldsymbol{Q}$ is an _orthogonal_ matrix such that $\boldsymbol{Q^\top Q} = \boldsymbol{I}$ (the identity matrix.) 

## What happens when $\boldsymbol{X^\top X}$ is not invertible?
