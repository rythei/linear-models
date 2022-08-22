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

# The Eigenvalue decomposition for special types of matrices

In the previous section, we saw that the general eigenvalue problem for an $n\times n$ matrix $\boldsymbol{A}$ involves finding $n$ value vector pairs $(\lambda_1, \boldsymbol{u}_1),\dots, (\lambda_n, \boldsymbol{u}_n)$ satisfying

$$
\boldsymbol{Au}_i = \lambda_i \boldsymbol{u}_i.
$$

where we find the values $\lambda_i$ by finding the roots of the _characteristic polynomial_ $p(\lambda) = \det(\boldsymbol{A} - \lambda \boldsymbol{I})$, and we find the associated eigenvectors by solving the linear system $(\boldsymbol{A}-\lambda \boldsymbol{I})\boldsymbol{u} = 0$.

If we form the eigenvector into the columns of an $n\times n$ matrix $\boldsymbol{U} = \begin{bmatrix} \boldsymbol{u_1} & \dots & \boldsymbol{u}_n \end{bmatrix}$ and the eigenvalues into a diagonal matrix $\boldsymbol{\Lambda} = \text{diag}(\lambda_1,\dots,\lambda_n)$, then this can equivalently be expressed as

$$
\boldsymbol{AU} = \boldsymbol{U\Lambda} \implies \boldsymbol{A} = \boldsymbol{U\Lambda U}^{-1} \,\,\,\, \text{(assuming $\boldsymbol{U}$ invertible)}
$$

This form is typically called the _eigenvalue decomposition_ of the matrix $\boldsymbol{A}$. In this section, we will see special cases of this decomposition resulting from special types of matrices $\boldsymbol{A}$. The most important of these is the case of symmetric matrices $\boldsymbol{A}$, which we discuss next.

## Symmetric matrices

In general, the eigenvalues and eigenvectors of a matrix may be real or complex valued, as they come from the roots of polynomial. For example, in the $2 \times 2$ case, with

$$
\boldsymbol{A} = \begin{bmatrix} a_{11}& a_{12}\\ a_{21} & a_{22}\end{bmatrix}
$$

the characteristic polynomial will be $p(\lambda) = \lambda^2 - (a_{11}+a_{22})\lambda + a_{11}a_{22}-a_{12}a_{21}$, which, by the quadratic formula, will have complex roots (and hence complex eigenvalues) whenever $(a_{11}+a_{22})^2 - 4(a_{11}a_{22}-a_{12}a_{21}) < 0$. On the other hand, if $\boldsymbol{A}$ is _symmetric_, so that $\boldsymbol{A}^\top = \boldsymbol{A}$, then the eigenvalues and eigenvectors of $\boldsymbol{A}$ will always satisfy the following two properties:

- The eigenvalues of a symmetric matrix $\boldsymbol{A}$ are always real numbers.
- The eigenvectors of a symmetric matrix $\boldsymbol{A}$ are always orthogonal.

In particular, the latter condition means that the matrix $\boldsymbol{U}$ of eigenvectors is an _orthogonal matrix_ satisfying $\boldsymbol{U^\top U} = \boldsymbol{I}$ and $\boldsymbol{UU^\top} = \boldsymbol{I}$ (since $\boldsymbol{U}$ is square). This means that $\boldsymbol{U}^{-1} = \boldsymbol{U}^\top$, and so in the symmetric case we can simplify the eigenvalue decomposition:

$$
\boldsymbol{A} = \boldsymbol{U\Lambda U}^\top.
$$

Here we briefly provide proofs for both of the above statements about the eigenvalues and eigenvectors of symmetric matrices $\boldsymbol{A}$.

**Eigenvalues are real numbers.** Let's quickly see why symmetric matrices always have real-valued eigenvalues. For a number/vector $x$, let $\bar{x}$ denote its complex conjugate (for a vector, this is just the complex conjugate in each coordinate). Then for any real matrix $\boldsymbol{A}$, let $(\lambda, \boldsymbol{v})$ be an eigenvalue/eigenvector pair of $\boldsymbol{A}$. Since $\boldsymbol{A}$ is real-valued, we have that $\overline{\boldsymbol{Av}} = \bar{\lambda}\bar{\boldsymbol{v}}$. Then if $\boldsymbol{A}=\boldsymbol{A}^\top$, we have

$$
\bar{\boldsymbol{v}}^\top \boldsymbol{A} \boldsymbol{v} = \bar{\boldsymbol{v}}^\top (\lambda \boldsymbol{v}) = \lambda \bar{\boldsymbol{v}}^\top \boldsymbol{v}
$$

and

$$
\bar{\boldsymbol{v}}^\top \boldsymbol{A v} = \bar{\boldsymbol{v}}^\top \boldsymbol{A^\top v} = (\boldsymbol{A}\bar{\boldsymbol{v}}^\top)\boldsymbol{v} = \bar{\lambda}\bar{\boldsymbol{v}}^\top \boldsymbol{v}.
$$

Therefore $\lambda \bar{\boldsymbol{v}}^\top \boldsymbol{v} = \bar{\lambda}\bar{\boldsymbol{v}}^\top \boldsymbol{v}$ which implies $\lambda = \bar{\lambda}$, and so $\lambda$ must be a real number.

**Eigenvectors are orthogonal.** Next, let's see why the eigenvectors of symmetric matrices are orthogonal. Suppose $(\lambda, \boldsymbol{u})$ and $(\mu, \boldsymbol{v})$ are two eigenvalue/eigenvector pairs for a symmetric matrix $\boldsymbol{A}$ such that $\lambda \neq \mu$ (i.e. they are distinct eigenvalues). Then $\boldsymbol{Au} = \lambda \boldsymbol{u}$ and $\boldsymbol{Av} = \mu \boldsymbol{v}$. Then

$$
\begin{align*}
\lambda \boldsymbol{u^\top v} &= (\lambda \boldsymbol{u})^\top \boldsymbol{v} = (\boldsymbol{Au})^\top \boldsymbol{v}\\
&= \boldsymbol{u}^\top \underbrace{\boldsymbol{A}^\top}_{=\boldsymbol{A}} \boldsymbol{v}= \boldsymbol{u}^\top (\boldsymbol{Av})\\ &= \boldsymbol{u}^\top (\mu \boldsymbol{v}) = \mu \boldsymbol{u^\top v}
\end{align*}
$$

Thus, rearranging we get

$$
(\lambda - \mu)\boldsymbol{u^\top v} = 0 \implies \boldsymbol{u^\top v} = 0
$$

since $(\lambda - \mu) \neq 0$, because by assumption the eigenvectors are orthogonal.

Let's see an example in Python, using the `np.linalg.eig(A)` function to find eigenvalues. To find a symmetric matrix $\boldsymbol{A}$, we will use the following approach: first, draw a random $n\times n$ matrix $\boldsymbol{B}$, and then let $\boldsymbol{A} = \boldsymbol{B^\top B}$.

```{code-cell}
import numpy as np

n = 10
B = np.random.normal(size=(n,n))
A = np.dot(B.T, B) # B^T B is always a symmetric matrix

Lambda, U = np.linalg.eig(A)
Lambda = np.diag(Lambda) # numpy returns Lambda as an array, so let's make it a diagonal matrix
```

Now let's verify that $\boldsymbol{A} = \boldsymbol{U\Lambda U^\top}$.

```{code-cell}
ULUT = np.dot(U, np.dot(Lambda, U.T))
np.allclose(A, ULUT)
```

Indeed, the two matrices are the same. Next, let's check that $\boldsymbol{U}$ is in fact orthogonal, by checking that $\boldsymbol{U^\top U} = \boldsymbol{I}$:

```{code-cell}
UTU = np.dot(U.T, U).round(8)
UTU
```

We get the identity back, and so $\boldsymbol{U}$ is in fact orthogonal.

## Projection matrices

Recall that a projection matrix $\boldsymbol{P}$ is a square matrix for which $\boldsymbol{P}^2 = \boldsymbol{P}$. Let's see how the eigenvalues work out for such a matrix. If $\lambda, \boldsymbol{u}$ is an eigenvalue/eigenvector pair for $\boldsymbol{P}$ then we have

$$
\lambda \boldsymbol{u} = \boldsymbol{Pu} = \boldsymbol{P}^2\boldsymbol{u} = \boldsymbol{P}(\lambda \boldsymbol{u}) = \lambda^2 \boldsymbol{u}
$$

Hence $(\lambda - \lambda^2)\boldsymbol{u} = 0$, and since the eigenvector $\boldsymbol{u}$ is not equal to zero, we have $\lambda(1 - \lambda) = 0$, which implies $\lambda$ is 0 zero or 1. Moreover, it turns out that the number of eigenvalues with value 1 is exactly equal to the _dimension_ of the subspace that $\boldsymbol{P}$ projects onto. Let's see why this works in the case of an orthogonal projection. Suppose that $\boldsymbol{P}$ projects onto a subspace $V = \text{Range}(\boldsymbol{P})$ of dimension $k$, and let $q_1,\dots,q_k$ be an orthonormal basis for this subspace. Then define $\boldsymbol{Q} = \begin{bmatrix} q_1 & \dots & q_k\end{bmatrix}$, and note that we have that $\boldsymbol{P} = \boldsymbol{QQ}^\top$, and $\boldsymbol{Q^\top Q} = \boldsymbol{I}_k$. To see the next step, we first need the following fact

> Let $\boldsymbol{A}$ be a symmetric $n\times n$ matrix with real eigenvalues $\lambda_{1},\dots,\lambda_n$. Then $\text{trace}(\boldsymbol{A}) = \sum_{i=1}^n a_{i,i} = \sum_{i=1}^n \lambda_i$. That is, the sum of the diagonal entries of $\boldsymbol{A}$ is equal to the sum of the eigenvalues of $\boldsymbol{A}$.

With this fact in hand, we observe that

$$
\# \{\lambda_i = 1\} = \sum_{i=1}^n \lambda_i = \text{trace}(\boldsymbol{P}) = \text{trace}(\boldsymbol{QQ}^\top) = \text{trace}(\boldsymbol{Q^\top Q}) = \text{trace}(\boldsymbol{I}_k) = k.
$$

Hence any orthogonal projection matrix onto a subspace of dimension $k$ will have exactly $k$ eigenvalues equal to $1$, and $n-k$ eigenvalues equal to $0$. We will see an example of this in Python after introducing the next type of special matrix.

## Triangular matrices

Another import example is triangular matrices. Consider an $n\times n$ (upper) triangular matrix $\boldsymbol{R}$ of the form

$$
\boldsymbol{R} = \begin{bmatrix} r_{1,1} & r_{1,2} & \cdots & r_{1,n}\\ 0 & r_{2,2} & \cdots & r_{2,n}\\ \vdots & \vdots &\ddots & \vdots\\ 0 & 0 & \cdots & r_{n,n}  \end{bmatrix}.
$$

Let's find the eigenvalues of $\boldsymbol{R}$. Recall that one way to characterize the eigenvalues of a matrix is by finding all values of $\lambda$ for which $(\boldsymbol{R} - \lambda \boldsymbol{I})$ is _not_ invertible. We claim that for triangular matrices, this occurs exactly when $\lambda$ is equal to one of the diagonal entries of $\boldsymbol{R}$. Let's see what happens when we set $\lambda = r_{1,1}$. Then we have

$$
\boldsymbol{R} - \lambda \boldsymbol{I} = \boldsymbol{R} - r_{1,1} \boldsymbol{I} = \begin{bmatrix} 0 & r_{1,2} & \cdots & r_{1,n}\\ 0 & r_{2,2}-r_{1,1} & \cdots & r_{2,n}\\ \vdots & \vdots &\ddots & \vdots\\ 0 & 0 & \cdots & r_{n,n} - r_{1,1} \end{bmatrix}
$$

Now clearly since the first column is equal to zero, the columns of $\boldsymbol{R} - \lambda \boldsymbol{I}$ are linearly dependent and hence the matrix is not invertible. Therefore $r_{1,1}$ is an eigenvalue of $\boldsymbol{R}$. Similarly, if we take $\lambda = r_{2,2}$, then we get

$$
\boldsymbol{R} - \lambda \boldsymbol{I} = \boldsymbol{R} - r_{2,2} \boldsymbol{I}= \begin{bmatrix} r_{1,1}-r_{2,2} & r_{1,2} & \cdots & r_{1,n}\\ 0 & 0 & \cdots & r_{2,n}\\ \vdots & \vdots &\ddots & \vdots\\ 0 & 0 & \cdots & r_{n,n}-r_{2,2}  \end{bmatrix}.
$$

Now the columns are again linearly dependent, since $r_{1,1}-r_{2,2}$ will be some scalar multiple of $r_{1,2}$, and so $r_{2,2}$. It's easy to see that this would continue for all $n$ diagonal entries, and therefore that the $n$ eigenvalues of $\boldsymbol{R}$ are exactly equal to its diagonal entries, i.e. $\lambda_{1} = r_{1,1}, \lambda_2 = r_{2,2},\dots, \lambda_n = r_{n,n}$.

Now let's see an example in Python which illustrates both of the previous concepts. To do this, let's first draw a random $n\times k$ matrix $\boldsymbol{A}$, and compute it's QR decomposition $\boldsymbol{A} = \boldsymbol{QR}$.

```{code-cell}
n,k = 10, 5
A = np.random.normal(size=(n,k))
Q, R = np.linalg.qr(A)
```

Now note that the matrix $\boldsymbol{P} = \boldsymbol{QQ^\top}$ will be a projection matrix onto a subspace of dimension $k$ (spanned by the $k$ columns of $\boldsymbol{A}$), and $\boldsymbol{R}$ is upper triangular. Let's first compute the eigenvalue decomposition of the projection $\boldsymbol{QQ^\top}$.

```{code-cell}
P = np.dot(Q, Q.T)
values_P, vectors_P = np.linalg.eig(P)
print('the sum of the the eigenvalues of P is ', round(np.sum(values_P),4))
```

Indeed, it is equal to $k=5$! Next, let's check inspect the diagonal entries of the upper triangular matrix $\boldsymbol{R}$ that we got from the QR decomposition.

```{code-cell}
print('the diagonal entries of R are ', np.diag(R))
values_R, vectors_R = np.linalg.eig(R)
print('the eigenvalues of R are ', values_R)
```

Indeed, the eigenvalues of the upper triangular matrix $\boldsymbol{R}$ are exactly equal to its diagonal entries.
