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

# Matrices and matrix operations

In this section, we introduce the concept of matrices, and the associated operation of multiplication of two matrices.

## Matrices in Python
One way to define a matrix is as follows: an $m\times n$ matrix $\boldsymbol{A} \in \mathbb{R}^{m\times n}$, is an array of real numbers consisting of $m$ rows and $n$ columns. In Python, we can define such an array using `numpy`:

```{code-cell}
import numpy as np

m,n = 5,3

A = np.random.normal(size=(m,n))
print(A)
```

Note that by this definition of a matrix, a vector is simply a special case of a matrix with either just one column or one row. By convention, we usually think of a vector $\boldsymbol{x}\in \mathbb{R}^n$ as being a _column vector_, with $n$ rows and $1$ column, so that $\boldsymbol{x}$ is really a $n\times 1$ matrix.

In `numpy`, we can specify a vector as being a column vector by suitably reshaping it.

```{code-cell}
x = np.random.normal(size=n)
print(x.shape) # defaults to shape (n,)
x = x.reshape(n,1)
print(x.shape) # explicitly making x a column vector
```

<!-- We remark that at times, we will want to refer to the individual rows or columns of a matrix. To do this we use the notation $\boldsymbol{A}[i,:]$ to refer to the $i^{th}$ row of $\boldsymbol{A}$, and $\boldsymbol{A}[:,j]$ to refer to the $j^{th}$ column of $\boldsymbol{A}$ (this has the added convenience of also being the notation to select rows/column of an array in `numpy`).  -->

Note that by default, `numpy` stores 1-d arrays as having shape `(n,)`, which is, somewhat subtly, different from a column vector, which has shape `(n,1)`. So to work with a column vector in Python, we have to explictly specify its shape. For many operations we will want to perform, this distinction won't matter much, though for some operations this distinction is in fact important, and so we will want to be careful. We will see examples of this in the coming sections. We can also represent a vector explicitly as a row vector in a similar way.

```{code-cell}
x = x.reshape(1,n)
print(x.shape) # explicitly making x a row vector
```

## The transpose operation

Suppose we were given an $m\times n$ matrix $\boldsymbol{A}$ of the form

$$
\boldsymbol{A} = \begin{bmatrix}a_{11}& \cdots &a_{1n}\\ a_{21}&\cdots & a_{2n}\\ \vdots & \ddots & \vdots \\ a_{m1}&\cdots & a_{mn}\end{bmatrix} \in \mathbb{R}^{m\times n}.
$$

One of the most important operations we can perform on such a matrix is to take its _transpose_, which means to form the $n\times m$ matrix $\boldsymbol{A}^\top$ by defining the $i^{th}$ row of $\boldsymbol{A}^\top$ be the $i^{th}$ column of $\boldsymbol{A}$. Specifically, this would give us

$$
\boldsymbol{A}^\top = \begin{bmatrix}a_{11}& \cdots &a_{m1}\\ a_{12}&\cdots & a_{m2}\\ \vdots & \ddots & \vdots \\ a_{1n}&\cdots & a_{mn}\end{bmatrix} \in \mathbb{R}^{n\times m}.
$$

Note that this operation takes a matrix of shape $m\times n$ and returns a matrix of shape $n\times m$. It is easy to find the transpose of a matrix (i.e. `numpy` array) in Python:

```{code-cell}
print(A.shape)
AT = A.T # take the transpose of A
print(AT.shape)
```

We can also use this to convert between row and column vectors in `numpy`.

```{code-cell}
x = np.random.normal(size=n)
x = x.reshape(n,1)
print(x.shape) #column vector
xT = x.T
print(xT.shape) #row vector
```

## Matrix multiplcation

The second operation on matrices which will we frequently encounter is matrix multiplication. To best introduce matrix multiplication, however, we first need to introduce a somewhat simpler operation on vectors called the _dot product_ or _inner product_. Given two vectors $\boldsymbol{x},\boldsymbol{y} \in \mathbb{R}^n$, their dot product is

$$
\langle \boldsymbol{x},\boldsymbol{y}\rangle = \boldsymbol{x}^\top \boldsymbol{y} = \sum_{i=1}^n x_iy_i.
$$

Here by using the notation $\boldsymbol{x}^\top \boldsymbol{y}$, we implicitly assumed that $\boldsymbol{x}, \boldsymbol{y}$ were both _column vectors_, to that the operation $\boldsymbol{x}^\top \boldsymbol{y}$ involved multiplying a row vector, $\boldsymbol{x}^\top$, with a column vector, $\boldsymbol{y}$. This returns a single real _number_, which is the sum $\sum_{i=1}^n x_iy_i$, i.e. multiplying and summing up the entries of the two vectors pairwise.

Later, we will discuss in more detail the geometric meaning of this operation. For now, however, it will suffice to simply take this as a definition. In `numpy`, we can compute the dot product of two vectors using the function `np.dot`. For example,

```{code-cell}
x = np.random.normal(size=n)
y = np.random.normal(size=n)
x_dot_y = np.dot(x,y)
print(x_dot_y)
```

Note that here we didn't explicitly define that either $\boldsymbol{x}$ or $\boldsymbol{y}$ was a row or column vector; fortunately, `numpy` automatically handles the shaping of arrays for us in this problem.

Now that we have defined the dot product, we can define the more general operation of matrix multiplication. Given matrices $\boldsymbol{A}\in \mathbb{R}^{m\times n}$, with rows $\boldsymbol{a}_{1:},\dots,\boldsymbol{a}_{m:}$, and  $\boldsymbol{B}\in \mathbb{R}^{n\times p}$, with columns $\boldsymbol{b}_{:1},\dots, \boldsymbol{b}_{:p}$, we define the matrix product $\boldsymbol{AB}$ to be the $m\times p$ matrix whose $(i,j)^{th}$ entry is

$$
[\boldsymbol{A}\boldsymbol{B}]_{ij} = \boldsymbol{a}_{i:}^\top \boldsymbol{b}_{:j}.
$$

That is, the $(i,j)^{th}$ entry of the matrix $\boldsymbol{AB}$ is the dot product of the $i^{th}$ row of $\boldsymbol{A}$ with the $j^{th}$ column of $\boldsymbol{B}$.

Note that for this operation to be well-defined, we need that the rows of $\boldsymbol{A}$ are of the same dimension as the columns of $\boldsymbol{B}$, or equivalently that the number of columns of $\boldsymbol{A}$ is equal to the number of rows of $\boldsymbol{B}$. Let's see some examples in Python. Note that we can also use the `numpy` function `np.dot` to perform matrix multiplication.

```{code-cell}
m, n, p = 10,5,3

A = np.random.normal(size=(m,n))
B = np.random.normal(size=(n,p))
AB = np.dot(A,B)
print(AB.shape)
```

This is an example where the matrix product is well-defined, since the number of columns of $\boldsymbol{A}$ (5) is equal to the number of rows of $\boldsymbol{B}$ (also 5). Let's see an example where this doesn't work.

```{code-cell}
# now the inner dimensions don't match
m, n, k, p = 10,5,4, 3

A = np.random.normal(size=(m,n))
B = np.random.normal(size=(k,p))
AB = np.dot(A,B)
print(AB.shape)
```

As we'd expect, `numpy` gives us an error, because the two matrices are not of coherent dimensions to perform matrix multiplcation.

## An aside: vectorizing operations

One important point from a computational perspective is that matrix multiplication can often be a slow task for large matrices. Because of this, the functions like `np.dot` in `numpy` are actually written in a faster language, C in this case, and wrapped in Python functions that are easy to use. However, there is some overhead with converting the results from C back to Python. Therefore, in practice, we want to minimize the number of calls we need to make to function `np.dot`.

Let's see an example of this by computing a matrix product "by hand", i.e. by computing each of the dot products comprising the product $\boldsymbol{A},\boldsymbol{B}$ in Python. We do this using the following function.

```{code-cell}
def slow_mat_mul(A,B):
  assert A.shape[1] == B.shape[0], 'invalid dimensions for matrix multiplication'

  AB = np.empty(shape=(A.shape[0],B.shape[1]))

  for i in range(A.shape[0]):
    for j in range(B.shape[1]):
      AB[i,j] = np.dot(A[i,:], B[:,j])

  return AB
```

Let's compare how this function performs in terms of speed with calling the `np.dot` function just a single time to perform the matrix multiply.

```{code-cell}
import time

m, n, p = 1000, 500, 500

A = np.random.normal(size=(m,n))
B = np.random.normal(size=(n,p))

tic = time.time()
slow_mat_mul(A,B)
print(f'time using our function: {time.time()-tic} seconds')

tic = time.time()
np.dot(A,B)
print(f'time using np.dot: {time.time()-tic} seconds')
```

As you can see, calling `np.dot` just once is many times faster than manually computing each of the entries, even though the same things need to be computed using either approach. This is because the "fast" method calls the underlying C function just one, instead of $m\cdot p$ times, as the "slow" method does. The practice of optimizing performance by minimizing the number of function calls one needs to make is often called _vectorizing_, and is often critically important when working with large problems.
