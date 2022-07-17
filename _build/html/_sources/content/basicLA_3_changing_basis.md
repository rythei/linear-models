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

# Changing Basis

We previously introduced the concept of a _basis_ for a vector space $V$such as $\mathbb{R}^n$.
Recall that a set $\boldsymbol{v}_1,\dots,\boldsymbol{v}_k$ of vectors is called a _basis_ for $V$ if

- $\text{span}(\boldsymbol{v}_1,\dots,\boldsymbol{v}_k) = V$, and
- $\boldsymbol{v}_1,\dots,\boldsymbol{v}_k$ are linearly independent.

If $V = \mathbb{R}^{n}$, then we have $k=n$ vectors in any basis for $\mathbb{R}^{n}$.
In this case, any set of $n$ linearly independent vectors form a basis.

In this section, we will demonstrate the following.

- How vectors are represented with respect to bases, and how a given vector can have different numerical values when represented with respect to different bases.
- How matrices are represented with respect to bases, and how a given matrix can have different numerical values when represented with respect to different bases.
- The tranformation that taking in coordinates $(\alpha_1,\dots, \alpha_n)$ of a vector $\boldsymbol{x}$ with respect to a basis $B_1$, and returning the coordinates $(\beta_1,\dots, \beta_n)$ with respect to a different basis $B_2$ is a linear transformation. We will show you how to find the linear map $T : (\alpha_1,\dots, \alpha_n) \mapsto (\beta_1,\dots,\beta_n)$, which is a matrix called the _change of basis matrix_.
- Linear maps $\boldsymbol{A}$ (or equivalently matrices) are also implicitly represented with respect to a particular basis. We will show you how to represent such linear maps with respect to a different basis.


## Representing vectors with respect to bases

A natural basis to use in many contexts is the standard (or canonical) basis $E = \{\boldsymbol{e}_1,\dots, \boldsymbol{e}_n \}$, where

$$
\boldsymbol{e}_i = \begin{bmatrix}0\\ \vdots \\ 1 \\ \vdots \\0\end{bmatrix}  .
$$

That is, $\boldsymbol{e}_i$ is $1$ in the $i$th component, and $0$ elsewhere.

The reason that this is such a convenient basis is the following.
When we write a vector


$$
\boldsymbol{x} = \begin{bmatrix} x_1 \\ \vdots \\ \\ x_n \end{bmatrix} \in \mathbb{R}^n,
$$

we are implicitly writing it with respect to the standard basis.
The reason for this is that
$$
\boldsymbol{x} = x_1 \boldsymbol{e}_1 + x_2\boldsymbol{e}_2 + \dots + x_n\boldsymbol{e}_n  .
$$

_Said another way, a vector is an abstract thing that obeys the rules of scalar multiplication and vector addition, but when we represent a vector as an array or list of numbers, we are representing that vector with respect to the standard basis._

On the other hand, if $B = \{\boldsymbol{v}_1,\dots, \boldsymbol{v}_n\}$ is a different basis for $\mathbb{R}^n$, we could equivalently represent the vector $\boldsymbol{x}$ with respect to this basis as $\boldsymbol{x} = \alpha_1 \boldsymbol{v}_1 + \alpha_2\boldsymbol{v}_2 + \dots + \alpha_n\boldsymbol{v}_n$ for some scalars $\alpha_1,\dots,\alpha_n$. Here $(\alpha_1,\dots,\alpha_n)$ are called the _coordinates of $\boldsymbol{x}$ with respect to $B$_.

## The change of basis matrix

### Changing to and from the standard basis

Let's start by assuming we have a vector $\boldsymbol{x} = (x_1,\dots, x_n)$ represented with respect to the standard basis. We want to find the coordinates $\boldsymbol{\alpha} = (\alpha_1,\dots,\alpha_n)$ of this vector with respect to a new basis $B = \{\boldsymbol{v}_1,\dots,\boldsymbol{v}_n\}$. That is, we want to find numbers $\alpha_1,\dots,\alpha_n$ satisfying the following:


$$
\boldsymbol{x} = \alpha_1 \boldsymbol{v}_1 + \alpha_2\boldsymbol{v}_2 + \dots + \alpha_n\boldsymbol{v}_n
$$


Let's define the following matrix:


$$
\boldsymbol{V} = \begin{bmatrix}\boldsymbol{v}_1 & \boldsymbol{v}_2 & \cdots & \boldsymbol{v}_n\end{bmatrix}
$$


i.e. the $n\times n$ matrix whose columns are the vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_n$. Then if we recall the definition of matrix-vector multiplication, we realize that we can conveniently write


$$
\boldsymbol{x} = \alpha_1 \boldsymbol{v}_1 + \alpha_2\boldsymbol{v}_2 + \dots + \alpha_n\boldsymbol{v}_n = \boldsymbol{V\alpha}
$$


The equation $\boldsymbol{x} = \boldsymbol{V\alpha}$ is important: in fact, it immediately gives us the equation to change from the basis $B$ to the standard basis $E$: given a representation $\boldsymbol{\alpha}$ of $\boldsymbol{x}$ with respect to $B$, we can simply apply the linear transformation $\boldsymbol{V}$ to $\boldsymbol{\alpha}$ and get back the coordinates of $\boldsymbol{x}$ with respect to the standard basis.

But we would like to go the other way: change from the standard basis $E$ to the new basis $B$. To do this, all we need to do is _invert_ the transformation $\boldsymbol{V}$. Indeed, it is easy to show that $\boldsymbol{V}$ is invertible using what we've seen in previous sections, since it is square and has linearly independent columns. Therefore we have


$$
\boldsymbol{x} = \boldsymbol{V\alpha} \iff \boldsymbol{\alpha} = \boldsymbol{V}^{-1}\boldsymbol{x}
$$


Therefore, we've seen that we can take a vector $\boldsymbol{x}$ represented with respect to the standard basis $E$ and obtain the coordinates of $\boldsymbol{x}$ with respect to a different basis $B$ by applying the linear transformation $\boldsymbol{V}^{-1}$. For the remainder of this section, we denote $\boldsymbol{T}_{E\to B} = \boldsymbol{V}^{-1}$ as the _change of basis matrix_ from $E$ to $B$. Notice that we also have $\boldsymbol{T}_{B\to E} = \boldsymbol{V}$, and that we always have the relation


$$
\boldsymbol{T}_{E\to B} = \boldsymbol{T}_{B\to E}^{-1}
$$


Let's see a few examples of changing bases in Python. Let's start out with a vector $\boldsymbol{x} = (1,2,3,4)$ represented with respect to the standard basis

```{code-cell}
import numpy as np

x = np.array([1,2,3,4])
x
```

Suppose we want to represent this with respect to the following basis:


$$
\boldsymbol{v}_1 = \begin{bmatrix}1\\ 0\\0\\0\end{bmatrix},\;\;\; \boldsymbol{v}_2 = \begin{bmatrix}1\\ 1\\0\\0\end{bmatrix},\;\;\; \boldsymbol{v}_3 = \begin{bmatrix}1\\ 1\\1\\0\end{bmatrix},\;\;\; \boldsymbol{v}_4 = \begin{bmatrix}1\\ 1\\1\\1\end{bmatrix}
$$

```{code-cell}
v1 = np.array([1,0,0,0])
v2 = np.array([1,1,0,0])
v3 = np.array([1,1,1,0])
v4 = np.array([1,1,1,1])
```

As we saw above, we need to form the matrix $\boldsymbol{V}$ whose columns are $\boldsymbol{v}_1,\boldsymbol{v}_2,\boldsymbol{v}_3,\boldsymbol{v}_4$. We can do this in numpy with the following.

```{code-cell}
V = np.stack([v1,v2,v3,v4], axis=1)
V
```

Now we want to find the change of basis matrix $\boldsymbol{T}_{E\to B} = \boldsymbol{V}^{-1}$. To do this, we need to _invert_ the matrix $\boldsymbol{V}$. We can do this in numpy with the function `np.linalg.inv()`.

```{code-cell}
T_EtoB = np.linalg.inv(V)
T_EtoB
```

 To compute the coordinates $\boldsymbol{\alpha}$ of $\boldsymbol{x}$ with respect to $\boldsymbol{v}_1,\dots,\boldsymbol{v}_4$, we just need to apply this tranformation to $\boldsymbol{x}$:

```{code-cell}
alpha = np.dot(T_EtoB, x)
alpha
```

Now let's check that this actually gave us the correct answer by computing $\alpha_1\boldsymbol{v}_1 + \alpha_2\boldsymbol{v}_2 + \alpha_3\boldsymbol{v}_3 + \alpha_4\boldsymbol{v}_4$:

```{code-cell}
alpha[0]*v1 + alpha[1]*v2 + alpha[2]*v3 + alpha[3]*v4
```

Indeed, we obtain the original vector $\boldsymbol{x}$ back, and so $\boldsymbol{\alpha}$ are the correct coordinates for $\boldsymbol{x}$ with respect to $\boldsymbol{v}_1,\dots,\boldsymbol{v}_4$.

We can also get the vector $\boldsymbol{x}$ back by applying the map $\boldsymbol{T}_{B\to E} = \boldsymbol{V}$:

```{code-cell}
T_BtoE = V
np.dot(T_BtoE, alpha)
```

As expected, this also gives us the original vector $\boldsymbol{x}$ back.

### Changing to and from two arbitrary bases

In the previous section, we saw how to derive the change of basis matrix between the standard basis and another non-standard matrix. Here, we see how to use this concept to find the change of basis matrix between two arbitrary bases. Suppose we have two bases $B_1 =\{ \boldsymbol{v}_1,\dots,\boldsymbol{v}_n\}$ and $B_2 = \{\boldsymbol{u}_1,\dots,\boldsymbol{u}_n\}$ for $\mathbb{R}^n$. Like before, let's define $\boldsymbol{V} = \begin{bmatrix} \boldsymbol{v}_1,\dots,\boldsymbol{v}_n\end{bmatrix}$,  $\boldsymbol{U} = \begin{bmatrix} \boldsymbol{u}_1,\dots,\boldsymbol{u}_n\end{bmatrix}$ be the matrices whose columns are the basis vectors of $B_1$ and $B_2$, respectively.

In the previous section, we saw that the change of basis matrix for changing from the standard basis $E$ to $B_1$ or $B_2$, and back again. In particular, we have that


$$
\boldsymbol{T}_{E\to B_1} = \boldsymbol{V}^{-1},\hspace{5mm} \boldsymbol{T}_{B_1\to E} = \boldsymbol{V},\hspace{5mm} \boldsymbol{T}_{E\to B_2} = \boldsymbol{U}^{-1},\hspace{5mm} \boldsymbol{T}_{B_2\to E} = \boldsymbol{U}
$$


Now we can derive the change of basis matrix from $B_1$ to $B_2$ and vice versa by _composing_ these changing of basis matrices. Our strategy is as follows: say we start with coordinates $\boldsymbol{\alpha} = (\alpha_1,\dots,\alpha_n)$ in terms of $B_1$. We first apply $\boldsymbol{T}_{B_1\to E}$, to obtain the coordinates in terms of the standard basis, and then apply $\boldsymbol{T}_{E\to B_2}$ to obtain coordinates $\boldsymbol{\beta} = (\beta_1,\dots,\beta_n)$ in terms of the basis $B_2$. This gives us


$$
\boldsymbol{\beta} = \boldsymbol{T}_{E\to B_2}\boldsymbol{T}_{B_1\to E}\boldsymbol{\alpha} = \boldsymbol{U}^{-1}\boldsymbol{V\alpha}
$$


Hence we see that the change of basis matrix from $B_1\to B_2$ is simply


$$
\boldsymbol{T}_{B_1 \to B_2} = \boldsymbol{U}^{-1}\boldsymbol{V}.
$$


Similarly, the change of basis matrix to go back, i.e. from $B_2\to B_1$ is simply


$$
\boldsymbol{T}_{B_2\to B_1} = \boldsymbol{T}_{E\to B_1}\boldsymbol{T}_{B_2 \to E} = \boldsymbol{V}^{-1}\boldsymbol{U}.
$$


And that's all there is to it! By _composing_ linear maps, we can obtain the linear map changing between any two arbitrary bases.

Let's see some examples in Python.

We'll use the same basis $B_1 =\{ \boldsymbol{v}_1,\boldsymbol{v}_2,\boldsymbol{v}_3,\boldsymbol{v}_4\}$ defined above, as well as the basis $B_2 = \{\boldsymbol{u}_1, \boldsymbol{u}_2,\boldsymbol{u}_3,\boldsymbol{u}_4\}$ defined below:


$$
\boldsymbol{u}_1 = \begin{bmatrix}-1\\ 0\\0\\0\end{bmatrix},\;\;\; \boldsymbol{u}_2 = \begin{bmatrix}1\\ -1\\0\\0\end{bmatrix},\;\;\; \boldsymbol{u}_3 = \begin{bmatrix}1\\ 1\\-1\\0\end{bmatrix},\;\;\; \boldsymbol{u}_4 = \begin{bmatrix}-1\\ 1\\1\\-1\end{bmatrix}
$$


We define these as numpy arrays:

```{code-cell}
u1 = np.array([-1, 0, 0, 0])
u2 = np.array([1, -1, 0, 0])
u3 = np.array([1, 1, -1, 0])
u4 = np.array([-1, 1, 1, -1])
```

Let's store these in a matrix $U$.

```{code-cell}
U = np.stack([u1,u2,u3,u4], axis=1)
U
```

Now we can compute the tranformations $\boldsymbol{T}_{B_1 \to B_2}$ and $\boldsymbol{T}_{B_2 \to B_1}$ using the formulas above.

```{code-cell}
T_B1toB2 = np.dot(np.linalg.inv(U), V)
T_B2toB1 = np.dot(np.linalg.inv(V), U)
```

Let's see what the coordinates of $\boldsymbol{\alpha}$ are with respect to the basis $B_2$:

```{code-cell}
beta = np.dot(T_B1toB2, alpha)
beta
```

We should be able to confirm that $\boldsymbol{x} = \beta_1 \boldsymbol{u}_1 + \beta_2 \boldsymbol{u}_2 + \beta_3 \boldsymbol{u}_3 + \beta_4 \boldsymbol{u}_4$:

```{code-cell}
beta[0]*u1 + beta[1]*u2 + beta[2]*u3 + beta[3]*u4
```

As expected, this gives us our vector $\boldsymbol{x}$, now represented with respect to the basis $B_2$. We should also we able to confirm that we get $\boldsymbol{\alpha}$ back when we apply the transformation $\boldsymbol{T}_{B_2 \to B_1}$ to $\boldsymbol{\beta}$:

```{code-cell}
np.dot(T_B2toB1, beta)
```

Indeed, we get back the coordinates $\boldsymbol{\alpha}$ with respect to $B_1$.

## Representing a matrix with respect to a basis

Suppose we have a linear function $f(\boldsymbol{x})$, which is represented as $f(\boldsymbol{x}) = \boldsymbol{Ax}$ for a vector $\boldsymbol{x}$ represented with respect to the standard basis. Then $(i,j)^{th}$ entry of $\boldsymbol{A}$ is the $i$th coordinate of the vector $f(\boldsymbol{e}_j)$, where $\boldsymbol{e}_j$ is the $j^{th}$ standard basis vector. To see this, notice that


$$
f(\boldsymbol{e}_j) = \boldsymbol{Ae}_j = \begin{bmatrix}a_{11} & \cdots & a_{1n}\\ \vdots &\ddots &\vdots\\ a_{n1} & \cdots & a_{nn}\end{bmatrix}\begin{bmatrix}0\\ \vdots \\ 1 \\ \vdots \\0\end{bmatrix} = \begin{bmatrix}a_{1j}\\ a_{2j}\\\vdots \\ a_{nj}\end{bmatrix}
$$
Thus we see that the $i$th entry of the vector $\boldsymbol{Ae}_j$ is $a_{ij}$. Hence if we want to represent the vector $f(\boldsymbol{e}_j)$ with respect to the standard basis, we can do so with the representation


$$
f(\boldsymbol{e}_j) = a_{1j}\boldsymbol{e}_1 + a_{2j}\boldsymbol{e}_2 + \cdots + a_{nj}\boldsymbol{e}_j
$$
More generally, this also works with other bases. For example, consider a basis $B = \{\boldsymbol{v}_1,\dots, \boldsymbol{v}_n\}$. If $\boldsymbol{A}_B$ is a matrix representing $f$ with respect to this basis, then the $(i,j)^{th}$ entry of $\boldsymbol{A}_B$ is the $i^{th}$ coordinate of the vector $f(\boldsymbol{v}_j)$ with respect to $B$. In what follows, we show how we can find the matrix $\boldsymbol{A}_B$ given the linear transformation $f(\boldsymbol{x}) = \boldsymbol{Ax}$ represented with respect to the standard basis.

### Changing basis for a matrix

Suppose we have a vector $\boldsymbol{x}\in \mathbb{R}^n$, represented with respect to the standard basis $E$ as $\boldsymbol{x} = (x_1,\dots,x_n)$. We know that we can transform such a vector with a matrix $\boldsymbol{A} \in \mathbb{R}^{m\times n}$, representing a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$. Now suppose we chose to change basis, and represent $\boldsymbol{x}$ with respect to a new basis $B = \{\boldsymbol{v}_1,\dots, \boldsymbol{v}_n\}$ using the change of basis matrix $\boldsymbol{T}_{E\to B}$ to get the new coordinates $\boldsymbol{\alpha} = \boldsymbol{T}_{E\to B}\boldsymbol{x}$. If we directly apply the matrix $A$ to the new coordinates $\boldsymbol{\alpha}$, we won't in general get the same result as if we applied it to the original vector $\boldsymbol{x}$. This is because when we represent a linear map as a matrix, we are also implicitly doing so with a fixed basis. However, we will see in this section that we can also represent a linear map (i.e. a matrix) with respect to different bases.

To see how this works, we will use a similar approach as in the previous section. Since $\boldsymbol{A}$ is represented with respect to the standard basis, we need to pass it vectors represented with respect to this same basis. Therefore, a natural approach when trying to apply the linear map $\boldsymbol{A}$ to vectors $\boldsymbol{\alpha}$ represented with respect to $B$ is to first transform $\boldsymbol{\alpha}$ to be in terms of the standard basis, using $\boldsymbol{T}_{B\to E}$, then apply the linear map $\boldsymbol{A}$, and then finally tranform back to the basis $B$. In symbols, we apply the following linear map:


$$
\boldsymbol{A}_B = \boldsymbol{T}_{E\to B}\boldsymbol{A}\boldsymbol{T}_{B\to E}
$$


Here we use the notation $\boldsymbol{A}_B$ to denote the linear map $A$ represented with respect to the basis $B$. If we let $\boldsymbol{V} = \begin{bmatrix} \boldsymbol{v}_1 &\cdots & \boldsymbol{v}_n\end{bmatrix}$ be the matrix whose columns are the basis vectors of $B$, we can alternatively write the matrix $\boldsymbol{A}_B$ as


$$
\boldsymbol{A}_B = \boldsymbol{V}^{-1}\boldsymbol{AV}
$$


We will see formulas of this kind frequently later in the course when we discuss eigenvalue and singular value decompositions, and so it is important to understand that these decompositions are really just representing a particular linear transformation with respect to a new basis.

Let's see some example of how this works in Python, using the same basis $\boldsymbol{v}_1,\boldsymbol{v}_2,\boldsymbol{v}_3,\boldsymbol{v}_4$ defined above. Let's consider the linear map represented with respect to the standard basis as


$$
\boldsymbol{A} = \begin{bmatrix}1 & 1 & 1 & -1\\ 1 & 1 & -1 & 1\\ 1& -1 & 1 &1\\ -1 & 1 & 1& 1\end{bmatrix}
$$


In numpy, we can define this with the following code.

```{code-cell}
A = np.array([[1, 1, 1, -1], [1,1,-1,1], [1, -1, 1, 1], [-1,1,1,1]])
A
```

Now let's see what answer we get when we apply this matrix to the vector $\boldsymbol{x} = (1,2,3,4)$.

```{code-cell}
Ax = np.dot(A,x)
Ax
```

This gives us a new vector $\boldsymbol{Ax}$ represented with repsect to the standard basis $E$.

Now suppose that instead we choose to work with the basis $B$, and use the representation of $\boldsymbol{x}$ in terms of the coordinates $\boldsymbol{\alpha}$ in this basis. As we saw above, we can compute the matrix $\boldsymbol{A}$ with respect to the basis $B$ as $\boldsymbol{A}_B = \boldsymbol{T}_{E\to B}\boldsymbol{A}\boldsymbol{T}_{B\to E}$. We do this in Python below.

```{code-cell}
A_B = np.dot(T_EtoB, np.dot(A, T_BtoE))
A_B
```

Now let's apply this to the coordinates $\boldsymbol{\alpha}$ to get $\boldsymbol{A}_B\boldsymbol{\alpha}$.

```{code-cell}
A_Balpha = np.dot(A_B, alpha)
A_Balpha
```

 If we did everything correctly, we should see that this gives us the coordinates of the vector $\boldsymbol{Ax}$ with respect to the basis $B$. Let's check that this is true by computing

$$
 [\boldsymbol{A}_B\boldsymbol{\alpha}]_1\boldsymbol{v}_1 + [\boldsymbol{A}_B\boldsymbol{\alpha}]_2\boldsymbol{v}_2 +[\boldsymbol{A}_B\boldsymbol{\alpha}]_3\boldsymbol{v}_3 +[\boldsymbol{A}_B\boldsymbol{\alpha}]_4\boldsymbol{v}_4.
 $$

```{code-cell}
A_Balpha[0]*v1 + A_Balpha[1]*v2 + A_Balpha[2]*v3 + A_Balpha[3]*v4
```

As expected, this indeed gives us the same thing as $\boldsymbol{Ax}$!
