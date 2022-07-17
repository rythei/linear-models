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

```{code-cell} ipython3
import numpy as np
```

# Left Inverses, Right Inverses, and Inverses

As we've seen in the previous section, matrices really just represent _linear functions_ between vector spaces.
In particular, a $m\times n$ matrix $\boldsymbol{A}$ is a linear function mapping vectors $\boldsymbol{x} \in \mathbb{R}^n$ to vectors $\boldsymbol{y} = \boldsymbol{Ax} \in \mathbb{R}^m$.

From our discussion on functions earlier in the semester, we know that functions may have left inverses, right inverses, or both, depending on whether the function is injective or surjective or both.
This is in particular true for linear functions $f: \mathbb{R}^n \to \mathbb{R}^m$, which are all of the form $f(\boldsymbol{x}) = \boldsymbol{Ax}$ for some $m\times n$ matrix $\boldsymbol{A}$.
In this section, we discuss if and when such a function $f$ is injective or surjective or both, in terms of properties of the matrix $\boldsymbol{A}$.


## Left inverses for matrices


### When is a linear function injective

As we saw earlier in the last section, a function $f$ has (at least one) left inverse as long as it is _injective_.
Recall that a function $f$ is injective if $f(\boldsymbol{x}) = f(\boldsymbol{y})$ implies that $\boldsymbol{x}=\boldsymbol{y}$.

Now suppose we have a linear function $f:\mathbb{R}^n \to \mathbb{R}^m$ given by $f(\boldsymbol{x}) = \boldsymbol{Ax}$.
Then, by linearity of $f$,

$$
f(\boldsymbol{x}) = f(\boldsymbol{y}) \iff f(\boldsymbol{x}) - f(\boldsymbol{y}) = \boldsymbol{0} \iff f(\boldsymbol{x} - \boldsymbol{y}) = \boldsymbol{0} \iff \boldsymbol{A}(\boldsymbol{x}-\boldsymbol{y}) = \boldsymbol{0}  .
$$

Therefore, supposing that $f(\boldsymbol{x}) = f(\boldsymbol{y})$ is the same as supposing that $\boldsymbol{x} - \boldsymbol{y}$ is a vector such that $\boldsymbol{A}(\boldsymbol{x}-\boldsymbol{y}) = \boldsymbol{0}$.
What injectivity is saying is that this is only possible if $\boldsymbol{x} - \boldsymbol{y} = \boldsymbol{0}.$
In other words, $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is injective if and only if the only vector that $\boldsymbol{A}$ maps to zero is the zero vector.

Before continuing, it's important to come up with a convenient representation for $\boldsymbol{Ax}$.
For this section, we will denote the $i^{th}$ column of $\boldsymbol{A}$ by $\boldsymbol{A}[:,i]$, which is a vector in $\mathbb{R}^m$.
We can express the vector $\boldsymbol{Ax}$ as

$$
\boldsymbol{Ax} = \begin{bmatrix} \boldsymbol{A}[:,1] & \cdots & \boldsymbol{A}[:,n] \end{bmatrix}\begin{bmatrix}x_1\\\vdots \\ x_n\end{bmatrix} = x_1\boldsymbol{A}[:,1] + x_2\boldsymbol{A}[:,2] + \cdots + x_n\boldsymbol{A}[:,n].
$$

Therefore we see that the vector $\boldsymbol{Ax}$ is really a linear combination of the columns of $\boldsymbol{A}$.

Now let's return to our original problem of determining whether or not $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is injective.
Suppose that $\boldsymbol{Ax} = \boldsymbol{0}$.
Then, from the above, we have that

$$
x_1\boldsymbol{A}[:,1] + x_2\boldsymbol{A}[:,2] + \cdots + x_n\boldsymbol{A}[:,n] = \boldsymbol{0}   .
$$

That is, our function $f(\boldsymbol{x})$ is injective if and only if the above identity is possible only if $\boldsymbol{x} = \boldsymbol{0}$, or in other words if $x_1 = x_2 = \dots = x_n = 0$.
From our discussion on linear combinations and linear dependence in the previous chapter, we know that this is true if and only if the vectors $\boldsymbol{A}[:,1],\dots,\boldsymbol{A}[:,n]$ are linearly independent.
Therefore, we have the following statement, which ties together our concepts from injective functions for general functions, and our linear algebraic concepts of vectors and linear dependence:

> _For an $m\times n$ matrix $\boldsymbol{A}$, the linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is injective if and only if the columns of $\boldsymbol{A}$ are linearly independent_.

A simple corollary of this fact is the following: if $n > m$, then $f(\boldsymbol{x}) = \boldsymbol{Ax}$ can _never_ be injective.
This is because the maximum number of linearly independent vectors in $\mathbb{R}^m$ is $m$, and so no $n$ vectors can be linearly independent.
Thus, at the very least, for $f(\boldsymbol{x}) = \boldsymbol{Ax}$ to be injective, we need that $n\leq m$.
Notice that this is coherent with our understanding of injective functions from the previous section: there we said that, intuitively, $f: X\to Y$ can only be injective if $X$ is "smaller than" $Y$. Here, this translates to the fact that if $n\leq m$, then $\mathbb{R}^n$ is "smaller than" $\mathbb{R}^m$.


### Left inverses for injective linear functions

We know that a function every injective function $f: X \to Y$ has a least one left inverse $g:Y \to X$ such that $g\circ f = \text{id}_X$.
For linear functions of the form $f(\boldsymbol{x}) = \boldsymbol{Ax}$, a left inverse is another linear function $g(\boldsymbol{y}) = \boldsymbol{By}$ where $\boldsymbol{B}$ is an $n\times m$ matrix such that

$$
\boldsymbol{x} = (g\circ f)(\boldsymbol{x}) = g(f(\boldsymbol{x})) = g(\boldsymbol{Ax}) = \boldsymbol{B}\boldsymbol{Ax}   .
$$

In other words, $g(\boldsymbol{y}) = \boldsymbol{By}$ is a left inverse if and only if $\boldsymbol{BA} = \boldsymbol{I}$ is the identity matrix on $\mathbb{R}^n$.

The condition $\boldsymbol{BA} = \boldsymbol{I}$ constitutes a linear system of equations, with $n^2$ constraints and $n\cdot m$ unknown variables (the entries of the matrix $\boldsymbol{B}$).
Since we know that for $f$ to be injective we need $n\leq m$, we have that $n^2 \leq n\cdot m$, and so in fact $\boldsymbol{BA} = \boldsymbol{I}$ is a linear system with more unknowns than constraints -- this is also commonly known as an _underdetermined system_.
Typically, undetermined solutions have many possible solutions, and hence in general an injective function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ will have many left inverses.

In what follows, we walk through a simple example in Python of finding a left inverse of a matrix.
Consider the function $f: \mathbb{R}^2 \to \mathbb{R}^3$ where $f(\boldsymbol{x}) = \boldsymbol{Ax}$ with

$$
\boldsymbol{A} = \begin{bmatrix}1 & 2 \\ 0 & 0 \\ 0 &3\end{bmatrix}  .
$$

It's easy to see that the vectors $(1,0,0)$ and $(2,0,3)$ are linearly independent, and so we know from the previous subsection that $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is indeed injective.

Let's define $\boldsymbol{A}$ as a numpy array.

```{code-cell}
A = np.array([[1,2], [0, 0], [0,3]])
A
```

A left inverse $\boldsymbol{B}$ for $\boldsymbol{A}$ will be a $2\times 3$ matrix of the form

$$
\boldsymbol{B} = \begin{bmatrix}b_{11} & b_{12} & b_{13}\\ b_{21} & b_{22} & b_{23}\end{bmatrix}  .
$$

The constraint $\boldsymbol{BA} = \boldsymbol{I}$ becomes

$$
\boldsymbol{BA} = \begin{bmatrix}b_{11} & b_{12} & b_{13}\\ b_{21} & b_{22} & b_{23}\end{bmatrix} \begin{bmatrix}1 & 2 \\ 0 & 0 \\ 0 &3\end{bmatrix} =  \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}  .
$$

If we carry out the above matrix multiplication, we are left with the following $4$ constraints:

$$
\begin{cases}
b_{11} = 1 & (1)\\
2b_{11} + 3 b_{13} = 0 & (2)\\
b_{21} = 0 & (3)\\
2b_{21} + 3b_{23} = 1& (4)   .
\end{cases}
$$

We've immediately determined that $b_{11} = 1$ and $b_{21} = 0$, so let's define these:

```{code-cell}
b11 = 1
b21 = 0
```

It remains to determine the remaining entries of the matrix $\boldsymbol{B}$. From equation (2), we have that

$$
0= 2b_{11} + 3b_{13} = 2 + 3b_{13} \implies b_{13} = -\frac{2}{3}   .
$$

Similarly, from (4) we have

$$
1 = 2b_{21} + 3b_{23} = 3b_{23} \implies b_{23} = \frac{1}{3}   .
$$

Let's define these in Python as well.

```{code-cell}
b13 = -2./3
b23 = 1./3
```

But what about $b_{12}$ and $b_{22}$?
These two variables don't appear in our constraints at all; indeed, this is precisely because these variables can be _anything_.
Let's see that this is indeed true.
First, we'll define a function `left_inverse_for_A(b12, b22)` which takes in values of $b_{12}, b_{22}$ and returns the matrix


$$
\boldsymbol{B} = \begin{bmatrix}1 & b_{12} & -2/3\\ 0 & b_{22} & 1/3\end{bmatrix}   .
$$

```{code-cell}
def left_inverse_for_A(b12, b22):
    B = np.array([[b11, b12, b13], [b21, b22, b23]])
    return B
```

Now let's try plugging in different values of $b_{12}$ and $b_{22}$ and see that these all give us valid left inverses for $\boldsymbol{A}$.

```{code-cell}
B1 = left_inverse_for_A(b12 = 1, b22 = 2)
print('For b12 = 1, b22 = 2, we have BA = ')
print(np.round(np.dot(B1, A),4))

B2 = left_inverse_for_A(b12 = -341, b22 = 0.1)
print('For b12 = -341, b22 = 0.1, we have BA = ')
print(np.round(np.dot(B2, A),4))

B3 = left_inverse_for_A(b12 = 0, b22 = 50)
print('For b12 = 0, b22 = 50, we have BA = ')
print(np.round(np.dot(B3, A),4))
```

Indeed, no matter what values of $b_{12}$ and $b_{22}$ we plug in, we always get that $\boldsymbol{BA} = \boldsymbol{I}$, and hence $\boldsymbol{A}$ has many left inverses.

Here we illustrated the existence of many left inverses using a simple example which we could solve by hand; to find left inverses for bigger matrices we will need tools that we introduce in the next chapter.


## Right inverses for matrices

### When is a linear function surjective

As we saw in the previous section, a function $f$ has (at least one) right inverse as long as it is _surjective_.
Recall that a function $f:X\to Y$ is surjective if for every $y\in Y$ there exists $x\in X$ such that $f(x)= y$.

Now suppose again we have a linear function $f:\mathbb{R}^n \to \mathbb{R}^m$ given by $f(\boldsymbol{x}) = \boldsymbol{Ax}$.
Let $\boldsymbol{y}$ be an arbitrary vector in $\mathbb{R}^m$.
To check whether $f$ is surjective, we want to know whether there is a vector $\boldsymbol{x}\in \mathbb{R}^n$ such that $\boldsymbol{y} = \boldsymbol{Ax} = f(\boldsymbol{x})$.
Using our representation of $\boldsymbol{Ax}$ from the previous section, we want to know if there are coordinates $\boldsymbol{x} = (x_1,\dots,x_n)$ such that

$$
\boldsymbol{y} = x_1 \boldsymbol{A}[:,1] + x_2 \boldsymbol{A}[:, 2] + \cdots + x_n \boldsymbol{A}[:, n]   .
$$

By definition, we will always be able to find such coordinates $x_1,\dots,x_n$ as long as the columns $\boldsymbol{A}[:,1],\dots,\boldsymbol{A}[:,n]$ span all of $\mathbb{R}^m$.
Hence we have the following characterization of surjective functions which ties together our concepts for general functions, and our linear algebraic concept of span:

> _For an $m\times n$ matrix $\boldsymbol{A}$, the linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is surjective if and only if the columns of $\boldsymbol{A}$ span $\mathbb{R}^m$_.

A simple corollary of this fact is the following: if $m > n$, then $f(\boldsymbol{x}) = \boldsymbol{Ax}$ can _never_ be surjective.
This is because the minimum number of vectors which can span $\mathbb{R}^m$ is $m$, and so if $n$ is smaller than $m$, no $n$ vectors can ever span all of $\mathbb{R}^m$.
Thus, at the very least, for $f(\boldsymbol{x}) = \boldsymbol{Ax}$ to be surjective, we need that $m\leq n$.
Notice that this is coherent with our understanding of surjective functions from before: there we said that, intuitively, $f: X\to Y$ can only be surjective if $X$ is "bigger than" $Y$.
Here, this translates to the fact that if $m\leq n$, then $\mathbb{R}^n$ is "bigger than" $\mathbb{R}^m$.


### Right inverses for surjective linear functions

From before, we know that a function every surjective function $f: X \to Y$ has a least one right inverse $g:Y \to X$ such that $f\circ g = \text{id}_Y$.
In the case of linear functions of the form $f(\boldsymbol{x}) = \boldsymbol{Ax}$, a right inverse is another linear function $g(\boldsymbol{y}) = \boldsymbol{By}$ where $\boldsymbol{B}$ is an $n\times m$ matrix such that

$$
\boldsymbol{y} = (f\circ g)(\boldsymbol{y}) = f(g(\boldsymbol{y})) = f(\boldsymbol{By}) = \boldsymbol{A}\boldsymbol{By}   .
$$

In other words $g(\boldsymbol{y}) = \boldsymbol{By}$ is a right inverse if and only if $\boldsymbol{AB} = \boldsymbol{I}$ is the identity matrix on $\mathbb{R}^m$.
Like in the previous section, the condition $\boldsymbol{AB} = \boldsymbol{I}$ constitutes a linear system of equations, with $m^2$ constraints and $n\cdot m$ unknown variables (the entries of the matrix $\boldsymbol{B}$).
Since we know that for $f$ to be surjective we need $m \leq n$, we have that $m^2 \leq n\cdot m$, and so in fact $\boldsymbol{AB} = \boldsymbol{I}$ is a linear system with more unknowns than constraints -- this is also commonly known as an _underdetermined system_.
Typically, undetermined solutions have many possible solutions, and hence in general an surjective function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ will have many right inverses.

In what follows, we walk through a simple example in Python of finding a right inverse of a matrix.
Consider the function $f: \mathbb{R}^3 \to \mathbb{R}^2$ where $f(\boldsymbol{x}) = \boldsymbol{Ax}$ with

$$
\boldsymbol{A} = \begin{bmatrix}1 & 2  & 0\\ 0 & 0 & 3\end{bmatrix}  .
$$

It's easy to see that the vectors $(1,0)$, $(2,0)$ and $(0,3)$ span $\mathbb{R}^2$, and so we know from the previous subsection that $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is indeed surjective.

Let's define this as a numpy array.

```{code-cell}
A = np.array([[1,2,0], [0,0,3]])
A
```

A right inverse $\boldsymbol{B}$ for $\boldsymbol{A}$ will be a $3\times 2$ matrix of the form

$$
\boldsymbol{B} = \begin{bmatrix}b_{11} & b_{12}\\ b_{21} & b_{22}\\ b_{31} & b_{32}\end{bmatrix}  .
$$

The constraint $\boldsymbol{AB} = \boldsymbol{I}$ becomes

$$
\boldsymbol{AB} = \begin{bmatrix}1 & 2  & 0\\ 0 & 0 & 3\end{bmatrix}\begin{bmatrix}b_{11} & b_{12}\\ b_{21} & b_{22}\\ b_{31} & b_{32}\end{bmatrix} =  \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}  .
$$

If we carry out the above matrix multiplication, we are left with the following $4$ constraints:

$$
\begin{cases}
b_{11} + 2b_{21} = 1 & (1)\\
b_{12} + 2 b_{22} = 0 & (2)\\
3b_{31} = 0 & (3)\\
3b_{32} = 1& (4)  .
\end{cases}
$$

From (3) and (4) we immediately know that $b_{31} = 0$ and $b_{32} = 1/3$.
Moreover, from (1) and (2) we know that $b_{11} = 1-2b_{21}$ and $b_{12} = -2b_{22}$.
As we can see, we do not have enough constraints to fully determine the matrix $\boldsymbol{B}$: $b_{21}$ and $b_{22}$ are free to vary.
Let's check that no matter what choices of these values, we still get a right inverse.

```{code-cell}
b31 = 0
b32 = 1./3

def right_inverse_for_A(b21, b22):
    b11 = 1-2*b21
    b12 = -2*b22
    B = np.array([[b11, b12], [b21, b22], [b31, b32]])
    return B
```

Let's again try several values for $b_{21}$ and $b_{22}$, and check that they all give valid right inverses, i.e. that $\boldsymbol{AB} = \boldsymbol{I}$.

```{code-cell}
B1 = right_inverse_for_A(b21 = 1, b22 = 2)
print('For b21 = 1, b22 = 2, we have AB = ')
print(np.round(np.dot(A, B1),4))

B2 = right_inverse_for_A(b21 = -341, b22 = 0.1)
print('For b21 = -341, b22 = 0.1, we have AB = ')
print(np.round(np.dot(A, B2),4))

B3 = right_inverse_for_A(b21 = 0, b22 = 50)
print('For b21 = 0, b22 = 50, we have AB = ')
print(np.round(np.dot(A, B3),4))
```

Indeed, as expected, these each give valid right inverses for $f(\boldsymbol{x}) = \boldsymbol{Ax}$, reflecting the fact that, in general, surjective functions will have many right inverses.

Here we illustrated the existence of many right inverses using a simple example which we could solve by hand; to find right inverses for bigger matrices we will need tools that we introduce in the next chapter.


## Linear functions with inverses

As a special case of the previous two sections, we can have functions which are both injective _and_ surjective.
Such functions are called _bijective_.

For linear functions, we saw in the previous two sections that the injective and surjective linear functions are characterized by the following two statements:

- For an $m\times n$ matrix $\boldsymbol{A}$, the linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is injective if and only if the columns of $\boldsymbol{A}$ are linearly independent.
- For an $m\times n$ matrix $\boldsymbol{A}$, the linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is surjective if and only if the columns of $\boldsymbol{A}$ span $\mathbb{R}^m$.

We also saw that the function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is injective only if $n\leq m$, and surjective only if $m\leq n$.
Combining these two facts, a linear function can only be bijective if $m=n$, or in other words if the matrix $\boldsymbol{A}$ is square.
Therefore, we can characterize bijective linear functions with the following.

> _For an $m\times n$ matrix $\boldsymbol{A}$, the linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is bijective if and only if $\boldsymbol{A}$ is a square matrix, and the columns of $\boldsymbol{A}$ are linearly independent and span $\mathbb{R}^m$. That is, $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is bijective if and only if the columns of $\boldsymbol{A}$ form a basis for $\mathbb{R}^m$._

Recall that bijective functions have a unique complementary function $f^{-1}$ called an _inverse_ function.
In the case of a linear function $f(\boldsymbol{x}) = \boldsymbol{Ax}$, an inverse is a function  $f^{-1}(\boldsymbol{x}) = \boldsymbol{A}^{-1}\boldsymbol{x}$, where $\boldsymbol{A}^{-1}$ is a square matrix such that

$$
\boldsymbol{A}\boldsymbol{A}^{-1} = I\hspace{10mm}\text{ and }\hspace{10mm} \boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}  .
$$

The matrix $\boldsymbol{A}^{-1}$ is called the _inverse matrix_ of $\boldsymbol{A}$.
We call a matrix _invertible_ if the function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ has an inverse function.
Equivalently, using our characterization above, a square matrix $\boldsymbol{A}$ is invertible if and only if its columns are linearly independent and span all of $\mathbb{R}^m$.
Note that this is equivalent to saying that $\boldsymbol{A}$ is invertible if and only if its columns form a basis.

In numpy, we can use the function `np.linalg.inv(A)` to find the inverse of a matrix $\boldsymbol{A}$.
Before giving examples of when this works, let's see what happens if we try to use this function to invert functions which _aren't_ invertible.
For example, consider again the matrix $\boldsymbol{A}$ from the previous section

$$
\boldsymbol{A} = \begin{bmatrix}1 & 2  & 0\\ 0 & 0 & 3\end{bmatrix}  .
$$

This matrix is not square, and so by our discussion above, it cannot have an inverse.
Let's see what happens if we try to apply `np.linalg.inv()` to this matrix.

```{code-cell}
A = np.array([[1,2,0], [0,0,3]])
np.linalg.inv(A)
```

Indeed, numpy gives us an error saying that the array needs to be square in order to apply the function.
Next, let's what happens if the columns of $\boldsymbol{A}$ are not linearly independent and don't span all of $\mathbb{R}^m$.
For example, consider the matrix

$$
\boldsymbol{A} = \begin{bmatrix} 1 & -1\\ -1 &1\end{bmatrix}  .
$$

While this matrix is square, it's columns are not linearly independent, since $\boldsymbol{A}[:,1] = -\boldsymbol{A}[:,2]$, and for the same reason, the columns do not span $\mathbb{R}^2$.
When we try to apply the function `np.linalg.inv(A)` on a matrix we get:

```{code-cell}
A = np.array([[1,-1], [-1,1]])
np.linalg.inv(A)
```

In this case, we now get an error telling us that $\boldsymbol{A}$ is a singular matrix.
This is because $\boldsymbol{A}$ has lineary depdenent columns, and therefore is not invertible.
Thus we see that numpy requires us to pass the `np.linalg.inv()` function a valid, invertible matrix.
Let's see an example of doing this.
Consider the matrix

$$
\boldsymbol{A} = \begin{bmatrix} 1 & 2\\ 0 &3 \end{bmatrix}.
$$

This matrix is square and indeed has linearly independent columns/columns which span $\mathbb{R}^2$.
Let's try finding its inverse in Python.

```{code-cell}
A = np.array([[1,2], [0,3]])
A_inv = np.linalg.inv(A)
```

As we can see, this time numpy did not throw an error.
Moreover, we can check that `A_inv` is indeed an inverse by computing $\boldsymbol{A}\boldsymbol{A}^{-1}$ and $\boldsymbol{A}^{-1}\boldsymbol{A}$:

```{code-cell}
print(r'AA^{-1} = ')
print(np.round(np.dot(A, A_inv)))
print('A^{-1}A = ')
print(np.round(np.dot(A, A_inv)))
```

Indeed, we have that $\boldsymbol{A}\boldsymbol{A}^{-1} = \boldsymbol{I}$ and $\boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}$, verifying that $\boldsymbol{A}^{-1}$ is a valid inverse for $\boldsymbol{A}$.

Of course, we can also do this for bigger matrices, for example a $10 \times 10$ matrix. Below we give an example of such a matrix where each of the entries are drawn randomly from a normal distribution (such matrices can be shown to be invertible with very high probability).

```{code-cell}
n = 10

A = np.random.normal(size = (n,n))
A_inv = np.linalg.inv(A)

print('AA^{-1} = ')
print(np.round(np.dot(A, A_inv)))
print('A^{-1}A = ')
print(np.round(np.dot(A, A_inv)))
```
