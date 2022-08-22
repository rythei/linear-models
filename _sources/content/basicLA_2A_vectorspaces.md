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

# Vectors and vector spaces

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Real Vector Spaces

In data science, we often want to view vectors as a one-dimensional data array (like a list of numbers) that represents a point in a higher-dimensional analogue of the two-dimensional Euclidean plane.
To be able to manipulate vectors, it is important to understand that such a vector is an element of a vector space, where a vector space is an abstract thing whose elements satisfy certain mathematical rules.
These rules are essentially vector addition and scalar multiplication.
Anything that satisfies these rules is a vector.
Here we limit out discussion to real vector spaces, meaning vector spaces involving real numbers, since in data science we typically deal with real vectors and real scalars.

Here is a definition for real vector spaces.
A vector space is a set $V$ equipped with two operations:
* (i) Addition: adding any pair of vectors $\boldsymbol{v},\boldsymbol{w} \in V$ yields another vector $\boldsymbol{z}=\boldsymbol{v}+\boldsymbol{w}$ that is also in the same vector space, i.e., $\boldsymbol{z}=\boldsymbol{v}+\boldsymbol{w} \in V$
* (ii) Scalar Multiplication: multiplying a vector $\boldsymbol{v} \in V$ by a scalar $c \in \mathbb{R}$ yields another vector $\boldsymbol{z}=c \boldsymbol{v}$ that is also in the same vector space, i.e., $\boldsymbol{z} = c \boldsymbol{v} \in V$.

The  operations  of  adding  two  vectors  and  multiplying  a  vector  by  a  scalar  are  simple,  but  they  are very powerful.
Indeed, these two operations form the foundation for all of linear algebra.
One reason the two operations of adding two vectors and multiplying a vector by a scalar are so powerful is that the output of each of these two operations is itself a vector, upon which these operations can be performed again.
Further, these two operations obey the following axioms, that are valid for all $\boldsymbol{u}, \boldsymbol{v},\boldsymbol{w} \in V$ and all scalars $c,d \in \mathbb{R}$:

* Commutativity of addition: $\boldsymbol{v} + \boldsymbol{w} = \boldsymbol{w} + \boldsymbol{v}$.
* Associativity of addition: $\boldsymbol{u} + (\boldsymbol{v}+\boldsymbol{w}) = (\boldsymbol{u}+\boldsymbol{v}) + \boldsymbol{w}$.
* Additive identity: There is a element $\boldsymbol{0} \in V$ that satisfy $\boldsymbol{v} + \boldsymbol{0} = \boldsymbol{v}$ and $\boldsymbol{0}+\boldsymbol{v}=\boldsymbol{v}$.
* Additive inverse: For each $\boldsymbol{v} \in V$ there is an element $-\boldsymbol{v} \in V$ such that $\boldsymbol{v}+(-\boldsymbol{v}) = \boldsymbol{0}$ and $-\boldsymbol{v}+\boldsymbol{v} = \boldsymbol{0}$.
* Distributivity of scalar multiplication: $(c+d) \boldsymbol{v} = (c\boldsymbol{v} ) + (d\boldsymbol{v})$, and $c (\boldsymbol{v}+\boldsymbol{w)} = (c\boldsymbol{v} ) + (c\boldsymbol{w})$.
* Associativity of scalar multiplication: $c(d\boldsymbol{v}) = (cd)\boldsymbol{v}$.
* Unit for scalar multiplication: the scalar $1\in \mathbb{R}$ satisfies $1\boldsymbol{v}=\boldsymbol{v}$.

To better understand the two operations and the axioms, it is best to start by looking at vectors in $2$ dimensions.
In the following, we will use the two vectors $\boldsymbol{v}$ and $\boldsymbol{w}$ that are vectors in $\boldsymbol{v},\boldsymbol{w} \in \mathbb{R}^2$.

```{code-cell}
v = np.array([3,1])
w = np.array([2,4])
```

You know already how add two vectors,



$$ \boldsymbol{v} + \boldsymbol{w} =
\begin{bmatrix}
v_{1}+w_{1}   \\
v_{2}+w_{2}   \\
\end{bmatrix},$$



and we see indeed that addition produces another vector in $\mathbb{R}^2$. Let's use NumPy to do the computations.

```{code-cell}
v+w
```

You can also simply check whether the axioms that we formulated above hold for this example, for instance, we can start by checking commutativity of addition.

```{code-cell}
v+w == w+v
```

It is also easy to verify that the additive inverse holds for this example.

```{code-cell}
-v + v
```

You can try and check all the other axioms yourself. In any case, it is often a good idea to visualize vectors in the the two-dimensional plane to build some intuition. Here we think about vectors in a geometric sense.

```{code-cell}
origin = np.zeros(2) # we need to define the origin of the coordinate system
plt.figure(figsize=(5,5))
plt.quiver(*origin, *v, color=['r'], scale=1, units='xy') # plot the vector v
plt.quiver(*origin, *w, color=['b'], scale=1, units='xy') # plot the vector w
plt.quiver(*origin, *v+w, color=['g'], scale=1, units='xy') # plot the vector v+w
plt.grid()

plt.xlim(-1,5)
plt.ylim(-1,5)
plt.gca().set_aspect('equal')
plt.show()
```

In the above, we use the `quiver` function from `matplotlib`, allows us to visualize vectors.

The law of parallelogram of vector addition states that if two adjacent sides of a parallelogram represents two given vectors in magnitude and direction, then the diagonal starting from the intersection of two vectors represent their sum.

```{code-cell}
plt.figure(figsize=(5,5))
plt.quiver(*origin, *v, color=['r'], scale=1, units='xy') # plot the vector v
plt.quiver(*v, *w, color=['b'], scale=1, units='xy') # plot the vector w

plt.quiver(*origin, *w, color=['b'], scale=1, units='xy') # plot the vector v
plt.quiver(*w, *v, color=['r'], scale=1, units='xy') # plot the vector w

plt.quiver(*origin, *v+w, color=['g'], scale=1, units='xy') # plot the vector v+w
plt.grid()
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.gca().set_aspect('equal')
plt.show()
```

These concepts generalize to higher dimensional spaces, of course. While it becomes difficult to visualize for high-dimensional spaces, we can still visualize vectors in a 3 dimensional space. Let's consider the following 3 vectors.

```{code-cell}
x = np.array([3,1,2])
y = np.array([2,2,2])
z = np.array([1,2,3])
```

We can also add the three vectors.

```{code-cell}
xyz = x + y + z
print(xyz)
```

Now, let's plot the three vectors and the new vector that is produced by vector addition in 3d.

```{code-cell}
origin = np.zeros(3) # we need to define the origin of the coordinate system

fig = plt.figure(figsize=(7,7))
ax = fig.gca(projection='3d')

ax.quiver(*origin,*x, length=1, arrow_length_ratio=0.1, colors='b') # blue
ax.quiver(*origin,*y, length=1, arrow_length_ratio=0.1, colors='g') # green
ax.quiver(*origin,*z, length=1, arrow_length_ratio=0.1, colors='y') # yellow

ax.quiver(*origin,*xyz, arrow_length_ratio=0.05, colors='r') # red

plt.grid()
ax.set_xlim(-1,6)
ax.set_ylim(-1,6)
ax.set_zlim(-1,6)

plt.show()
```

### Linear Combinations

So far we have seen how vector addition produces a new vector.
However, we can also compute a weighted sum that takes the form:


$$
c_1 \boldsymbol{v}_1 + c_2 \boldsymbol{v}_2 + \dots + c_n \boldsymbol{v}_n
$$



where $c_1,\dots,c_n \in \mathbb{R}$ are scalar values and $\boldsymbol{v}_1, \dots, \boldsymbol{v}_n \in \mathbb{R}^n$ are vectors.
This expression is called a _linear combination_.
Linear combinations are a central concept in linear algebra.

We will spend a lot of time on linear combinations.
Let's start by looking at them in $\mathbb{R}^2$ and $\mathbb{R}^3$.

### Linear Combinations in $\mathbb{R}^2$

Given the vectors $\boldsymbol{v} = [3,1]^T$ and $\boldsymbol{w}= [2,4]^T$, we can express any vector in $\mathbb{R}^2$ as a linear combination of these two vectors:

$$ c_1 \boldsymbol{v} + c_2 \boldsymbol{w} =
c_1\begin{bmatrix}
{3}   \\
{1}   \\
\end{bmatrix}
+
c_2\begin{bmatrix}
{2}   \\
{4}   \\
\end{bmatrix}.$$

For instance, we obtain the vector $[1.5,5.5]^T$ as the following linear combination:

```{code-cell}
-0.5*v + 1.5*w
```

So how do we know that the coefficients are $c_1=-0.5$ and $c_2=1.5$? Well, you obtain the coefficients by solving the following system of linear equations:


$$
\begin{cases} 3 c_1 + 2 c_2 = 1.5 \\ c_1 + 4 c_2 = 5.5\end{cases}
$$


You know how to solve simple systems of liner equations. Note, that you can express the system of linear equations also by using vector notation


$$
c_1\begin{bmatrix}
{3}   \\
{1}   \\
\end{bmatrix}
+
c_2\begin{bmatrix}
{2}   \\
{4}   \\
\end{bmatrix}
= \begin{bmatrix}
{1.5}   \\
{5.5}   \\
\end{bmatrix}
.
$$


### Linear Combinations in $\mathbb{R}^3$

Here is an example in $\mathbb{R}^3$. Given the vectors $\boldsymbol{x} = [3,1,2]^T$, $\boldsymbol{y}= [2,2,2]^T$ and $\boldsymbol{z}= [1,2,3]^T$, we can express any vector in $\mathbb{R}^3$ as a linear combination of these three vectors:


$$
c_1 \boldsymbol{x} + c_2 \boldsymbol{y} + c_3 \boldsymbol{z}=c_1\begin{bmatrix} {3}   \\{1}   \\{2}   \\\end{bmatrix}+c_2\begin{bmatrix} {2}   \\{2}   \\{2}   \\\end{bmatrix}+c_3\begin{bmatrix} {1}   \\{2}   \\{3}   \\\end{bmatrix}.
$$


For instance, we obtain the vector $[3.5,3.5,5]^T$ as the following linear combination:

```{code-cell}
0.5*x + 0.5*y + 1*z
```

Again, you obtain the coefficients $c_1=0.5$, $c_2=0.5$, and $c_3=1$ by solving the following system of linear equations:


$$
\begin{cases} 3 c_1 + 2 c_2 + c_3 = 1 \\ c_1 + 2 c_2 + 2 c_3 = 2 \\ 2 c_1 + 2 c_2 + 3 c_3 = 3\end{cases}
$$


In a later chapter we will express this system more concisely by using matrix notation, and you will learn about methods that help you to efficiently solve such systems of linear equations. For instance, using NumPy (without explaining the details here) you can readily obtain the solution to this system as follows.

```{code-cell}
A = np.array((x,y,z)).T
b = np.array((3.5,3.5,5))
out = np.linalg.solve(A,b)
print(out)
```

## Span and Linear Independence

We have seen that we can express any vector in $\mathbb{R}^2$ using the vectors $\boldsymbol{v}$ and $\boldsymbol{w}$, and any vector in $\mathbb{R}^3$ using the vectors $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$. A natural question that arises is whether there is something special about the vectors that we picked. The answer is yes, they are special because $\boldsymbol{v}$ and $\boldsymbol{w}$ span the entire plane in $\mathbb{R}^2$ and $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$ span the entire space in $\mathbb{R}^3$. That is, because for any $a,b \in \mathbb{R}$ we an always find a solution:


$$
c_1\begin{bmatrix}
{3}   \\
{1}   \\
\end{bmatrix}
+
c_2\begin{bmatrix}
{2}   \\
{4}   \\
\end{bmatrix}
= \begin{bmatrix}
{a}   \\
{b}   \\
\end{bmatrix}
$$


Similar for any $a,b,c \in \mathbb{R}$ we an always find a solution:


$$
c_1\begin{bmatrix}
{3}   \\
{1}   \\
{2}   \\
\end{bmatrix}
+
c_2\begin{bmatrix}
{2}   \\
{2}   \\
{2}   \\
\end{bmatrix}
+
c_3\begin{bmatrix}
{1}   \\
{2}   \\
{3}   \\
\end{bmatrix}= \begin{bmatrix}
{a}   \\
{b}   \\
{c}   \\
\end{bmatrix}.
$$


To be more precise, $\boldsymbol{v}$ and $\boldsymbol{w}$ span the entire space in $\mathbb{R}^2$ and $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$ span the entire space in $\mathbb{R}^3$. In order to span the entire space in $\mathbb{R}^2$ we require that we have two independent vectors and to span the entire space $\mathbb{R}^3$ we require that we have three independent vectors, and to span the entire space $\mathbb{R}^n$ we require that we have $n$ independent vectors.

We call the vector space elements $\boldsymbol{v}_1,\dots,\boldsymbol{v}_k \in V$ linearly dependent if there exists scalars $c_1,\dots,c_k$, not all zero, such that


$$
c_1 \boldsymbol{v}_1 + \dots + c_k \boldsymbol{v}_k = \boldsymbol{0}
$$


Elements that are not linearly dependent are called linearly independent.


Here is an example of linear dependent vectors: $\boldsymbol{x}=(1,2,3)^T$, $\boldsymbol{y}=(0,3,2)^T$ and $\boldsymbol{z}=(-1,7,3)^T$. That is, because $x-3y+z = \boldsymbol{0}$.

```{code-cell}
x = np.array([1,2,3])
y = np.array([0,3,2])
z = np.array([-1,7,3])
```

```{code-cell}
x -3*y + z
```

But, note that $\boldsymbol{x}$ and $\boldsymbol{y}$ are linearly independent. To see this, suppose that that


$$
c_1 \boldsymbol{x} + c_2 \boldsymbol{y} =
\begin{bmatrix}
{c_1}   \\
{2 c_1 + 3 c_2}   \\
{3 c_1 + 2 c_2}   \\
\end{bmatrix}
= \begin{bmatrix}
{0}   \\
{0}   \\
{0}   \\
\end{bmatrix}.
$$


For this to happen $c_1$ and $c_2$ must satisfy the linear system


$$
\begin{cases} c_1 = 0 \\ 2 c_1 + 3 c_2  = 0 \\ 3 c_1 + 2 c_2 = 0\end{cases}
$$


But this system has only the trivial solution $c_1 = c_2 = 0$. Hence, $\boldsymbol{x}$ and $\boldsymbol{y}$ are linearly independent.

Another way to illustrate this, is by visualizing that $\boldsymbol{z}$ lies in the plane that is spanned by $\boldsymbol{x}$ and $\boldsymbol{y}$ .
Recall, if $\boldsymbol{x}$  and $\boldsymbol{y}$ are parallel to given plane $P$, then the plane $P$ is said to be spanned by $\boldsymbol{x}$ and $\boldsymbol{y}$.

```{code-cell}
# the cross product is a vector normal to the plane
cp = np.cross(x, y)
a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
d = np.dot(cp, origin)

origin = np.zeros(3) # we need to define the origin of the coordinate system

fig = plt.figure(figsize=(16,7))
ax = fig.gca(projection='3d')

ax.quiver(*origin,*x, length=1, arrow_length_ratio=0.1, colors='b') # blue
ax.quiver(*origin,*y, length=1, arrow_length_ratio=0.1, colors='g') # green
ax.quiver(*origin,*z, length=1, arrow_length_ratio=0.1, colors='y') # yellow

xx, yy = np.meshgrid(np.arange(-2,8), np.arange(-2,8))
q = (d - a * xx - b * yy) / c
# plot the plane
ax.plot_surface(xx, yy, q, alpha=0.5)
ax.view_init(15, 120)

plt.grid()
ax.set_xlim(-1,6)
ax.set_ylim(-1,6)
ax.set_zlim(-1,6)

plt.show()
```

Clearly, we can see that $\boldsymbol{z}$ lies within the plane that is spanned by $\boldsymbol{x}$ and $\boldsymbol{z}$. We can also create a few random linear combinations of $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$ and observe that they all these vectors will lie within the plane.

```{code-cell}
np.random.seed(1)
fig = plt.figure(figsize=(16,7))
ax = fig.gca(projection='3d')

for i in range(30):
    random_linear_combination = np.random.uniform(-1,1,1) * x + np.random.uniform(-1,1,1) * y + np.random.uniform(-1,1,1) * z
    ax.quiver(*origin,*random_linear_combination, length=1, arrow_length_ratio=0.1, colors='r') # blue

xx, yy = np.meshgrid(np.arange(-9,10), np.arange(-9,10))
q = (d - a * xx - b * yy) / c
# plot the plane
ax.plot_surface(xx, yy, q, alpha=0.5)
ax.view_init(15, 120)

plt.grid()
ax.set_xlim(-1,6)
ax.set_ylim(-1,6)
ax.set_zlim(-1,6)

plt.show()
```

Now, let's change the vector $\boldsymbol{z}$ so that $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$ are linearly independent. For instance, let's pick $\boldsymbol{z} = (-2,3,9)^T$

```{code-cell}
z = np.array([-2,3,9])
```

Because $\boldsymbol{x}$, $\boldsymbol{y}$ and $\boldsymbol{z}$ are linearly independent, we will observe that vectors start to stick out of the plane.

```{code-cell}
np.random.seed(1)
fig = plt.figure(figsize=(16,7))
ax = fig.gca(projection='3d')

for i in range(30):
    random_linear_combination = np.random.uniform(-1,1,1) * x + np.random.uniform(-1,1,1) * y + np.random.uniform(-1,1,1) * z
    ax.quiver(*origin,*random_linear_combination, length=1, arrow_length_ratio=0.1, colors='r') # blue

xx, yy = np.meshgrid(np.arange(-4,10), np.arange(-4,10))
q = (d - a * xx - b * yy) / c
# plot the plane
ax.plot_surface(xx, yy, q, alpha=0.5)
ax.view_init(15, 120)

plt.grid()
ax.set_xlim(-1,6)
ax.set_ylim(-1,6)
ax.set_zlim(-1,6)

plt.show()
```

## Basis vectors

We have seen that we require a sufficient number of distinct vectors in order to span a vector space. But, if we have too many vectors in the spanning set, then we will have that some of these vectors are linear dependent. Hence, an optimal spanning set is a set of linearly independent vectors for a given space. This optimal spanning set is called a basis. A basis of a vector space $V$ is a set of vectors $\boldsymbol{v}_1,\dots,\boldsymbol{v}_n \in V$ that (i) spans $V$ and (ii) is linearly independent.

In summary:

* Basis vectors must be linearly independent, i.e., if you multiply $\boldsymbol{v}_1$ by any scalar $c$ you will never be able to produce $\boldsymbol{v}_2$.

* Basis vectors must span the whole space, i.e., any vector in the space can be written as a linear combination of the basis vectors for a given space.


We have seen above already some examples for basis vectors for $\mathbb{R}^2$ and $\mathbb{R}^3$. The basis vectors that we considered allowed us to represent all the vectors in a given space as a linear combination of them. But, note that basis vectors are not unique. There are many sets of basis vectors that satisfy the two properties above and there are ``better`` basis vectors for $\mathbb{R}^2$ and $\mathbb{R}^3$ than those that we considered above.


### Standard basis

The standard basis of $\mathbb{R}^n$ consists of $n$ vectors


$$
\boldsymbol{e}_1=
\begin{bmatrix}
1  \\
0   \\
\vdots \\
0
\end{bmatrix}, \quad
%
\boldsymbol{e}_2=
\begin{bmatrix}
0  \\
1   \\
\vdots \\
0
\end{bmatrix}, \quad \dots \quad
%
\boldsymbol{e}_n=
\begin{bmatrix}
0  \\
0   \\
\vdots \\
1
\end{bmatrix}
$$


For instance, the standard basis vectors for $\mathbb{R}^2$ are


$$
\boldsymbol{e}_1=
\begin{bmatrix}
1  \\
0   \\
\end{bmatrix}, \quad \text{and} \quad
%
\boldsymbol{e}_2=
\begin{bmatrix}
0  \\
1   \\
\end{bmatrix}.
$$


We require two basis vectors, because the vector space has dimension 2. In order to represent all vectors in $\mathbb{R}^3$ we need three basis vectors, and the standard basis vectors for $\mathbb{R}^3$ are


$$
\boldsymbol{e}_1=
\begin{bmatrix}
1  \\
0   \\
0 \\
\end{bmatrix}, \quad
%
\boldsymbol{e}_2=
\begin{bmatrix}
0  \\
1   \\
0 \\
\end{bmatrix},\quad
%
\boldsymbol{e}_3=
\begin{bmatrix}
0  \\
0   \\
1 \\
\end{bmatrix}.
$$


Let's visualize the standard basis vectors $\boldsymbol{e}_1$ and $\boldsymbol{e}_2$ for $\mathbb{R}^2$.

```{code-cell}
e1 = np.array([1,0])
e2 = np.array([0,1])
```

```{code-cell}
origin = np.array((0,0))
plt.figure(figsize=(5,5))
plt.quiver(*origin, *e1, color=['g'], scale=1, units='xy') # green
plt.quiver(*origin, *e2, color=['r'], scale=1, units='xy') # red
plt.grid()
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.gca().set_aspect('equal')
plt.show()
```

Recall, our vector $\boldsymbol{v}=(3,1)^T$ was defined as follows.

```{code-cell}
v
```

We can express this vector also as a linear combination using the standard basis vectors $\boldsymbol{e}_1$ and $\boldsymbol{e}_2$ as


$$
3 \begin{bmatrix}
1  \\
0   \\
\end{bmatrix} +
%
1 \begin{bmatrix}
0  \\
1   \\
\end{bmatrix}
=\begin{bmatrix}
3  \\
1   \\
\end{bmatrix}.
$$

```{code-cell}
3*e1 + 1*e2
```

Here, the coefficients are $c_1=3$ and $c_2=1$.

### Other (non-standard) bases and changing between bases

We can also consider other bases for $\mathbb{R}^2$. For example, the following vectors constitute a basis for $\mathbb{R}^2$.

$$
\boldsymbol{e}'_1=
\begin{bmatrix}
3  \\
0   \\
\end{bmatrix}, \quad \text{and} \quad
%
\boldsymbol{e}'_2=
\begin{bmatrix}
0  \\
0.5   \\
\end{bmatrix}.
$$


This is a valid basis, since the two vectors $\boldsymbol{e}'_1$ and $\boldsymbol{e}'_2$ span the the space in $\mathbb{R}^2$ and they are linearly independent.

Again, we can express the vector $\boldsymbol{v}$ also as a linear combination using the new basis vectors $\boldsymbol{e}'_1$ and $\boldsymbol{e}'_2$ as


$$
1 \begin{bmatrix}
3  \\
0   \\
\end{bmatrix} +
%
2 \begin{bmatrix}
0  \\
0.5   \\
\end{bmatrix}
=\begin{bmatrix}
3  \\
1   \\
\end{bmatrix}.
$$

```{code-cell}
e1_new = np.array([3,0])
e2_new = np.array([0,0.5])
```

```{code-cell}
1*e1_new + 2*e2_new
```

The coefficients $(1,2)$ here are called the coordinates of $V$ with respect to the basis $\boldsymbol{e}'_1,\boldsymbol{e}'_2$. Thus the same vector $V$ can be represented by different numbers, depending on the basis we choose to work with. Generally when we write $\boldsymbol{v}=(3,1)^T$, we are implicitly assuming that we are working with the standard basis, however it is equally valid to define $\boldsymbol{v} = (1,2)^T$ if we make clear that we are working with the basis $\boldsymbol{e}'_1,\boldsymbol{e}'_2$.

Note, that we obtain the standard normal basis if we normalize $\boldsymbol{e}'_1$ and $\boldsymbol{e}'_2$, i.e., $\boldsymbol{e}_1 = \frac{\boldsymbol{e}'_1}{\|\boldsymbol{e}'_1 \|_2}$ and $\boldsymbol{e}_2 = \frac{\boldsymbol{e}'_2}{\|\boldsymbol{e}'_2 \|_2}$.

```{code-cell}
e1_new / np.linalg.norm(e1_new)
```

```{code-cell}
e2_new / np.linalg.norm(e2_new)
```

Here is another basis.

```{code-cell}
e1_new = np.array([0.70710678, 0.70710678])
e2_new = np.array([-0.70710678,  0.70710678])
```

If we visualize this basis, we can see that this basis is rotated by 45 degrees counterclockwise

```{code-cell}
origin = np.array((0,0))
plt.figure(figsize=(5,5))
plt.quiver(*origin, *e1_new, color=['g'], scale=1, units='xy') # green
plt.quiver(*origin, *e2_new, color=['r'], scale=1, units='xy') # red
plt.grid()
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.gca().set_aspect('equal')
plt.show()
```

Again, we can express the vector $\boldsymbol{v}$ in terms of this new basis as follows.

```{code-cell}
2.82842713 * e1_new -1.41421356 * e2_new
```

So how did we obtain the new coefficients $c_1=2.82842713$ and $c_2=-1.41421356$? For the example $\boldsymbol{e}'_1, \boldsymbol{e}'_2$ we could more or less guess the coefficients $(1,2)$, here it is less obvious. Since we can see that the basis vectors are rotated 45 degrees counterclockwise, we can assume that the new basis was formed by the following functions:


$$
e'_1 = f(\boldsymbol{e}_1, \boldsymbol{e}_2) = \cos(\theta) \boldsymbol{e}_1 + \sin(\theta) \boldsymbol{e}_2
$$



and


$$
\boldsymbol{e}'_2 = f(\boldsymbol{e}_1, \boldsymbol{e}_2) = -\sin(\theta) \boldsymbol{e}_1 + \cos(\theta) \boldsymbol{e}_2
$$



Let's verify this.

```{code-cell}
np.cos(np.radians(45)) * e1 + np.sin(np.radians(45)) * e2
```

```{code-cell}
-np.sin(np.radians(45)) * e1 + np.cos(np.radians(45)) * e2
```

Indeed, we see that we yield the new basis vectors. We can also use the same function to rotate the coefficients form the old basis into the space of the basis as follows.

```{code-cell}
3 * np.cos(np.radians(45)) + 1 * np.sin(np.radians(45))
```

```{code-cell}
3 * - np.sin(np.radians(45)) + 1 * np.cos(np.radians(45))
```

This method is somewhat specialized to the current choice of basis, since we guessed that it was a rotation. Later, when we discuss matrices, will see how to change between any arbitrary bases.
