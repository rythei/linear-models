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

# Projections

In this section, we study special linear maps called _projections_. Formally, a projection $P(\boldsymbol{x})$ is any linear map such that $P^2 = P$. In other words, a projection is simply a _idemptotent_ linear map.

## Projections onto a vector

We begin by considering perhaps the simplest possible projection: a projection onto a single vector. Intuitively, this is probably something you've already seen in high school math. The usual diagram given for this concept is below.

<img src="img/projection_2d.png" style="zoom:50%;" />

In the above figure, we are projecting a vector $\boldsymbol{a}$ onto a vector $\boldsymbol{b}$. The resulting projection is the vector $\text{proj}_\boldsymbol{b}(\boldsymbol{a}) = \boldsymbol{a}_1$, which is always parallel to $\boldsymbol{b}$. The vector $\boldsymbol{a}_2$ is the "residual" of the projection, which is  $\boldsymbol{a}_2 = \boldsymbol{a} - \boldsymbol{a}_1 = \boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a})$. Note that visually from the diagram, we have that $\boldsymbol{a} = \boldsymbol{a}_1 + \boldsymbol{a}_2$, which is of course obvious from the definitions of $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$. We will see below that this diagram is in fact representing a special case of projection onto a vector -- namely, it represents an _orthogonal_ projection.

### Orthogonal projections onto a vector

There is a simple formula for the orthogonal projection of a vector $\boldsymbol{a} \in\mathbb{R}^n$ onto another vector $\boldsymbol{b}\in \mathbb{R}^n$. It is given by the following:



$$
\text{proj}_\boldsymbol{b}(\boldsymbol{a}) = \frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b}
$$



Notice that $\frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}$ is just a scalar (assuming $\boldsymbol{b}\neq 0$), and so $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$ is really just a rescaled version of the vector $\boldsymbol{b}$. This means that for any vector $\boldsymbol{a}$, $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$ is always parallel to $\boldsymbol{b}$ -- this is why we say that it is a projection "onto" $\boldsymbol{b}$. Why is this called an 'orthogonal' projection? This is because $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$ is always orthogonal to the "residual" $\boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a})$. Let's check that this is in fact true by computing $\text{proj}_\boldsymbol{b}(\boldsymbol{a})^\top (\boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a}))$.



$$
\text{proj}_\boldsymbol{b}(\boldsymbol{a})^\top (\boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a})) = \left(\frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b}\right)^\top\left(\boldsymbol{a} - \frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b}\right) = \frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\left(\boldsymbol{b}^\top \boldsymbol{a} - \boldsymbol{b}^\top \boldsymbol{a}\frac{\boldsymbol{b}^\top \boldsymbol{b}}{\boldsymbol{b}^\top \boldsymbol{b}}\right) = \frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}(\boldsymbol{b}^\top \boldsymbol{a} - \boldsymbol{b}^\top \boldsymbol{a}) = 0
$$



Hence the angle between $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$ and $\boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a})$ is always $90^\circ$. You can also see this visually in the figure above.

**Remark:** In the QR decomposition section, we saw the formula $\frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b}$ appear in the Gram--Schmidt orthogonalization procedure. This is no coincidence: there, we computed



$$
\boldsymbol{u}_j = \boldsymbol{a}_j - \sum_{i = 1}^{j-1}\frac{\boldsymbol{u}_i^\top \boldsymbol{a}_j}{\boldsymbol{u}_i^\top \boldsymbol{u}_i}\boldsymbol{u}_i = \boldsymbol{a}_j - \sum_{i=1}^{j-1}\text{proj}_{\boldsymbol{u}_i}(\boldsymbol{a}_j)
$$



That is, $\boldsymbol{u}_j$ was the residual after projecting $\boldsymbol{a}_j$ onto each of $\boldsymbol{u}_1,\dots, \boldsymbol{u}_{j-1}$.

As we mentioned previously, the projections we consider are _linear_ maps. We know that all linear maps can be represented as matrices. Let's see how we can represent $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$ as a matrix transformation. Using the associativity of inner and outer products, we can rearrange the formula for $\text{proj}_\boldsymbol{b}$ to see



$$
\text{proj}_\boldsymbol{b}(\boldsymbol{a}) = \frac{\boldsymbol{b}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b} = \frac{1}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{b}\boldsymbol{b}^\top \boldsymbol{a} = \frac{\boldsymbol{b}\boldsymbol{b}^\top}{\boldsymbol{b}^\top \boldsymbol{b}}\boldsymbol{a} = \boldsymbol{P_ba}
$$



where $\boldsymbol{P_b} = \frac{\boldsymbol{bb}^\top}{\boldsymbol{b}^\top \boldsymbol{b}}$ is an $n\times n$ matrix.

As we mentioned before, projections should by definition satisfy the idempotence property $\boldsymbol{P}^2 = \boldsymbol{P}$. Let's check that this is true for $\boldsymbol{P_b}$. We have



$$
\boldsymbol{P_b}^2 = \frac{\boldsymbol{bb}^\top}{\boldsymbol{b}^\top \boldsymbol{b}}\frac{\boldsymbol{bb}^\top}{\boldsymbol{b}^\top \boldsymbol{b}} = \frac{1}{(\boldsymbol{b}^\top \boldsymbol{b})^2}\boldsymbol{bb}^\top \boldsymbol{bb}^\top = \frac{1}{(\boldsymbol{b}^\top \boldsymbol{b})^2}\boldsymbol{b}(\boldsymbol{b}^\top \boldsymbol{b})\boldsymbol{b}^\top=\frac{\boldsymbol{b}^\top \boldsymbol{b}}{(\boldsymbol{b}^\top \boldsymbol{b})^2}\boldsymbol{bb}^\top = \frac{\boldsymbol{bb}^\top}{\boldsymbol{b}^\top \boldsymbol{b}} = \boldsymbol{P_b}
$$



Indeed, $\boldsymbol{P_b}$ is idempotent.

Let's look at some $2$-d examples of orthogonal projections. First, let's define a function `orthogonal_projection(b)` which takes in a vector $\boldsymbol{b}$ and returns the projection matrix $\boldsymbol{P_b} = \frac{\boldsymbol{bb}^\top}{\boldsymbol{b}^\top \boldsymbol{b}}$.

```{code-cell}
import numpy as np

def orthogonal_projection(b):
    return np.outer(b,b)/np.dot(b,b)
```

 Now let's test this out with a vector that we'd like to project onto, say $\boldsymbol{b}=\begin{bmatrix}1\\2\end{bmatrix}$. Let's visualize $\boldsymbol{b}$.

```{code-cell}
import matplotlib.pyplot as plt

b = np.array([1,2])
origin = np.zeros(2)

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend()
plt.show()
```

Next, let's compute the projection matrix $\boldsymbol{P_b}$ using the function we defined above.

```{code-cell}
Pb = orthogonal_projection(b)
```

Just to make sure we've done things correctly, let's verify that $\boldsymbol{P_b}$ is idempotent, by checking that $\boldsymbol{P_b}^2 = \boldsymbol{P_b}$.

```{code-cell}
Pb2 = np.dot(Pb, Pb)
np.allclose(Pb2, Pb)
```

Indeed it is. Now, let's try projecting a vector, say $\boldsymbol{a} = \begin{bmatrix}1\\ 1\end{bmatrix}$, onto $\boldsymbol{b}$.

```{code-cell}
a = np.array([1, 1])
proj_b_a = np.dot(Pb, a) # compute the projection of a onto b
residual = a - proj_b_a

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.quiver(*origin, *a, label='a', scale=1, units='xy', color='green')
plt.quiver(*origin, *proj_b_a, label='proj_b(a)', scale=1, units='xy', color='red')
plt.quiver(*proj_b_a, *residual, label='a - proj_b(a)', scale=1, units='xy', color='orange')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend(loc='upper left')
plt.show()
```

This plot now largely replicates the figure we saw earlier: we see that 1) the projection of $\boldsymbol{a}$ onto $\boldsymbol{b}$ is a vector (in red) which is parallel to $\boldsymbol{b}$ and 2) the residual $\boldsymbol{a} - \text{proj}_\boldsymbol{b}(\boldsymbol{a})$ is at a $90^\circ$ angle from $\text{proj}_\boldsymbol{b}(\boldsymbol{a})$.

### Oblique projections onto a vector

While orthogonal projections are commonly used, and in many ways special, they are not the only way we can project onto a vector. Indeed, we can define a projection of a vector $\boldsymbol{a}$ onto another vector $\boldsymbol{b}$ not just along the direction orthogonal to $\boldsymbol{b}$, but along any arbitrary direction. The projection of a vector $\boldsymbol{a}$ onto the vector $\boldsymbol{b}$ _along the direction perpendicular to $\boldsymbol{c}$_ is given by the following:



$$
\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a}) = \frac{\boldsymbol{c}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{c}}\boldsymbol{b}
$$



Again, $\frac{\boldsymbol{c}^\top \boldsymbol{a}}{\boldsymbol{b}^\top \boldsymbol{c}}$ is just a scalar, so this vector is again just a rescaled version of $\boldsymbol{b}$. We can also rearrange this formula to write it as a linear function in terms of a matrix



$$
\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}} = \frac{\boldsymbol{bc}^\top}{\boldsymbol{b}^\top \boldsymbol{c}}
$$



So that $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a}) = \boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}\boldsymbol{a}$. Let's verify that $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$ also satisfies the idempotence property $\boldsymbol{P}^2 = \boldsymbol{P}$.



$$
\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}^2 = \frac{\boldsymbol{bc}^\top}{\boldsymbol{b}^\top \boldsymbol{c}}\frac{\boldsymbol{bc}^\top}{\boldsymbol{b^\top c}} = \frac{1}{(\boldsymbol{b^\top c})^2}\boldsymbol{bc}^\top \boldsymbol{bc}^\top = \frac{1}{(\boldsymbol{b^\top c})^2}\boldsymbol{b}(\boldsymbol{c^\top b})\boldsymbol{c}^\top = \frac{\boldsymbol{c^\top b}}{(\boldsymbol{b^\top c})^2}\boldsymbol{bc}^\top = \frac{\boldsymbol{bc}^\top}{\boldsymbol{b^\top c}} = \boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}.
$$



Indeed, $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$ is also a valid projection. So then what is the difference between $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$ and the orthogonal projection $\boldsymbol{P}_\boldsymbol{b}$ that we saw before? The difference lies in the fact that the _residuals_ are no longer orthogonal; that is, the angle between $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ and $\boldsymbol{a} - \text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ is no longer $90^\circ$. Let's check that this.



$$
\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})^\top (\boldsymbol{a} - \text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})) = \left(\frac{\boldsymbol{c^\top a}}{\boldsymbol{b^\top c}}\boldsymbol{b}\right)^\top\left(\boldsymbol{a} - \frac{\boldsymbol{c^\top a}}{\boldsymbol{b^\top c}}\boldsymbol{b}\right) = \frac{\boldsymbol{c^\top a}}{\boldsymbol{b^\top c}}\left(\boldsymbol{b^\top a} - \frac{\boldsymbol{c^\top a}}{\boldsymbol{b^\top c}}\boldsymbol{b^\top b}\right)
$$



This quantity will only be zero for any $\boldsymbol{a}$ if  $\boldsymbol{b^\top a} - \frac{\boldsymbol{c^\top a}}{\boldsymbol{b^\top c}}\boldsymbol{b^\top b} = 0$. This happens when $\boldsymbol{b}=\boldsymbol{c}$, in which case we return to get the orthogonal projection back, but for any other $\boldsymbol{c}$ we will not have that the residuals are orthogonal to $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$. Therefore, projections of this form are called _oblique_ projections. Let's see an example in $\mathbb{R}^2$.

Let's write a function to compute $\text{proj}_{b,c}$.

```{code-cell}
def oblique_projection(b, c):
    return np.outer(b,c)/np.dot(b,c)
```

Let's again project the vector $\boldsymbol{b}=\begin{bmatrix}1\\1\end{bmatrix}$ onto the vector $\boldsymbol{b}=\begin{bmatrix}1\\2\end{bmatrix}$, but this time along the direction $\boldsymbol{c} =\begin{bmatrix}1\\1/4\end{bmatrix}$. First, we'll compute $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$ using our oblique projection function.

```{code-cell}
c = np.array([1,0.25])

Pbc = oblique_projection(b,c)
```

Let's verify that $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}^2 = \boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$.

```{code-cell}
Pbc2 = np.dot(Pbc, Pbc)

np.allclose(Pbc2, Pbc)
```

So $\boldsymbol{P}_{\boldsymbol{b},\boldsymbol{c}}$ is indeed idempotent. Now let's visualize $\boldsymbol{a},\boldsymbol{b}, \text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ and $\boldsymbol{a}-\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$.

```{code-cell}
proj_bc_a = np.dot(Pbc, a) # compute the projection of a onto b
residual = a - proj_bc_a

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.quiver(*origin, *a, label='a', scale=1, units='xy', color='green')
plt.quiver(*origin, *proj_bc_a, label='proj_bc(a)', scale=1, units='xy', color='red')
plt.quiver(*proj_bc_a, *residual, label='a - proj_bc(a)', scale=1, units='xy', color='orange')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend(loc='upper left')
plt.show()
```

In this plot, we see that $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ is indeed parallel to $\boldsymbol{b}$, but it is not at a $90^\circ$ angle from the residual $\boldsymbol{a}-\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$. Let's actually compute the angle between $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ and $\boldsymbol{a}-\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ using techniques that we learned earlier in the chapter (recall that the angle between vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ is $\arccos\left(\frac{\boldsymbol{x^\top y}}{\|\boldsymbol{x}\|_2\|\boldsymbol{y}\|_2}\right)$).

```{code-cell}
temp = np.dot(proj_bc_a, residual)
residual_norm = np.linalg.norm(residual)
proj_bc_a_norm = np.linalg.norm(proj_bc_a)

angle = np.arccos(temp/(residual_norm*proj_bc_a_norm))
angle
```

Indeed, we get that the angle between $\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ and $\boldsymbol{a}-\text{proj}_{\boldsymbol{b},\boldsymbol{c}}(\boldsymbol{a})$ is approximately $2.43$ radians, which is roughly $140^\circ$.

## Projections onto a subspace

In the above sections, we phrased projections as projecting a vector $\boldsymbol{a}$ onto another _vector_ $\boldsymbol{b}$. In reality, what we computed was actually the projection of $\boldsymbol{a}$ onto the _subspace_ $V = \text{span}(\boldsymbol{b})$.

It turns out that there's nothing special about projecting onto a $1$-dimensional subspace: we can define orthogonal and oblique projections onto any subspace, as we will see below.

### Orthogonal projections onto a subspace

Let's begin with the concept of an orthogonal projection onto a subspace. Let $V$ be a subspace of $\mathbb{R}^n$, spanned by vectors $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$. Let's let $\boldsymbol{A}$ be the matrix whose columns are $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$, i.e.



$$
\boldsymbol{A} = \begin{bmatrix} | & | && |\\ \boldsymbol{a}_1 & \boldsymbol{a}_2 & \cdots & \boldsymbol{a}_k \\ | & | & & |\end{bmatrix}
$$



Let's see how we can derive the orthogonal projection onto $V$, which is just the column space of $\boldsymbol{A}$. Consider projecting a vector $\boldsymbol{b}$ onto $V$ -- call this projection $\hat{\boldsymbol{b}} = \text{proj}_V(\boldsymbol{b})$. Since $\hat{\boldsymbol{b}}$ should belong to the column space of $\boldsymbol{A}$, it should be of the form $\hat{\boldsymbol{b}} = \boldsymbol{A}\hat{\boldsymbol{x}}$ for some vector $\hat{\boldsymbol{x}}.$ For this projection to be orthogonal, we want that $\boldsymbol{b}- \hat{\boldsymbol{b}} =\boldsymbol{b}-\boldsymbol{A}\hat{\boldsymbol{x}}$ to be orthogonal to all the columns of $\boldsymbol{A}$. Earlier in the chapter, we saw that this means that $\boldsymbol{A}^\top(\boldsymbol{b} - \boldsymbol{A}\hat{\boldsymbol{x}}) = 0$. Then



$$
\boldsymbol{A^\top}(\boldsymbol{b}-\boldsymbol{A} \hat{\boldsymbol{x}}) = 0 \iff \boldsymbol{A^\top b} = \boldsymbol{A^\top A}\hat{\boldsymbol{x}} \iff \hat{\boldsymbol{x}} = (\boldsymbol{A^\top A})^{-1}\boldsymbol{A^\top b}
$$



Since $\hat{\boldsymbol{b}} = \boldsymbol{A}\hat{\boldsymbol{x}}$, we get



$$
\text{proj}_V(\boldsymbol{b}) = \hat{\boldsymbol{b}} = \boldsymbol{A}\hat{\boldsymbol{x}} = \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\boldsymbol{A}^\top \boldsymbol{b}
$$



This immediately gives us a formula for the projection matrix $\boldsymbol{P}_V$:



$$
\boldsymbol{P}_V = \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\boldsymbol{A}^\top
$$



This is an important formula that we will see again later in the semester. Let's check that $\boldsymbol{P}_V$ satisfies the idempotence condition $\boldsymbol{P}_V^2 = \boldsymbol{P}_V$. We have



$$
\boldsymbol{P}_V^2 = \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\underbrace{\boldsymbol{A^\top A}(\boldsymbol{A^\top A})^{-1}}_{\boldsymbol{I}}\boldsymbol{A}^\top = \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\boldsymbol{A}^\top = \boldsymbol{P}_V
$$



Indeed it does. Now let's look at an example numerically. Consider the matrix



$$
\boldsymbol{A} =\begin{bmatrix} \boldsymbol{a}_1 & \boldsymbol{a}_2\end{bmatrix} = \begin{bmatrix} 1 & 0\\ 1 & 1 \\ 0 &1\end{bmatrix}
$$



Where here $\boldsymbol{a}_1 = \begin{bmatrix}1\\1\\0\end{bmatrix}$ and $\boldsymbol{a}_2 = \begin{bmatrix}0\\ 1\\1\end{bmatrix}$. Let's compute the projection onto $V = \text{span}(\boldsymbol{a}_1,\boldsymbol{a}_2)$.



```{code-cell}
A = np.array([[1,0], [1,1], [0,1]])
ATA_inv = np.linalg.inv(np.dot(A.T, A)) # compute (A^TA)^{-1}
PV = np.dot(A, np.dot(ATA_inv, A.T))
```

Let's verify that this worked, by checking numerically that $P_V^2 = P_V$.

```{code-cell}
PV2 = np.dot(PV, PV)
np.allclose(PV2, PV)
```

Indeed it does.

Let's now verify that this projection is orthogonal to the column space of $\boldsymbol{A}$, by computing $\boldsymbol{A}^\top(\boldsymbol{b} - \text{proj}_V(\boldsymbol{b}))$. For this example, we'll use the vector $\boldsymbol{b} = \begin{bmatrix}1\\ 2\\ 3\end{bmatrix}$.

```{code-cell}
b = np.array([1,2,3])

proj_V_b = np.dot(PV, b)
residual = b - proj_V_b

np.dot(A.T, residual).round(8)
```

Indeed, we get the zeros vector, and so $\boldsymbol{b}-\text{proj}_V(\boldsymbol{b})$ is orthogonal to the columns of $\boldsymbol{A}$.

#### Relationship with the QR decomposition

In the previous workbook, we saw that we can write any matrix $\boldsymbol{A}$ as $\boldsymbol{A} = \boldsymbol{QR}$ where $\boldsymbol{Q}$ is an orthogonal matrix and $\boldsymbol{R}$ is upper triangular. Here, we'll see that we can write the projection onto the column space conveniently in terms of $\boldsymbol{Q}$. Let's plug in $\boldsymbol{A} = \boldsymbol{QR}$ into our formula for $\boldsymbol{P}_V$ (and recall that $\boldsymbol{Q^\top Q} = \boldsymbol{I}$).



$$
\boldsymbol{P}_V = \boldsymbol{QR}((\boldsymbol{QR})^\top \boldsymbol{QR})^{-1}(\boldsymbol{QR})^\top = \boldsymbol{QR}(\boldsymbol{R}^\top \underbrace{\boldsymbol{Q^\top Q}}_{\boldsymbol{I}} \boldsymbol{R})^{-1}\boldsymbol{R^\top Q}^\top = \boldsymbol{QR}(\boldsymbol{R^\top R})^{-1}\boldsymbol{R^\top Q}^\top = \boldsymbol{Q}\underbrace{\boldsymbol{RR}^{-1}}_{\boldsymbol{I}}\underbrace{(\boldsymbol{R}^\top)^{-1}\boldsymbol{R}^\top}_{\boldsymbol{I}} \boldsymbol{Q}^\top = \boldsymbol{QQ}^\top
$$



Therefore, if we have the QR decomposition of $\boldsymbol{A}$, the projection onto the column space of $\boldsymbol{A}$ can be easily computed with $\boldsymbol{QQ}^\top$. This is convenient as it doesn't require taking any matrix inverses, which can be difficult to work with numerically.

**Remark:** Recall that we always have that $\boldsymbol{Q^\top Q} = \boldsymbol{I}$ for an orthogonal matrix $\boldsymbol{Q}$. Here we see clearly that $\boldsymbol{QQ}^\top$ is emphatically _not_ equal to the identity in general.

Let's use this method to compute the projection $\boldsymbol{P}_V$ using the same matrix $\boldsymbol{A}$ as above. Here we use the built-in numpy function for the QR decomposition, but we could just as well have used the QR function that we wrote ourselves in the previous workbook.

```{code-cell}
Q, R = np.linalg.qr(A)
QQT = np.dot(Q, Q.T)
np.allclose(QQT, PV)
```

Indeed, the two approaches give us the same answer.

The last point we make before moving on is that there are _many_ possible matrices $\boldsymbol{A}$ whose columns span a given subspace $V$. For example, the matrix



$$
\boldsymbol{B} = \begin{bmatrix} -2 & 0\\ -2 & 4 \\ 0 &4\end{bmatrix}
$$



has the same column space as $\boldsymbol{A}$. Let's check that computing the projection using this matrix gives us the same result.

```{code-cell}
B = np.array([[-2, 0], [-2, 4], [0, 4]])

Q2, R2 = np.linalg.qr(B)
QQT2 = np.dot(Q2, Q2.T)
np.allclose(QQT, QQT2)
```

Indeed, the projection onto $V$ is the same no matter which spanning vectors we use.

### Oblique projections onto a subspace

Like in the case of projecting onto vectors, we can also have _oblique_ projections onto the column space of a matrix $\boldsymbol{A}$, which is the subspace $V$. Let $\boldsymbol{C}$ be any $n\times k$ matrix such that $\boldsymbol{C^\top A}$ is invertible. Then the matrix



$$
P_{V,\boldsymbol{C}} = \boldsymbol{A}(\boldsymbol{C^\top A})^{-1}\boldsymbol{C}^\top
$$



is always a projection onto $V$. It will of course reduce to the orthogonal projection when $\boldsymbol{C} = \boldsymbol{A}$, in which case we obtain the same  formula that we had before. To check that it is indeed a projection, we need to verify that $\boldsymbol{P}_{V,\boldsymbol{C}}^2 = \boldsymbol{P}_{V,\boldsymbol{C}}$. We calculate



$$
\boldsymbol{P}_{V,\boldsymbol{C}}^2 = \boldsymbol{A}(\boldsymbol{C^\top A})^{-1}\underbrace{\boldsymbol{C^\top A}(\boldsymbol{C^\top A})^{-1}}_{\boldsymbol{I}}\boldsymbol{C}^\top = \boldsymbol{A}(\boldsymbol{C^\top A})^{-1}\boldsymbol{C}^\top = \boldsymbol{P}_{V,\boldsymbol{C}}
$$


So $\boldsymbol{P}_{V,\boldsymbol{C}}$ is in fact a valid projection. Let's first look at an example with the matrix $\boldsymbol{A} = \begin{bmatrix} 1 & 0\\ 1 & 1 \\ 0 &1\end{bmatrix}$ that we used above. There are many valid examples of matrices $\boldsymbol{C}$ that we can use to define an oblique projection $\boldsymbol{P}_{V,\boldsymbol{C}}$; indeed, for most matrices $\boldsymbol{C}$ we will have that $\boldsymbol{C^\top A}$ is invertible. Let's try choosing $\boldsymbol{C}$ to be a random matrix.

```{code-cell}
k = 2
n = 3

C = np.random.normal(size = (n,k))
```

Let's check that $\boldsymbol{C^\top A}$ is invertible.

```{code-cell}
CTA_inv = np.linalg.inv(np.dot(C.T, A))
```

Indeed, computing the inverse works without error.

Now, let's use this to compute $\boldsymbol{P}_{V,\boldsymbol{C}}$.

```{code-cell}
PVC = np.dot(A, np.dot(CTA_inv, C.T))
```

We can check numerically that $\boldsymbol{P}_{V,\boldsymbol{C}}^2 = \boldsymbol{P}_{V,\boldsymbol{C}}$:

```{code-cell}
PVC2 = np.dot(PVC, PVC)
np.allclose(PVC2, PVC)
```

So $\boldsymbol{P}_{V,\boldsymbol{C}}$ is in fact idempotent, and thus a valid projection. However, it is not orthogonal; we can check this by computing $\boldsymbol{A}^\top (\boldsymbol{b} - \boldsymbol{P}_{V,\boldsymbol{C}}\boldsymbol{b})$, and verifying that it is not equal to zero (as it was in the orthogonal case). Let's do this for the same vector $\boldsymbol{b} = \begin{bmatrix}1\\ 2\\ 3\end{bmatrix}$ that we used before.

```{code-cell}
proj_VC_b = np.dot(PVC, b)
residuals = b - proj_VC_b
np.dot(A.T, residuals)
```

 Our answer is clearly not zero, and so the projection $\boldsymbol{P}_{V,\boldsymbol{C}}$ is _not_ an orthogonal projection, but rather an _oblique_ projection.

## Projecting onto the orthogonal complement of a subspace

The last type of projection we will discuss is the projection onto the _orthogonal complement_ of a subspace $V\subseteq \mathbb{R}^n$. The orthogonal complement is the subspace $V^\perp$ which is defined as follows:


$$
V^\perp = \{\boldsymbol{w}\in \mathbb{R}^n : \boldsymbol{w^\top v} = 0\text{ for all } \boldsymbol{v}\in V\}
$$


That is, the orthogonal complement of $V$ is the set of all vectors which are orthogonal to all vectors in $V$. It turns out that the projection onto the orthogonal complement is easy to find given the orthogonal projection onto $V$. If $\boldsymbol{P}_V$ is the orthogonal projection onto $V$, then the orthogonal projection onto $V^\perp$ is just


$$
\boldsymbol{P}_{V^\perp} = \boldsymbol{I} - \boldsymbol{P}_V
$$


Given $\boldsymbol{P}_V = \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\boldsymbol{A}^\top$ or $\boldsymbol{P}_V = \boldsymbol{QQ}^\top$ (where $\boldsymbol{Q}$ comes from the QR factorization of $\boldsymbol{A}$), this means $\boldsymbol{P}_{V^\perp} = \boldsymbol{I}- \boldsymbol{A}(\boldsymbol{A^\top A})^{-1}\boldsymbol{A}^\top$ or $\boldsymbol{P}_{V^\perp} = \boldsymbol{I}- \boldsymbol{QQ}^\top$. Since the range of $\boldsymbol{P}_V$ is $V$, and the range of $\boldsymbol{P}_{V^\perp}$ is $V^\perp$, we should always have that $\boldsymbol{P}_V \boldsymbol{x}$ is orthogonal to $\boldsymbol{P}_{V^\perp}\boldsymbol{y}$ for any vectors $\boldsymbol{x}$ and $\boldsymbol{y}$. Let's check that this is in fact true. We have


$$
(\boldsymbol{P}_{V^\perp}\boldsymbol{y})^\top \boldsymbol{P}_{V}\boldsymbol{x} = \boldsymbol{y}^\top \boldsymbol{P}_{V^\perp}\boldsymbol{P}_V\boldsymbol{x} = \boldsymbol{y}^\top (\boldsymbol{I}-\boldsymbol{P}_V)\boldsymbol{P}_V\boldsymbol{x} = \boldsymbol{y}^\top \boldsymbol{P}_V\boldsymbol{x} - \boldsymbol{y}^\top \boldsymbol{P}_V^2\boldsymbol{x} = \boldsymbol{y}^\top \boldsymbol{P}_V\boldsymbol{x} - \boldsymbol{y}^\top \boldsymbol{P}_V \boldsymbol{x} = 0
$$


where we used the fact that $\boldsymbol{P}_V$ is a projection, so $\boldsymbol{P}_V^2 = \boldsymbol{P}_V$.
