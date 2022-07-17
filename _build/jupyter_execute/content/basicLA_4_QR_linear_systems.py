#!/usr/bin/env python
# coding: utf-8

# # Solving linear systems with the QR decomposition
# 
# One of the most important applications of the QR decomposition is in solving (or approximately solving) a linear system of equations of the form $\boldsymbol{Ax} = \boldsymbol{b}$ where $\boldsymbol{A}\in \mathbb{R}^{m\times n}$ and $\boldsymbol{b} \in \mathbb{R}^m$ are known. This problem comes up frequently in data science problems, perhaps most notably in the context of linear regression.
# 
# Before we see how the QR decomposition can help us in this task, let's first study when we can expect to have a solution $\boldsymbol{x}_\star$ satisfying $\boldsymbol{Ax}_\star = \boldsymbol{b}$. Recall that if we write $\boldsymbol{A} = \begin{bmatrix}\boldsymbol{a}_1\cdots \boldsymbol{a}_m\end{bmatrix}$ in terms of its columns $\boldsymbol{a}_1,\dots,\boldsymbol{a}_n\in \mathbb{R}^m$, then $\boldsymbol{Ax} = \boldsymbol{b}$ can be expressed as
# 
# $$
# x_1\boldsymbol{a}_1 + \cdots + x_n \boldsymbol{a}_n = \boldsymbol{b}.
# $$
# 
# The question then is whether there are coefficients $\boldsymbol{x} = (x_1,\dots,x_n)$ that satisfy this equation. This will be true whenever $\boldsymbol{b}$ is the the _column space_ of $\boldsymbol{A}$ (i.e. the span of the columns of $\boldsymbol{A}$). This will be true for any vector $\boldsymbol{b}\in \mathbb{R}^m$ provided the columns of $\boldsymbol{A}$ span all of $\mathbb{R}^m$, equivalently if function $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is surjective, though it may also be true for a _particular_ vector $\boldsymbol{b}$ even if $f$ is not surjective. In fact, if the columns of $\boldsymbol{A}$ span all of $\mathbb{R}^m$, there may in fact be many solutions $\boldsymbol{x}_\star$ which satisfy $\boldsymbol{Ax}_\star = \boldsymbol{b}$. On the other hand, if $f(\boldsymbol{x}) = \boldsymbol{Ax}$ is not surjective, then there may be no solutions; in this case we are often satisfied with a approximate solution. This will lead us to the topic of _least squares_, which we will discuss later in the class. For now, we will consider situations when the system $\boldsymbol{Ax} = \boldsymbol{b}$ _does_ have a solution, either for all vectors $\boldsymbol{b}$ or only for particular vectors $\boldsymbol{b}$. In particular, we will focus here on when the matrix $\boldsymbol{A}$ is square and invertible, so that $\boldsymbol{Ax} = \boldsymbol{b}$ has a _unique_ solution given by $\boldsymbol{x}_\star = \boldsymbol{A}^{-1}\boldsymbol{b}$. However, computing a matrix inverse can often be expensive when done directly, and in many cases we don't actually need the matrix $\boldsymbol{A}^{-1}$, but rather just the solution $\boldsymbol{x}_\star$. In this section, we will see how we can use the QR decomposition to find this solution efficiently.
# 
# ## Solving upper triangular systems with backsubstition
# 
# Solving the system $\boldsymbol{Ax} = \boldsymbol{b}$, for an $n\times n$ matrix $\boldsymbol{A}$, involves solving a system of $n$ equations in $n$ variables, which in general can be a computationally expensive problem. The problem is simplified considerably when the system is of the form $\boldsymbol{Rx} = \boldsymbol{b}$ for some upper triangular matrix $\boldsymbol{R}$ of the form
# 
# $$
# \boldsymbol{R} = \begin{bmatrix} r_{1,1} & r_{1,2} & \cdots & r_{1,n}\\ 0 & r_{2,2} & \cdots & r_{2,n}\\ \vdots & \vdots &\ddots & \vdots\\ 0 & 0 & \cdots & r_{n,n}  \end{bmatrix}.
# $$
# 
# Then the system $\boldsymbol{Rx} = \boldsymbol{b}$ looks like
# 
# $$
# \begin{align}
# r_{1,1}x_1 + r_{12}x_2 + \cdots + r_{1,n}x_n &= b_1 && (1) \\
# r_{2,2}x_2 + \cdots + r_{2,n}x_n &= b_2 && (2)\\
# & \vdots  \\
# r_{n,n}x_n &= b_n && (n)
# \end{align}
# $$
# 
# This system is easy to solve using the following algorithm: first solve for $x_n$ in equation $(n)$ with $x_n = b_n/r_{nn}$, then plug this into equation $(n-1)$ to solve for $x_{n-1}$, and so on. Because the system is triangular, we can easily find another entry in the solution vector $\boldsymbol{x}_\star$ at each step, rather than needing to solve all $n$ equations simultaneously. This general algorithm is called _backsubstition_, and is very efficient in practice.
# 
# The general algorithm works as follows:
# 
# $$
# \begin{align}
# &\underline{\textbf{backsubstition algorithm}: \text{find vector $\boldsymbol{x}$ satisfying $\boldsymbol{Rx} = \boldsymbol{b}$.}} \\
# &\textbf{input}:\text{upper triangular matrix }\boldsymbol{R}\in \mathbb{R}^{n\times n}\text{, vector }\boldsymbol{b}=(b_1,\dots,b_n)\in \mathbb{R}^n  \\
# &\hspace{0mm}\text{initialize $\boldsymbol{x} = (0,\dots,0)$}\\
# &\hspace{0mm} \text{for $i=n,\dots, 1$:}\\
# &\hspace{10mm} t = b_i\\
# &\hspace{10mm} \text{for $j > i$:}\\
# &\hspace{20mm} t = t - \boldsymbol{R}_{ij}\cdot x_j\\
# &\hspace{10mm} x_i = t/\boldsymbol{R}_{ii}\\
# &\hspace{0mm} \text{return } \boldsymbol{x} = (x_1,\dots,x_n)\\
# \end{align}
# $$
# 
# If you write out an example of what the above algorithm does, you will see that it performs exactly the procedure we described above. We can easily implement this in a few lines of python:

# In[1]:


def back_substitution(R, b):
    n = len(b)
    x = np.zeros(n)

    # note: using -1 as the "by" argument counts backwards in steps of 1
    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(n-1, i, -1):
            tmp -= x[j]*R[i,j]

        x[i] = tmp/R[i,i]
    return x


# Let's use this function to solve some triangular systems. For example, consider the triangular matrix
# 
# $$
# \boldsymbol{R} = \begin{bmatrix} 1 & -2 & 4 & 5\\ 0 & 2 & 3 & -1\\ 0 & 0 & -6 & 1\\ 0& 0 &0 &2 \end{bmatrix}
# $$
# 
# and the vector $\boldsymbol{b} = (1,1,1,1)$. We can find a vector $\boldsymbol{x}$ satisfying $\boldsymbol{Rx} = \boldsymbol{b}$ with the following python code:

# In[2]:


import numpy as np
R = np.array([[1,-2,4,5], [0,2,3,-1], [0,0,-6,1], [0,0,0,2]])
b = np.ones(4)
x = back_substitution(R,b)
print(x)


# Let's check that this vector actually solves our system.

# In[3]:


print(np.dot(R,x).round(2))


# ## Using QR to solve linear systems in the general case
# 
#  Now that we've seen how to easily solve triangular systems, let's see why QR can be useful in the general case of solving $\boldsymbol{Ax} = \boldsymbol{b}$ when $\boldsymbol{A}$ is _not_ triangular. Recall that the QR decomposition allows us to write $\boldsymbol{A} = \boldsymbol{QR}$ where $\boldsymbol{Q}$ is an orthogonal matrix, obtained using the Gram-Schmidt procedure, and $\boldsymbol{R}$ is an upper triangular matrix. Now if we plug this decomposition into the equation $\boldsymbol{Ax} = \boldsymbol{b}$, we find
# 
# $$
# \boldsymbol{Ax} = \boldsymbol{b} \iff \boldsymbol{QRx} = \boldsymbol{b} \iff \underbrace{\boldsymbol{Q^\top Q}}_{=\boldsymbol{I}} \boldsymbol{Rx} = \boldsymbol{Q^\top b} \iff \boldsymbol{Rx} = \boldsymbol{Q^\top b}.
# $$
# 
# Thus if we define $\tilde{\boldsymbol{b}} = \boldsymbol{Q}^\top \boldsymbol{b}$, we see that we can equivalently find $\boldsymbol{x}$ by solving the triangular system
# 
# $$
# \boldsymbol{Rx} = \tilde{\boldsymbol{b}}.
# $$
# 
# Now we have arrived at a simple two-step procedure for solving an $n\times n$ system $\boldsymbol{Ax} = \boldsymbol{b}$:
# 
# 1. Factor $\boldsymbol{A} = \boldsymbol{QR}$ using the QR decomposition
# 2. Solve the triangular system $\boldsymbol{Rx} = \boldsymbol{Q^\top b}$ using backsubstitution
# 
# Let's see an example how we can use this method.
# 
# Suppose we have $n$ data samples $\boldsymbol{x}_1,\dots, \boldsymbol{x}_n \in \mathbb{R}^n$, each associated with a response, or label, generated via the following linear relationship:
# 
# $$
# y_i = \boldsymbol{\beta}_\star^\top \boldsymbol{x}_i
# $$
# 
# where $\boldsymbol{\beta}_\star$ is an unobserved vector in $\mathbb{R}^n$. We want to use the data $(\boldsymbol{x}_1,y_1),\dots,(\boldsymbol{x}_n, y_n)$ to determine what the vector $\boldsymbol{\beta}_\star$ is. If we collect the features $\boldsymbol{x}_i$ into the columns a matrix $\boldsymbol{X} = \begin{bmatrix}\boldsymbol{x}_1 & \cdots & \boldsymbol{x}_n\end{bmatrix}$, and the responses into a vector $\boldsymbol{y} = (y_1,\dots,y_n)$, we can equivalently pose this as finding a vector $\boldsymbol{\beta}$ that solves the linear system
# 
# $$
# \boldsymbol{X\beta} = \boldsymbol{y}.
# $$
# 
# Let's generate an example of this setup, and use the method described above to find the solution. First let's generate some data.

# In[4]:


n= 20

beta_star = np.array([1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8,9,-9,10,-10])
X = np.random.normal(size=(n,n))
y = np.dot(X, beta_star)


# Next, let's compute the QR decomposition of the matrix $\boldsymbol{X}$. We could do this ourselves using the functions we wrote in the previous section, but here we will simply use the built-in `numpy` function.

# In[5]:


Q, R = np.linalg.qr(X)


# Now, we can use our function `back_substitution` to try and recover $\boldsymbol{\beta}_\star$ from the data. First, we form $\tilde{\boldsymbol{y}} = \boldsymbol{Q^\top y}$, and then solve $\boldsymbol{R\beta} = \tilde{\boldsymbol{y}}$.

# In[6]:


y_tilde = np.dot(Q.T, y)
beta_hat = back_substitution(R, y_tilde)
print(beta_hat)


# The problem we have just solved here is a special case of the linear regression problem. Later in this course, we will see that essentially the same techniques can be used to solve more general versions of this problem, and in fact are the basis for how most common statistical software do so.
