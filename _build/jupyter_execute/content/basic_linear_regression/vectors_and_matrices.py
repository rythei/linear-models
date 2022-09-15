#!/usr/bin/env python
# coding: utf-8

# # Basic concepts from linear algebra: vectors and matrices
# 
# ## Motivation: linear regression with multiple predictor variables
# 
# Now that we have introduced the simple linear regression model, a natural extension will be to consider multiple predictor variables in a model of the form
# 
# $$
# y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i, \hspace{10mm} (1)
# $$
# 
# where each observation $i$ is associated with $p$ different features $x_{i1},\dots, x_{ip}$. We will discuss such models extensively throughout the remainder of the course, but in order to do so effectively, we must first review some basic mathematical tools from linear algebra which will be useful in our study.
# 
# ## Vectors and dot products
# 
# Central to the study of linear algebra are the concepts of _vectors_ and _vector spaces_. In general, a vector space is a set $V$ such that for any two vectors $u,v \in V$, we can add the vectors and get a new element $u+v \in V$, and we can multiply vectors by a scalar $\alpha \in \mathbb{R}$ and again get another vector $\alpha v \in V$.
# 
# In this course, we will primarily be interested in the most common vector space, $\mathbb{R}^n$, which is simply the set of all $n$-tuples of real numbers. We can denote a vector $\boldsymbol{x} \in \mathbb{R}^n$ using the notation
# 
# $$
# \boldsymbol{x} = \begin{bmatrix} x_1\\ x_2\\\vdots \\ x_n \end{bmatrix}.
# $$
# 
# For example, $\begin{bmatrix} 1\\ 2\end{bmatrix}$ is a vector in $\mathbb{R}^2$, which we can simply think of as a point in the Euclidean plane. In python, we typically will represent vectors using an `array` object from the numpy package. An array can be defined using the following:

# In[1]:


# first we have to import the numpy package
import numpy as np

# define the vector (1,2) as a numpy array
x = np.array([1,2])
x


# For vectors in $\mathbb{R}^n$, we can define vector addition in a simple way: given two vectors
# 
# $$
# \boldsymbol{u} = \begin{bmatrix} u_1\\ \vdots \\ u_n\end{bmatrix},\;\;\; \boldsymbol{v} = \begin{bmatrix}v_1\\ \vdots \\ v_n\end{bmatrix},
# $$
# 
# we can add them as follows:
# 
# $$
# \boldsymbol{u} + \boldsymbol{u} = \begin{bmatrix} u_1\\ \vdots \\ u_n\end{bmatrix} + \begin{bmatrix}v_1\\ \vdots \\ v_n\end{bmatrix} = \begin{bmatrix}u_1+ v_1 \\ \vdots \\ u_n + v_n\end{bmatrix}.
# $$
# 
# That is, when we add two vectors we just add their corresponding entries. This will of course give us another vector in $\mathbb{R}^n$ back. Adding two vectors is also easy using arrays in python. Let's define two arrays `u` and `v`:

# In[2]:


u = np.array([1,2,3])
v = np.array([4,5,6])


# Now we can add them using the usual `+` operation.

# In[3]:


u_plus_v = u+v
print(f"u = {u}")
print(f"v = {v}")
print(f"u+v = {u_plus_v}")


# We see that this gives us the expected result.
# 
# We can also perform scalar multiplication with vectors. For a vector $\boldsymbol{v} \in \mathbb{R}^n$ and a scalar $\alpha \in \mathbb{R}$, we can define
# 
# $$
# \alpha \boldsymbol{v} = \alpha \begin{bmatrix}v_1\\ \vdots \\ v_n\end{bmatrix} = \begin{bmatrix}\alpha v_1\\ \vdots \\ \alpha v_n\end{bmatrix}.
# $$
# 
# That is, $\alpha \boldsymbol{v}$ just means multiplying each entry of $\boldsymbol{v}$ by $\alpha$. This is similarly each to do in python with arrays. For example,

# In[4]:


print(f"v = {v}")
print(f"-1v = {-1*v}")
print(f"2v = {2*v}")


# These again give the expected results.
# 
# ### The dot product
# 
# There is one more operation on vectors that will be very important to us called the _dot product_ or _inner product_. Given two vectors $\boldsymbol{u},\boldsymbol{v} \in \mathbb{R}^n$, their dot product is
# 
# $$
# \langle \boldsymbol{u},\boldsymbol{v}\rangle = \boldsymbol{u}\cdot \boldsymbol{v} = \sum_{i=1}^n u_iv_i.
# $$
# 
# This returns a single real _number_, which is the sum $\sum_{i=1}^n u_iv_i$, i.e. multiplying and summing up the entries of the two vectors pairwise.
# 
# > Remark: we use two notations for the dot product here, $\langle \boldsymbol{u},\boldsymbol{v}\rangle$ and $\boldsymbol{u}\cdot \boldsymbol{v}$ both will be seen commonly, and can be used interchangeably. In fact, the dot product between two vectors is also sometimes written as $\boldsymbol{u}^\top \boldsymbol{v}$. This form will make more sense shortly when we discuss matrices.
# 
# In numpy, we can compute the dot product of two vectors using the function `np.dot`. For example,

# In[5]:


x = np.array([1,2,3])
y = np.array([4,5,6])
x_dot_y = np.dot(x,y) # computes 1*4 + 2*5 + 3*6 = 32
print(x_dot_y)


# ### Vector norms
# 
# Another important operation that we can do with vectors is to compute their _norm_. A norm is a function that measures the "size" of something.
# One example of a norm is the familiar Euclidean norm, which uses the expression $(x_1^2+x_2^2)^{1/2}$ to compute the "size" or "magnitude" of a point $\begin{bmatrix}x_1 \\ x_2\end{bmatrix}$ in the two-dimensional Euclidean plane. If we view that point as a vector, then that is the Euclidean norm of the vector.
# 
# The generalization of this notion of length to $n$ dimensions gives us the Euclidean norm, which is the most important vector norm used in practice. For a vector
# 
# $$
# \boldsymbol{x} = \begin{bmatrix}x_1\\\vdots\\ x_n\end{bmatrix},
# $$
# 
# it's Euclidean norm can be computed as
# 
# $$
# \|\boldsymbol{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.
# $$
# 
# > Remark: we use the notation $\|\cdot\|_2$ to indicate the Euclidean norm as it is often also referred to as the "2-norm". This is because it can be viewed as part of a family of norms called the $p$-norms. For $p\geq 1$, the $p$-norm of a vector $\boldsymbol{x}$ is given by $(\sum_{i=1}^n |x_i|^p)^{1/p}$.
# 
# Note that, importantly, the Euclidean norm squared is simply the dot product of $\boldsymbol{x}$ with itself, since
# 
# $$
# \|\boldsymbol{x}\|_2^2 = \sum_{i=1}^n x_i^2 = \boldsymbol{x}\cdot \boldsymbol{x}.
# $$
# 
# This would give us one way to compute the Euclidean norm in python, by using numpy's `dot` function again:

# In[6]:


norm_x_v1 = np.sqrt(np.dot(x,x))
norm_x_v1


# However, we can also use numpy's built in function for computing norms as follows:

# In[7]:


norm_x_v2 = np.linalg.norm(x, ord=2)
norm_x_v2


# Note that here we specify `ord=2` to make sure python knows we are referring to the 2-norm (however this is also the default, so we don't technically need to specify it).
# 
# Norms also give us a way to measure the distance between two vectors, by considering
# 
# $$
# \|\boldsymbol{x}-\boldsymbol{y}\|_2
# $$
# 
# for two vectors $\boldsymbol{x}, \boldsymbol{y}$.
# 
# Another norm that we will encounter in this class is the 1-norm, which is simply the sum of the abolute values of the entries in a vector:
# 
# $$
# \|\boldsymbol{x}\|_1 = \sum_{i=1}^n |x_i|.
# $$
# 
# This can similary be computed using numpy's norm function:

# In[8]:


norm1_x = np.linalg.norm(x, ord=1)
norm1_x


# ## Matrices and multiplication
# 
# One way to define a matrix is as follows: an $m\times n$ matrix $\boldsymbol{A} \in \mathbb{R}^{m\times n}$, is an array of real numbers consisting of $m$ rows and $n$ columns. For example, the following is a $2\times 3$ matrix
# 
# $$
# \boldsymbol{A} = \begin{bmatrix} 1 & 2 & 3\\ 4 & 5 & 6\end{bmatrix}.
# $$
# 
# This matrix can also be defined as a numpy array in python:

# In[9]:


import numpy as np

A = np.array([[1,2,3], [4,5,6]])
print(A)


# We can also think of a matrix a collection of vectors in two different ways: first, we can think of it as containing _row vectors_, $\begin{bmatrix} 1 & 2 & 3\end{bmatrix}$ and $\begin{bmatrix} 4&5&6\end{bmatrix}$. Alternatively, we can think of $\boldsymbol{A}$ as a collection of _column vectors_ $\begin{bmatrix}1\\ 4\end{bmatrix}, \begin{bmatrix}2\\ 5\end{bmatrix}, \begin{bmatrix}3\\ 6\end{bmatrix}$. Sometimes, we will use the notation
# 
# $$
# \begin{bmatrix} \boldsymbol{x}_1\\ \vdots \\ \boldsymbol{x}_m \end{bmatrix} \;\;\;\; \text{or}  \;\;\;\; \begin{bmatrix} \boldsymbol{x}_1 & \cdots & \boldsymbol{x}_n \end{bmatrix}
# $$
# 
# to denote a $m\times n$ matrix in terms of its row vectors or column vectors.
# 
# Note that by this definition of a matrix, a vector is simply a special case of a matrix with either just one column or one row. By convention, we usually think of a vector $\boldsymbol{x}\in \mathbb{R}^n$ as being a _column vector_, with $n$ rows and $1$ column, so that $\boldsymbol{x}$ is really a $n\times 1$ matrix.
# 
# In numpy, we can specify a vector as being a column vector by suitably reshaping it.

# In[10]:


n = 5
x = np.random.normal(size=n) # generate a random vector of dimension n
print(x.shape) # defaults to shape (n,)
x = x.reshape(n,1)
print(x.shape) # explicitly making x a column vector


# Note that by default, numpy stores 1-d arrays as having shape `(n,)`, which is, somewhat subtly, different from a column vector, which has shape `(n,1)`. So to work with a column vector in Python, we have to explictly specify its shape. For many operations we will want to perform, this distinction won't matter much, though for some operations this distinction is in fact important, and so we will want to be careful. We will see examples of this in the coming sections. We can also represent a vector explicitly as a row vector in a similar way.

# In[11]:


x = x.reshape(1,n)
print(x.shape) # explicitly making x a row vector


# Suppose we were given an $m\times n$ matrix $\boldsymbol{A}$ of the form
# 
# $$
# \boldsymbol{A} = \begin{bmatrix}a_{11}& \cdots &a_{1n}\\ a_{21}&\cdots & a_{2n}\\ \vdots & \ddots & \vdots \\ a_{m1}&\cdots & a_{mn}\end{bmatrix} \in \mathbb{R}^{m\times n}.
# $$
# 
# ### The transpose operation
# 
# One of the most important operations we can perform on such a matrix is to take its _transpose_, which means to form the $n\times m$ matrix $\boldsymbol{A}^\top$ by defining the $i^{th}$ row of $\boldsymbol{A}^\top$ be the $i^{th}$ column of $\boldsymbol{A}$. Specifically, this would give us
# 
# $$
# \boldsymbol{A}^\top = \begin{bmatrix}a_{11}& \cdots &a_{m1}\\ a_{12}&\cdots & a_{m2}\\ \vdots & \ddots & \vdots \\ a_{1n}&\cdots & a_{mn}\end{bmatrix} \in \mathbb{R}^{n\times m}.
# $$
# 
# Note that this operation takes a matrix of shape $m\times n$ and returns a matrix of shape $n\times m$. It is easy to find the transpose of a matrix (i.e. numpy array) in Python:

# In[12]:


print(A.shape)
AT = A.T # take the transpose of A
print(AT.shape)


# We can also use this to convert between row and column vectors in numpy.

# In[13]:


x = np.random.normal(size=n)
x = x.reshape(n,1)
print(x.shape) #column vector
xT = x.T
print(xT.shape) #row vector


# ### Matrix multiplcation
# 
# The second operation on matrices which will we frequently encounter is matrix multiplication.
# 
# Matrix multiplication is really a generalization of the dot product we defined earlier. Given matrices $\boldsymbol{A}\in \mathbb{R}^{m\times n}$, with rows $\boldsymbol{a}_{1:},\dots,\boldsymbol{a}_{m:}$, and  $\boldsymbol{B}\in \mathbb{R}^{n\times p}$, with columns $\boldsymbol{b}_{:1},\dots, \boldsymbol{b}_{:p}$, we define the matrix product $\boldsymbol{AB}$ to be the $m\times p$ matrix whose $(i,j)^{th}$ entry is
# 
# $$
# [\boldsymbol{A}\boldsymbol{B}]_{ij} = \boldsymbol{a}_{i:} \cdot \boldsymbol{b}_{:j}.
# $$
# 
# That is, the $(i,j)^{th}$ entry of the matrix $\boldsymbol{AB}$ is the dot product of the $i^{th}$ row of $\boldsymbol{A}$ with the $j^{th}$ column of $\boldsymbol{B}$.
# 
# Note that for this operation to be well-defined, we need that the rows of $\boldsymbol{A}$ are of the same dimension as the columns of $\boldsymbol{B}$, or equivalently that the number of columns of $\boldsymbol{A}$ is equal to the number of rows of $\boldsymbol{B}$. Let's see some examples in Python. Note that we can also use the numpy function `np.dot` to perform matrix multiplication.

# In[14]:


m, n, p = 10,5,3

A = np.random.normal(size=(m,n))
B = np.random.normal(size=(n,p))
AB = np.dot(A,B)
print(AB.shape)


# This is an example where the matrix product is well-defined, since the number of columns of $\boldsymbol{A}$ (5) is equal to the number of rows of $\boldsymbol{B}$ (also 5). Let's see an example where this doesn't work.

# In[15]:


# now the inner dimensions don't match
m, n, k, p = 10,5,4, 3

A = np.random.normal(size=(m,n))
B = np.random.normal(size=(k,p))
AB = np.dot(A,B)
print(AB.shape)


# As we'd expect, numpy gives us an error, because the two matrices are not of coherent dimensions to perform matrix multiplcation.
# 
# An important special case of matrix multiplication is when the matrix on the right only has a single column, so is really a vector. This gives us matrix-vector multiplication, which is performed as follows, for a $m\times n$ matrix $\boldsymbol{A}$ and $n$-dimensional vector $\boldsymbol{x}$:
# 
# $$
# \boldsymbol{Ax} = \begin{bmatrix}a_{11} & \cdots & a_{1n}\\ \vdots & \ddots & \vdots \\ a_{m1} & \cdots & a_{mn}\end{bmatrix}\begin{bmatrix}x_1\\ \vdots \\ x_n\end{bmatrix} = \begin{bmatrix}\boldsymbol{a}_{1:}\cdot \boldsymbol{x}\\ \vdots \\ \boldsymbol{a}_{m:}\cdot \boldsymbol{x}\end{bmatrix}
# $$
# 
# Note that $\boldsymbol{Ax}$ gives us an $m$-dimensional vector back. A useful fact about matrix-vector multiplication is that it can be represented as a linear combination of the columns of $\boldsymbol{A}$, i.e.
# 
# $$
# \boldsymbol{Ax} = x_1 \boldsymbol{a}_{:1} + \cdots + x_n \boldsymbol{a}_{:n}.
# $$
# 
# 
# ## Returning to the multiple linear regression model
# 
# To see why linear algebra is so closely related to the study of linear models of the form $(1)$, let us define a few special vectors and matrices. Given $p$ predictor variable $x_{i1},\dots, x_{ip}$, define the vector
# 
# $$
# \boldsymbol{x}_i = \begin{bmatrix}1\\ x_{i1}\\ x_{i2}\\ \vdots\\ x_{ip}\end{bmatrix}
# $$
# 
# as well as the vector of coefficients
# 
# $$
# \boldsymbol{\beta} = \begin{bmatrix}\beta_0 \\ \beta_1\\ \beta_2 \\ \vdots\\ \beta_p \end{bmatrix}.
# $$
# 
# Then let's see what happens when we take the dot product of $\boldsymbol{x}_i$ with $\boldsymbol{\beta}$. By definition, this is
# 
# $$
# \boldsymbol{\beta}\cdot \boldsymbol{x}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip}
# $$
# 
# This looks exactly the same as the right-hand side of equation (1) (only without the error term $\varepsilon$). Note that we cleverly added a 1 to the first entry of the vector $\boldsymbol{x}_i$ so that it would match up with the intercept term $\beta_0$ and give us a constant. This means that we can succinctly represent the regression model (1) in vector form as
# 
# $$
# y_i = \boldsymbol{\beta}\cdot \boldsymbol{x}_i + \varepsilon_i.
# $$
# 
# Furthermore, we can stack the vectors $\boldsymbol{x}_1\dots,\boldsymbol{x}_n$ for our $n$ observations as the row vectors of an $n\times (p+1)$ matrix $\boldsymbol{X}$ as follows:
# 
# $$
# \boldsymbol{X} = \begin{bmatrix} \boldsymbol{x}_1 \\ \boldsymbol{x}_2\\ \vdots\\ \boldsymbol{x}_n\end{bmatrix}
# $$
# 
# Also, we can define the vectors
# 
# $$
# \boldsymbol{y} = \begin{bmatrix}y_1\\y_2\\ \vdots \\ y_n\end{bmatrix},\;\;\; \boldsymbol{\varepsilon} = \begin{bmatrix}\varepsilon_1\\ \varepsilon_2\\ \vdots \\ \varepsilon_n\end{bmatrix}
# $$
# 
# and write the model simultaneously over all $n$ observations as
# 
# $$
# \boldsymbol{y} = \boldsymbol{X\beta} + \boldsymbol{\varepsilon}.
# $$
# 
# Note that at this point, we haven't done much new -- we've just defined some mathematical objects and used them to simplify the expression for a linear regression model with multiple predictors. However, in doing so, we will be able to use powerful tools from linear algebra to study such models.
