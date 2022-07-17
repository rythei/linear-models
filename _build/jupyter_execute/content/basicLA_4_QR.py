#!/usr/bin/env python
# coding: utf-8

# # Gram--Schmidt and the QR Decomposition
# 
# In the previous workbook, we saw the definition of an _orthogonal set_ of vectors. Indeed, the set $V = \{\boldsymbol{v}_1,\dots, \boldsymbol{v}_k\}$ is an _orthogonal set_ if $\boldsymbol{v}_i^\top \boldsymbol{v}_j = 0$ for all $i\neq j$. The set $V$ is _orthonormal_ if in addition to being orthogonal, we have that $\|\boldsymbol{v}_i\|_2 = 1$ for all $i=1,\dots,k$. Of course, if we have an orthogonal set it is easy to construct an orthonormal set by simply dividing each vector by its norm. Therefore, the hard work lies in finding orthogonal sets.
# 
# In this section, we will study a general procedure for constructing an orthogonal set of vectors from any given set of vectors, and use this method to define an important matrix decomposition called the _QR decomposition_.
# 
# ## The Gram--Schmidt procedure
# 
# Suppose we have a set of vectors $\boldsymbol{a}_1, \dots, \boldsymbol{a}_k \in \mathbb{R}^n$, which we might think of as being the columns of a $n\times k$ matrix $\boldsymbol{A}$. Can we find a set of orthonormal vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$ such that $\text{span}(\boldsymbol{a}_1,\dots, \boldsymbol{a}_k) = \text{span}(\boldsymbol{v}_1,\dots, \boldsymbol{v}_k)$? It turns out that we can use an algorithm called the _Gram--Schmidt procedure_ (or _Gram--Schmidt process_) to accomplish this.
# 
# The algorithm procedes as follows. Start with vectors $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$. Then proceed as follows:
# 
# - $\boldsymbol{u}_1 = \boldsymbol{a}_1$, and set $\boldsymbol{v}_1 = \frac{\boldsymbol{u}_1}{\|\boldsymbol{u}_1\|_2}$
# - $\boldsymbol{u}_2 = \boldsymbol{a}_2 - \frac{\boldsymbol{a}_2^\top \boldsymbol{u}_1}{\boldsymbol{u}_1^\top \boldsymbol{u}_1}\boldsymbol{u}_1 = \boldsymbol{a}_2 - (\boldsymbol{a}_2^\top \boldsymbol{v}_1)\boldsymbol{v}_1$, and set $\boldsymbol{v}_2 = \frac{\boldsymbol{u}_2}{\|\boldsymbol{u}_2\|_2}$
# - $\boldsymbol{u}_3 = \boldsymbol{a}_3 - \frac{\boldsymbol{a}_3^\top \boldsymbol{u}_1}{\boldsymbol{u}_1^\top \boldsymbol{u}_1}\boldsymbol{u}_1 - \frac{\boldsymbol{a}_3^\top \boldsymbol{u}_2}{\boldsymbol{u}_2^\top \boldsymbol{u}_2}\boldsymbol{u}_2 = \boldsymbol{a}_3 - (\boldsymbol{a}_3^\top \boldsymbol{v}_1)\boldsymbol{v}_1 - (\boldsymbol{a}_3^\top \boldsymbol{v}_2)\boldsymbol{v}_2$, and set $\boldsymbol{v}_3 = \frac{\boldsymbol{u}_3}{\|\boldsymbol{u}_3\|_2}$
# - $\vdots$
# - $\boldsymbol{u}_k = \boldsymbol{a}_k - \sum_{j=1}^{k-1} \frac{\boldsymbol{a}_k^\top \boldsymbol{u}_j}{\boldsymbol{u}_j^\top \boldsymbol{u}_j}\boldsymbol{u}_j = \boldsymbol{a}_k - \sum_{j=1}^{k-1}(\boldsymbol{a}_k^\top \boldsymbol{v}_j)\boldsymbol{v}_j$, and set $\boldsymbol{v}_k = \frac{\boldsymbol{u}_k}{\|\boldsymbol{u}_k\|_2}$
# 
# The vectors $\boldsymbol{u}_1,\dots, \boldsymbol{u}_k$ are really the important ones here: they form an orthogonal set with the same span as $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$. The set $\boldsymbol{v}_1,\dots,\boldsymbol{v}_k$ are simply the normalized versions of $\boldsymbol{u}_1,\dots,\boldsymbol{u}_k$, which are therefore an _orthonormal_ set.
# 
# To see why this procedure works, let's look at just the first step, and check that $\boldsymbol{u}_1$ and $\boldsymbol{u}_2$ are in fact orthogonal. To do this, we want to verify that $\boldsymbol{u}_1^\top \boldsymbol{u}_2 = 0$. We have
# 
# 
# $$
# \begin{align*}
# \boldsymbol{u}_1^\top \boldsymbol{u}_2 &= \boldsymbol{a}_1^\top (\boldsymbol{a}_2 - \frac{\boldsymbol{a}_2^\top \boldsymbol{a}_1}{\boldsymbol{a}_1^\top \boldsymbol{a}_1}\boldsymbol{a}_1) \\ &= \boldsymbol{a}_1^\top \boldsymbol{a}_2 - \boldsymbol{a}_1^\top \boldsymbol{a}_1 \frac{\boldsymbol{a}_2^\top \boldsymbol{a}_1}{\boldsymbol{a}_1^\top \boldsymbol{a}_1} = \boldsymbol{a}_1^\top \boldsymbol{a}_2 - \boldsymbol{a}_2^\top \boldsymbol{a}_1 \\&= \boldsymbol{a}_1^\top \boldsymbol{a}_2 - \boldsymbol{a}_1^\top \boldsymbol{a}_2 = 0
# \end{align*}
# $$
# 
# 
# In the second to last inequality, we used the fact that $\boldsymbol{x}^\top \boldsymbol{y} = \boldsymbol{y}^\top \boldsymbol{x}$ for any vectors $\boldsymbol{x},\boldsymbol{y}$. The same type of calculation can be used to check that $\boldsymbol{u}_i^\top \boldsymbol{u}_j = 0$ for any $i\neq j$. Thus the vectors are indeed orthogonal. Moreover, we can see that $\boldsymbol{u}_1,\dots,\boldsymbol{u}_k$ must have the same span as $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$, since $\boldsymbol{u}_j$ can be written as a linear combination of $\boldsymbol{a}_1,\dots, \boldsymbol{a}_j$ for any $j$.
# 
# ### Implementing the Gram--Schmidt procedure in Python
# 
# Let's use Python and numpy to implement the Gram--Schmidt algorithm.
# 
# Let's start with a few helper functions. First, we'll implement a function which takes vectors $\boldsymbol{u}$ and $\boldsymbol{v}$ and computes $\frac{\boldsymbol{v}^\top \boldsymbol{u}}{\boldsymbol{u}^\top \boldsymbol{u}}\boldsymbol{u}$. As we will see in a later section, this is the _orthogonal projection of $\boldsymbol{v}$ onto $\boldsymbol{u}$_, so to be consistent with that interpretation, we will call this function `project_v_onto_u`.

# In[1]:


import numpy as np

def project_v_onto_u(v,u):
    return (np.dot(v,u)/np.dot(u,u))*u


# Next, let's define a function `normalize` which takes a vector $u$ and returns a unit vector in the same direction: $\boldsymbol{v} = \frac{\boldsymbol{u}}{\|\boldsymbol{u}\|_2}$.

# In[2]:


def normalize(u):
    return u/np.linalg.norm(u)


# Finally, let's define a function `gram_schmidt` which uses our helper functions to compute the orthonormal vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$ given a set $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$. To do this, we need to decide how we should take the vectors $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$ as inputs. We will choose to assume that the input is a $n\times k$ matrix $\boldsymbol{A}$, which has the vectors $\boldsymbol{a}_1,\dots, \boldsymbol{a}_k$ as its columns. This will be convenient later on when we compute the QR decomposition. Then we will have our function output a matrix $\boldsymbol{Q}$ whose columns are $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$ -- again, this will be convenient for later on.
# 
# Our function will work as follows: we will have an outer for loop which loops through $i=1,\dots, k$. Then, we will have an inner for loop which loops through $j=1,\dots,i-1$ and iteratively subtracts $\frac{\boldsymbol{a}_i^\top \boldsymbol{u}_j}{\boldsymbol{u}_j^\top \boldsymbol{u}_j}\boldsymbol{u}_j$ from $\boldsymbol{a}_i$.

# In[3]:


def gram_schmidt(A):
    k = A.shape[1]
    u_list = [] # initialize a list to store the u vectors
    u_list.append(A[:, 0]) # u1 = a1
    for i in range(1,k):
        ui = A[:, i] # start with ui = ai
        for j in range(i):
            ui = ui - project_v_onto_u(ui, u_list[j]) # subtract out all the components (ai^T uj)/(uj^T uj)*uj
        u_list.append(ui) # add ui to the list of u vectors
    v_list = [normalize(u) for u in u_list] # normalize all the u vectors
    Q = np.stack(v_list, axis=1) # store the orthonormal vectors into a matrix Q
    return Q


# Let's test our function on a random matrix $\boldsymbol{A}$, and make sure that the matrix $\boldsymbol{Q}$ that we get back does in fact have orthonormal columns -- that is, $\boldsymbol{Q}$ should be an _orthogonal matrix_.

# In[4]:


k = 5
n = 10

A = np.random.normal(size = (n,k))
Q = gram_schmidt(A)


# Recall that we can check that $\boldsymbol{Q}$ is an orthogonal matrix by checking if $\boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I}$. Let's see that this is in fact true. Again, we round to 8 decimals to make the matrix easier to read.

# In[5]:


np.round(np.dot(Q.T, Q), 8)


# Indeed, we see that $\boldsymbol{Q}^\top \boldsymbol{Q}$ is in fact the identity matrix.
# 
# **Remark:** At this point, it is important to point out that the orthogonal matrix $\boldsymbol{Q}$ whose columns have the same span as $\boldsymbol{A}$ is not exactly unique. Indeed, it's easy to see that if we multiply any of the columns of $\boldsymbol{Q}$ by $-1$, we will have an orthogonal matrix with columns spanning the column space of $\boldsymbol{A}$.
# 
# ## From Gram-Schmidt to QR
# 
# Now that we've seen how to take the columns of an arbitrary matrix $\boldsymbol{A}$ and come up with an orthonormal set spanning the column space of $\boldsymbol{A}$, we are in a position to introduce one of the most important matrix decompositions in linear algebra: the _QR decomposition_. In the QR decomposition, we write any matrix $\boldsymbol{A}$ as a product $\boldsymbol{A} = \boldsymbol{QR}$ where $\boldsymbol{Q}$ is an orthogonal matrix, and $\boldsymbol{R}$ is an upper triangular matrix.
# 
# Let's start with the orthonormal vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$ that we obtain from the Gram-Schmidt procedure. Importantly, we can write the columns of $\boldsymbol{A}$ as a linear combination of the vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$. To see how this works, note that from the Gram--Schmidt procedure we have for any $j=1,\dots,k$,
# 
# 
# $$
# \boldsymbol{u}_j = \boldsymbol{a}_j - \sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i \iff \boldsymbol{a}_j = \boldsymbol{u}_j + \sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i = \|\boldsymbol{u}_j\|_2\boldsymbol{v}_j +\sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i
# $$
# 
# 
# Where for the last equality we used the fact that $\boldsymbol{v}_j = \frac{\boldsymbol{u}_j}{\|\boldsymbol{u}_j\|_2}$. Now notice that
# 
# 
# $$
# \begin{align*}
# \|\boldsymbol{u}_j\|_2^2 &= \boldsymbol{u}_j^\top \boldsymbol{u}_j = \boldsymbol{u}_j^\top \left(\boldsymbol{a}_j - \sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i\right) = \boldsymbol{u}_j^\top \boldsymbol{a}_j - \sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\underbrace{(\boldsymbol{u}_j^\top \boldsymbol{v}_i)}_{=0}\\&= \boldsymbol{u}_j^\top \boldsymbol{a}_j = \|\boldsymbol{u}_j\|_2(\boldsymbol{v}_j^\top \boldsymbol{a}_j) \implies \|\boldsymbol{u}_j\|_2 = (\boldsymbol{v}_j^\top \boldsymbol{a}_j)
# \end{align*}
# $$
# 
# 
# Hence we get the following expression for $\boldsymbol{a}_j$:
# 
# 
# $$
# \boldsymbol{a}_j = \|\boldsymbol{u}_j\|_2\boldsymbol{v}_j +\sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i = (\boldsymbol{a}_j^\top \boldsymbol{v}_j)\boldsymbol{v}_j + \sum_{i=1}^{j-1}(\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i = \sum_{i=1}^j (\boldsymbol{a}_j^\top \boldsymbol{v}_i)\boldsymbol{v}_i \hspace{20mm} (\star)
# $$
# 
# 
# Therefore $\boldsymbol{a}_j$ can be written as a linear combination of $c_1 \boldsymbol{v}_1 + \cdots + c_j\boldsymbol{v}_j$ where $c_i = (\boldsymbol{a}_j^\top \boldsymbol{v}_i)$.
# 
# Now let's collect the coefficients $(\boldsymbol{a}_j^\top \boldsymbol{v}_j)$ into the following matrix:
# 
# 
# $$
# \boldsymbol{R} = \begin{bmatrix}\boldsymbol{a}_1^\top \boldsymbol{v}_1 & \boldsymbol{a}_2^\top \boldsymbol{v}_1 & \boldsymbol{a}_3^\top \boldsymbol{v}_1&\cdots  \\
# 									0 & \boldsymbol{a}_2^\top \boldsymbol{v}_2 &\boldsymbol{a}_3^\top \boldsymbol{v}_2& \cdots \\
# 									0 & 0 & \boldsymbol{a}_3^\top \boldsymbol{v}_3 & \cdots \\
# 									\vdots & \vdots &\vdots &\ddots
# \end{bmatrix}
# $$
# 
# 
# That is, the $k\times k$ matrix $\boldsymbol{R}$ whose $(i,j)^{th}$ entry is $\boldsymbol{a}_j^\top \boldsymbol{v}_i$ if $i\leq j$, and $0$ otherwise (matrices of this form -- with zeros below the diagonal -- are called _upper triangular_). Let's again store the vectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_k$ as the columns of a $n\times k$
# 
# 
# $$
# \boldsymbol{Q} = \begin{bmatrix} | & | &  &| \\ \boldsymbol{v}_1 & \boldsymbol{v}_2 & \cdots & \boldsymbol{v}_k \\ | & | & & |\end{bmatrix}
# $$
# 
# 
# Using this notation, we can write the relationship $(\star)$ as $\boldsymbol{a}_j = \boldsymbol{Q}\boldsymbol{r}_j$, where $\boldsymbol{r}_j$ is the $j^{th}$ column of the matrix $\boldsymbol{R}$. In particular then, if we stack all these columns together, we get that
# 
# 
# $$
# \boldsymbol{A} = \boldsymbol{QR} = \begin{bmatrix} | & | & &| \\ \boldsymbol{v}_1 & \boldsymbol{v}_2 & \cdots & \boldsymbol{v}_k \\ | & | &  & |\end{bmatrix}\begin{bmatrix}\boldsymbol{a}_1^\top \boldsymbol{v}_1 & \boldsymbol{a}_2^\top \boldsymbol{v}_1 & \boldsymbol{a}_3^\top \boldsymbol{v}_1&\cdots  \\
# 									0 & \boldsymbol{a}_2^\top \boldsymbol{v}_2 &\boldsymbol{a}_3^\top \boldsymbol{v}_2& \cdots \\
# 									0 & 0 & \boldsymbol{a}_3^\top \boldsymbol{v}_3 & \cdots \\
# 									\vdots & \vdots &\vdots &\ddots
# \end{bmatrix}
# $$
# 
# 
# This expression -- writing $\boldsymbol{A}$ as a product of an orthogonal matrix $\boldsymbol{Q}$ and an upper triangular matrix of coefficients $\boldsymbol{R}$ -- is called the _$\boldsymbol{QR}$ decomposition of $\boldsymbol{A}$_. In words, this decomposition expresses the columns of $\boldsymbol{A}$ in terms of an orthogonal basis $\boldsymbol{Q}$, which we obtain through Gram--Schmidt.
# 
# ### Computing the QR decomposition in Python
# 
# Let's implement the QR decomposition in Python. Since we've already implemented Gram-Schmidt above, we can use that function to obtain the matrix $\boldsymbol{Q}$. Thus, all we have left to do is find the upper triangular matrix $\boldsymbol{R}$. We could go through and compute all the entries of $\boldsymbol{R}$ manually, however, if we notice that $\boldsymbol{Q}^\top \boldsymbol{Q}= \boldsymbol{I}$, we observe that
# 
# 
# $$
# \boldsymbol{A} = \boldsymbol{QR} \iff \boldsymbol{Q}^\top \boldsymbol{A} = \boldsymbol{Q}^\top \boldsymbol{QR} = \boldsymbol{R}
# $$
# 
# 
# Thus we can compute $\boldsymbol{R}$ immediately by calculating $\boldsymbol{Q}^\top \boldsymbol{A}$. Let's combine all these steps into a single function which computes $\boldsymbol{Q}$ and $\boldsymbol{R}$ for any given matrix $A$.

# In[6]:


def qr_decomposition(A):
    Q = gram_schmidt(A) #use Gram-Schmidt to compute Q
    R = np.dot(Q.T, A) #find R = Q^TA
    return Q, R


# Let's test this again on a random matrix $A$.

# In[7]:


k = 5
n = 10

A = np.random.normal(size = (n,k))
Q, R = qr_decomposition(A)


# Now let's check that $\boldsymbol{R}$ is indeed upper triangular.

# In[8]:


R.round(8)


# Let's also check that $\boldsymbol{A} = \boldsymbol{QR}$.

# In[9]:


np.allclose(A, np.dot(Q,R))


# Indeed, it does. For this section, we simply focus on the mechanics of the QR decomposition. In the following sections of this chapter, we will see that this is an extremely useful tool.
