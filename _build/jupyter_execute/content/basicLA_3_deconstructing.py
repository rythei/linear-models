#!/usr/bin/env python
# coding: utf-8

# # Deconstructing Matrix Multiplication
# 
# In this section, we will "deconstruct" matrix multiplication, by viewing it in terms of operations on the rows and columns of the matrices being multiplied. This will give us the opportunity to further our intuition for what matrix multiplication is really doing, and how this understanding can be useful from the perspective of computational efficiency.
# 
# ## Computing columns of a matrix product
# 
# Suppose we had two large matrices $\boldsymbol{A}\in \mathbb{R}^{n\times m}$ and $\boldsymbol{B}\in\mathbb{R}^{m\times p}$ that contain a bunch of information, but we're only interested in computing the $i^{th}$ column of the product $\boldsymbol{AB}$.
# 
# A naive way to find this column is to first compute the product $\boldsymbol{AB}$ and then select the $i^{th}$ column using slicing in Python.
# Let's try this approach.
# 
# Since we are interested in the properties of matrix multiplication, we can work with any matrices.
# So, let's keep things simple and use random matrices.
# We first define two random matrices $\boldsymbol{A}$ and $\boldsymbol{B}$.

# In[1]:


import numpy as np

n, m, p = 1000, 100, 1000

A = np.random.rand(n, m)
B = np.random.randn(m, p)


# Let's time how long it takes to compute $\boldsymbol{AB}$ and then select the $i^{th}$ column of the product.

# In[2]:


import time

i = 20

tic = time.time()
AB = np.dot(A,B)
ith_column = AB[:,i]
print('time taken to compute AB and select the ith column: ', time.time()- tic)


# This works, but as we'll see it is not the most effecient way to find the desired column.
# 
# Let's write $\boldsymbol{B}$ in block form, representing it in terms of its columns.
# 
# 
# $$
# \boldsymbol{B} = \begin{bmatrix}|& | && |\\ \boldsymbol{b}_{:1}&  \boldsymbol{b}_{:2}& \cdots & \boldsymbol{b}{:p}\\ |&|&&|\end{bmatrix}
# $$
# 
# 
# Then the product $\boldsymbol{AB}$ can be written as
# 
# 
# $$
# \boldsymbol{AB} = \boldsymbol{A}\begin{bmatrix}|& | && |\\ \boldsymbol{b}_{:1}&  \boldsymbol{b}_{:2}& \cdots & \boldsymbol{b}{:p}\\ |&|&&|\end{bmatrix}
# $$
# 
# 
# From this representation, we see that the $i^{th}$ column of $\boldsymbol{AB}$ is really just $\boldsymbol{A}\boldsymbol{b}_{:i}$ -- or the matrix-vector product of $\boldsymbol{A}$ with the $i^{th}$ column of $\boldsymbol{B}$.
# Therefore, we see that we can compute the $i^{th}$ column of $\boldsymbol{AB}$ without having to compute the whole matrix $\boldsymbol{AB}$ first: we can simply select the $i^{th}$ column $\boldsymbol{b}_{:i}$ of $\boldsymbol{B}$, and then apply $\boldsymbol{A}$ to it.
# Let's try this method, and compare the time with the above method.

# In[3]:


tic = time.time()
ith_column_fast = np.dot(A,B[:,i])
print('time taken to compute A*B[:,i]: ', time.time()- tic)


# As we can see, this method is much faster.
# These matrices were not too large; but as the matrices get larger, this speedup will only become greater.
# Let's also verify that the two approaches give the same result.

# In[4]:


np.allclose(ith_column, ith_column_fast)


# This method is easily generalized to selecting a subset of the columns of $\boldsymbol{AB}$.
# For example, suppose we wanted to select the $1^{st}$, $5^{th}$ and $11^{th}$ columns of $\boldsymbol{AB}$.
# Then we could multiply $\boldsymbol{A}$ by only the columns $1,5$ and $11$ of $\boldsymbol{B}$.
# In Python, we can do this with the following code.

# In[5]:


cols = [0,4,10]

tic = time.time()
AB = np.dot(A,B)
subset_of_columns_slow = AB[:,cols]
print('time taken to compute AB and select subset of columns: ', time.time()- tic)

tic = time.time()
subset_of_columns_fast = np.dot(A,B[:,cols])
print('time taken to compute A*B[:,cols]: ', time.time()- tic)


# Again, we can verify that the two approaches give the same result.

# In[6]:


np.allclose(subset_of_columns_slow, subset_of_columns_fast)


# ## Computing rows of a matrix product
# 
# Like in the above section with columns, we can also take advantage of the structure of matrix multiplication in computing a single row of a matrix product $\boldsymbol{AB}$.
# To see this, let's write
# 
# $$
# \boldsymbol{A} = \begin{bmatrix}- & \boldsymbol{a}_{1:}^\top & -\\ - & \boldsymbol{a}_{2:}^\top & -\\ & \vdots& \\ - &\boldsymbol{a}_{n,:}^\top& -\end{bmatrix}  ,
# $$
# 
# where $\boldsymbol{a}_{i:}^\top$ is the $i^{th}$ row of $\boldsymbol{A}$.
# Then if we write out the matrix product $\boldsymbol{AB}$ as
# 
# $$
# \boldsymbol{AB} = \begin{bmatrix}- & \boldsymbol{a}_{1:}^\top & -\\ - & \boldsymbol{a}_{2:}^\top & -\\ & \vdots& \\ - &\boldsymbol{a}_{n,:}^\top& -\end{bmatrix} \boldsymbol{B}
# $$
# 
# we observe that the $i^{th}$ row of $\boldsymbol{AB}$ is given by $\boldsymbol{a}_{i:}^\top \boldsymbol{B}$.
# Let's compare this method to the naive approach of computing the full product $\boldsymbol{AB}$ and then selecting the $i^{th}$ row.

# In[7]:


i = 20

tic = time.time()
AB = np.dot(A,B)
ith_row = AB[i,:]
print('time taken to compute AB and select the ith row: ', time.time()- tic)

tic = time.time()
ith_row_fast = np.dot(A[i,:],B)
print('time taken to compute A[i,:]*B: ', time.time()- tic)


# As expected, the method of computing $\boldsymbol{a}_{i:}^\top \boldsymbol{B}$ is substantially faster than computing $\boldsymbol{AB}$ and then extracting the $i^{th}$ row.
# Let's verify that they do indeed give the same results.

# In[8]:


np.allclose(ith_row, ith_row_fast)


# Likewise, we can follow the same approach as above to select a subset of rows of the product $\boldsymbol{AB}$.
# For example, if we wanted the $4^{th}$, $12^{th}$ and $20^{th}$ rows of $\boldsymbol{AB}$, we can do so with the following.

# In[9]:


rows = [3, 11, 19]

tic = time.time()
AB = np.dot(A,B)
subset_of_rows_slow = AB[rows,:]
print('time taken to compute AB and select subset of rows: ', time.time()- tic)

tic = time.time()
subset_of_rows_fast = np.dot(A[rows,:],B)
print('time taken to compute A[rows,:]*B: ', time.time()- tic)


# Again, we can verify that the two methods give the same result.

# In[10]:


np.allclose(subset_of_rows_slow, subset_of_rows_fast)


# For both of these examples (finding columns and finding rows of $\boldsymbol{AB}$), the speedup becomes even more dramatic and we make the matrices larger.
# This is because we are computing more unnecessary products to find $\boldsymbol{AB}$ as the dimensions get large.
# You can see this yourself by changing the values of $n,m$ and $p$ in the cells above and re-running the same code given here.
# In data science, we often encounter very large matrices when working with big datasets, and keeping the structure of operations like matrix multiplication in mind when working with these datasets can save you a great deal of computation time in practice.
