#!/usr/bin/env python
# coding: utf-8

# # Linear regression with multiple predictors
# 
# _Datasets used in throughout this book can be downloaded [here](https://drive.google.com/drive/folders/1OkXMcFo0urN0kSQYH4d75I4V3pnSpV6H?usp=sharing)._
# 
# Let us continue our discussion of linear regression with multiple predictor variables. In this case, we want to model a response $y_i$ as a linear combination of $p$ predictor variables $x_{i1},\dots, x_{ip}$ plus some noise:
# 
# $$
# y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i.
# $$
# 
# As we saw in the previous section, this can be written concisely for all $n$ observations simultaneously as
# 
# $$
# \boldsymbol{y} = \boldsymbol{X\beta} + \boldsymbol{\varepsilon}.
# $$
# 
# Like in the single predictor case, our objective will be to minimize the sum of squared errors:
# 
# $$
#  \sum_{i=1}^n (y_i - \boldsymbol{\beta}\cdot \boldsymbol{x}_i)^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2
# $$
# 
# Conveniently, this can be written in terms of the squared Euclidean norm:
# 
# $$
# \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\boldsymbol{y} - \hat{\boldsymbol{y}}\|_2^2 = \|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2
# $$
# 
# where
# 
# $$
# \boldsymbol{X} = \begin{bmatrix} \boldsymbol{x}_1 \\ \boldsymbol{x}_2\\ \vdots\\ \boldsymbol{x}_n\end{bmatrix} \in \mathbb{R}^{n\times (p+1)}
# $$
# 
# (where recall we made the first entry of each row $\boldsymbol{x}_i$ 1 to account for the intercept term) and
# 
# $$
# \boldsymbol{\beta} = \begin{bmatrix}\beta_0 \\ \beta_1\\ \beta_2 \\ \vdots\\ \beta_p \end{bmatrix} \in \mathbb{R}^{p+1},\;\;\; \boldsymbol{y} = \begin{bmatrix}y_1\\y_2\\ \vdots \\ y_n\end{bmatrix}\in \mathbb{R}^n,\;\;\; \boldsymbol{\varepsilon} = \begin{bmatrix}\varepsilon_1\\ \varepsilon_2\\ \vdots \\ \varepsilon_n\end{bmatrix}\in \mathbb{R}^n.
# $$
# 
# ## Solving the least squares problem
# 
# In the single predictor case, we individually took derivates with respect to $\beta_0$ and $\beta_1$ and set these equal to zero to find the least squares solutions. We could do the same here as well, however we can simplify the calculations considerably by using some basic formulas from matrix calculus:
# 
# 1. For vectors $\boldsymbol{x},\boldsymbol{y}$, $\frac{d}{d\boldsymbol{x}} \boldsymbol{x}\cdot \boldsymbol{y}= \boldsymbol{y}$.
# 2. For a symmetric matrix $\boldsymbol{A}$ (meaning that $\boldsymbol{A}^\top = \boldsymbol{A}$), we have $\frac{d}{d\boldsymbol{x}} \boldsymbol{x}\cdot \boldsymbol{Ax} = 2\boldsymbol{Ax}$.
# 
# Now using our fact from the previous section that $\|\boldsymbol{a}\|_2^2 = \boldsymbol{a}\cdot \boldsymbol{a}$, using a bit of algebra (mostly the distributive property) we can write
# 
# $$
# \begin{align*}
# \|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2 &= (\boldsymbol{y} - \boldsymbol{X\beta})\cdot (\boldsymbol{y} - \boldsymbol{X\beta})\\
# &= \boldsymbol{y}\cdot \boldsymbol{y} - 2\boldsymbol{y}\cdot \boldsymbol{X\boldsymbol{\beta}} + \boldsymbol{\beta}\cdot \boldsymbol{X^\top X \beta}
# \end{align*}
# $$
# 
# Now using our derivate formulas from above, we have
# 
# $$
# \begin{align*}
# \frac{d}{d\boldsymbol{\beta}} \|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2 =& \frac{d}{d\boldsymbol{\beta}}\{\boldsymbol{y}\cdot \boldsymbol{y} - 2\boldsymbol{y}\cdot \boldsymbol{X\boldsymbol{\beta}} + \boldsymbol{\beta}\cdot \boldsymbol{X^\top X \beta} \} \\
# &= -2\boldsymbol{X^\top y} + 2\boldsymbol{X^\top X\beta} = 0
# \end{align*}
# $$
# 
# Solving this, we obtain an important set of equations called the _normal equations_:
# 
# $$
# \boldsymbol{X^\top X\beta} = \boldsymbol{X^\top y} \hspace{10mm} (1)
# $$
# 
# To find the solution $\widehat{\boldsymbol{\beta}}$, we have to somehow solve the normal equations for $\boldsymbol{\beta}$. For now, we will focus on the simplest case when $\boldsymbol{X^\top X}$ is an invertible matrix, in which case we can simply multiply both sides of $(1)$ by $(\boldsymbol{X^\top X})^{-1}$ to get
# 
# $$
# \widehat{\boldsymbol{\beta}} = (\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top y}.
# $$
# 
# This formula gives the most standard form of the solution to the problem $\min_{\boldsymbol{\beta}} \|\boldsymbol{y} - \boldsymbol{X\beta}\|_2^2$.
# 
# Before proceeding to some examples, let's define a function which takes in a data matrix $\boldsymbol{X}$ and a vector of responses $\boldsymbol{y}$, and returns the least squares solution $\widehat{\boldsymbol{\beta}}$. To do this, we will use numpy's functions `np.dot` and `np.linalg.inv` (to compute $(\boldsymbol{X^\top X})^{-1}$).

# In[1]:


import numpy as np

def fit_linear_regression(X, y):
    # compute X^TX
    XTX = np.dot(X.T, X)

    # compute the inverse of X^TX
    XTX_inverse = np.linalg.inv(XTX)

    # compute X^Ty
    XTy = np.dot(X.T, y)

    # compute the least squares coefficients beta = (X^TX)^{-1}X^Ty
    beta_hat = np.dot(XTX_inverse, XTy)

    return beta_hat


# Now that we've written code to fit a generic multiple linear regression model, let's see an example with an actual dataset.
# 
# ## An example
# 
# As an example, we will use the California housing dataset, which can be loaded using `pandas` with the following.

# In[2]:


import pandas as pd

dataset = pd.read_csv("datasets/california_housing.csv")
dataset.head()


# This dataset contains features describing houses in the state of California, such as the average number of rooms in a house and the median income of persons in a given house, as well as a variable `MedHouseVal` describing the median house value in a given region (measured in $100,000 units). Let's first visualize our data to get a sense of whether or not any of the variables seem to be obviously skewed (in which case we might want to transform them). The python package `seaborn` let's us do this conveniently with the following code:

# In[3]:


import seaborn as sns

fig = sns.PairGrid(dataset)
fig.map_offdiag(sns.scatterplot)
fig.map_diag(sns.histplot)


# On the diagonals, we see histograms of each of the individual features. On the $(i,j)^{th}$ off-diagonal, we get scatterplots of feature $j$ against feature $i$. We observe that some of the features may be slightly skewed, which we could resolve by transforming them in some way. However, to keep the model simple for the sake of this example, we will keep with the features in their original units.
# 
# Before we actually fit a regression model, let's gather the data into a numpy arrays so that they are easier to work with.

# In[4]:


# first get all the columns except for the response, MedHouseVal
X = dataset[[c for c in dataset.columns if c!="MedHouseVal"]].to_numpy()
y = dataset["MedHouseVal"].to_numpy()
print(X.shape, y.shape)


# Next, we need to add a column of 1's to the `X` array to account for the intercept term. This can be done easily with the following:

# In[5]:


# create a (n, 1) array of all ones
ones = np.ones(X.shape[0]).reshape(-1,1)

# add this array as a new column of X
X = np.hstack([ones, X])

print(X.shape)


# Now we can use our method `fit_linear_regression` that we wrote above to actually fit the model.

# In[6]:


beta_hat = fit_linear_regression(X, y)


# Let's inspect the coefficients for each of the features:

# In[7]:


# get all the names of the columns
feature_names = ["Intercept"] + [c for c in dataset.columns if c!="MedHouseVal"]

for name, coeff in zip(feature_names, beta_hat):
    print(f"{name}: {coeff}")


# Now we can directly interpret the coefficients. For example, we see that the coefficient for population is almost zero, meaning that the house prices don't seem to depend on how large the population of the area around the house is. On the other hand, features like `MedInc` (median income) and `AvgBedrms` (average # bedrooms) seems to significantly increase the price of homes. Interestingly, the coefficients for latitude and longitude are both negative, indicating that the more north and east you go in California, the less expensive the houses.
# 
# ## Diagnostics for multiple least squares
# 
# In the simple linear regression setting, we developed a few diagnostic tools that we can use to assess the fit of a linear regression model, for by inspecting residuals and computing the $R^2$. For this section, denote $\hat{\boldsymbol{\beta}}$ the fitted least squares parameters and let $\boldsymbol{X}, \boldsymbol{y}$ be the data the model was fit to. Then the fitted response values are given by
# 
# $$
# \hat{\boldsymbol{y}} = \boldsymbol{X}\hat{\boldsymbol{\beta}}.
# $$
# 
# Like in the single predictor case, we can define the residual sum of squares
# 
# $$
# \text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2
# $$
# 
# and the total sum of squares
# 
# $$
# \text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2
# $$
# 
# where $\bar{y} = \sum_{i=1}^n y_i$. Likewise, the regression sum of squares is again
# 
# $$
# \text{RegSS} = \text{TSS} - \text{RSS}
# $$
# 
# Given these, the definition of the $R^2$ from the single predictor case immediately carries over to the multiple linear regression case, and is given by
# 
# $$
# R^2 = \frac{\text{RegSS}}{\text{TSS}}.
# $$
# 
# Let's compute the $R^2$ for the model we've just fit above. First, we need the fitted values $\hat{\boldsymbol{y}}$, and the mean response $\bar{y}$:

# In[8]:


y_bar = np.mean(y)
y_hat = np.dot(X, beta_hat)


# Now we can compute the RSS, TSS and RegSS,

# In[9]:


RSS = np.sum((y-y_hat)**2)
TSS = np.sum((y-y_bar)**2)
RegSS = TSS-RSS


# Finally, we can get the $R^2$:

# In[10]:


R2 = RegSS/TSS
R2


# The $R^2$ is $\approx 0.6$, meaning that this model is able to explain about 60% of the variance in the house prices.
# 
# Note: there are many other things that we could explore here. For example, it would be good to try various transformations of our features, or perhaps seeing if using only a subset of the features can produce a similar quality model.
