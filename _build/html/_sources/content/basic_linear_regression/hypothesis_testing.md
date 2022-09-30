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

# Hypothesis testing for the Gaussian model

In this section, we will show that, under the assumption that the Gaussian model is true, we can perform various forms of statistical inference on a fitted linear regression model. In particular, in this section, we will focus on performing hypothesis testing on the coefficients $\hat{\boldsymbol{\beta}}$, though the same ideas can be extended to do other forms of inference, such as obtaining confidence intervals.

We will derive two types of hypothesis tests:

1. A $t$-test for an individual coefficient $\hat{\beta}_j$, and
2. An $F$-test for a subset of coefficients $\hat{\beta}_{j_1},\dots,\hat{\beta}_{j_k}$.


Recall that for the Gaussian model of linear regression, we assume that the responses $y_i$ are generated as

$$
y_i = \boldsymbol{\beta}\cdot \boldsymbol{x}_i + \varepsilon_i
$$

where $\varepsilon_i \sim N(0,\sigma^2)$. Under this assumption, we saw that the least squares coefficients $\hat{\boldsymbol{\beta}} = (\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top y}$ be distributed as

$$
\hat{\boldsymbol{\beta}}\mid \boldsymbol{X} \sim N(\boldsymbol{\beta}, \sigma^2 \boldsymbol{X^\top X}^{-1}). \hspace{10mm} (1)
$$

In the following sections, we will use this fact to derive the aformentioned hypothesis tests. As working example, we will again use the fish toxicity dataset that we've used in the previous section. We load this dataset below.

```{code-cell}
import pandas as pd

dataset = pd.read_csv("datasets/qsar_fish_toxicity.csv")
dataset.head()
```

As usual, we will extract the response as a numpy array, and the other features as another array, adding a column of 1's to account for the intercept term in the regression.

```{code-cell}
import numpy as np

# extract the data
y = dataset["LC50"].to_numpy()
cols = [c for c in dataset.columns if c!="LC50"]
X = dataset[cols].to_numpy()

# add a column of ones to the X matrix
ones = np.ones(X.shape[0]).reshape(-1,1)
X = np.hstack([ones, X])

X.shape, y.shape
```

We will compare the output of our own code with that of the standard package for fitting statistical models in python, called `statsmodels`. This output will look similar to almost any other statistical software for fitting a linear model, such as `lm` in R.

To fit the model using `statsmodels`, we simply run:

```{code-cell}
import statsmodels.api as sm

results = sm.OLS(y, X).fit()
results.summary()
```

Now let's fit the model manually, and make sure we can recover the same coefficients (you could also check that the methods we've introduced for computing the $R^2$ and adjusted $R^2$ also match this output).

```{code-cell}
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
beta_hat = np.dot(XTX_inv, np.dot(X.T, y))
beta_hat
```

Indeed, these coefficients are exactly the same as the expected output.

## A $t$ test for the hypothesis $\beta_j = 0$

The first hypothesis test we will develop is for the null hypothesis that _one_ of the coefficients, say $\beta_j$ is equal to $0$. This corresponds to the fact that feature $j$ in the regression is not predictive of the response $y$. Formally, we can write this hypothesis as follows:

$$
\begin{align*}
H_0: \beta_j = 0\\
H_a: \beta_j \neq 0
\end{align*}
$$

To begin, let's suppose that the variance $\sigma^2$ of the errors were known. Then in this case, we can use the fact, from (1), that

$$
\hat{\beta}_j \sim N(\beta_j, \sigma^2 (\boldsymbol{X^\top X})_{jj}^{-1}).
$$

Under the null hypothesis $\beta_j = 0$, this becomes $\beta_j \sim N(0,\sigma^2 (\boldsymbol{X^\top X})^{-1})$, and so in particular we can compute the statistic

$$
\hat{z}_j = \frac{\hat{\beta}_j}{\sqrt{\sigma^2 (X^\top X)^{-1}_{jj}}}
$$

and obtain the usual $p$-value by computing $P(|Z| > |\hat{z}_j|)$, where $Z\sim N(0,1)$.

While simple, this test won't quite be correct in practice, as we don't actually know the true variance $\sigma^2$. Instead, we must estimate this variance using the data. To do this, we will use the estimator

$$
\hat{\sigma}^2 = \frac{1}{n-p}\|\boldsymbol{y} - \boldsymbol{X}\hat{\boldsymbol{\beta}}\|_2^2.
$$

Note that this is slightly different than the maximum likelihood estimate of the variance that we derived in the previous section; choosing to scale by $1/(n-p)$ ensures that $\hat{\sigma}^2$ is an unbiased estimate of $\sigma^2$.

Before we can return to hypothesis testing, we first need one more fact regarding the distribution of $\hat{\sigma}^2$, namely that it is distributed as

$$
(n-p)\hat{\sigma}^2/\sigma^2 \sim \chi^2(n-p).
$$


At the bottom of this section, we walk through a sketch of a proof of this fact using some basic linear algebra. In the meantime, we can use this fact to define a slightly different statistic to $\hat{z}$:

$$
\hat{t}_j = \frac{\hat{\beta}_j}{\sqrt{\hat{\sigma}(\boldsymbol{X^\top X})^{-1}_{jj}}} = \frac{\hat{\beta}_j/\sqrt{\sigma^2 (\boldsymbol{X^\top X})^{-1}_{jj}}}{\hat{\sigma}/\sigma}
$$

The numerator $\hat{\beta}_j/\sqrt{\sigma^2 (\boldsymbol{X^\top X})^{-1}_{jj}}$ is exactly $\hat{z}$, which we saw under the null hypothesis $\beta_j = 0$ follows a standard normal distribution. The denominator of $\hat{t}$ is equal to

$$
\sqrt{\frac{(n-p)\hat{\sigma}^2/\sigma^2}{n-p}}
$$

which is exactly the square root of a $\chi^2(n-p)$ distribution divided by its degrees of freedom, $n-p$. This means that $\hat{t}_j$ will follow a $t$-distribution with degrees of freedom $n-p$. Thus when $\sigma^2$ is unknown, we can test the null hypothesis $\beta_j = 0$ by computing the $p$-value $P(|T| > |\hat{t}_j|)$ where $T \sim t(n-p)$.

Now that we've derived the $t$-test for testing $\beta_j = 0$, let's try to implement these ourselves. Let's begin by computing the estimate $\hat{\sigma}^2$ of the variance.

```{code-cell}
y_hat = np.dot(X, beta_hat)
sigma2_hat = 1./(X.shape[0]-X.shape[1])*np.linalg.norm(y-y_hat)**2
```

Next, we can compute the $t$ statistics for each of the coefficients.

```{code-cell}
cols = ["intercept"] + cols
t_stats = []
for j in range(X.shape[1]):
    sigma2_hat_j = sigma2_hat*XTX_inv[j,j]
    t_stat_j = beta_hat[j]/np.sqrt(sigma2_hat_j)
    t_stats.append(t_stat_j)

t_stats
```
These again produce the same output as the table from `statsmodels`! Finally, we can use `scipy` to compute probabilities under the $t$ distribution to get the desired $p$-values.

```{code-cell}
from scipy.stats import t
p_vals = []
t_dist = t(df=(X.shape[0]-X.shape[1]))

for j in range(X.shape[1]):
    p_val_j = 2*(1-t_dist.cdf(t_stats[j]))
    p_vals.append(p_val_j)

p_vals
```

These again match the results from the `statsmodels` implementation. To make this a bit clearer, let's organize all the outputs we've had so far into a nice table.

```{code-cell}
regression_outputs = pd.DataFrame({"variable": cols, "coefficient": beta_hat, "t-stat": t_stats, "p-value": p_vals})
regression_outputs
```

Let's now actually interpret what the results of these $t$-tests mean. To do this, let's suppose we've set a significance threshold of $0.05$. Then for each _individual_ coefficient, with the exception of the chemical descriptor `NdssC` (measuring the counts of certain atom-types in a molecule), we reject the null hypothesis that the coefficient is equal to zero.

Assuming the Gaussian model is correct, these are valid $p$-values for testing each one of these coefficients individually. However, we need to be careful when interpreting more than one of these tests at a time. To understand why, suppose we wanted to test whether both $\beta_1 = 0$ _and_ $\beta_2 = 0$. This hypothesis fails when _either one_ of $\beta_1$ or $\beta_2$ is different from zero. The $p$-values we obtain give us the probability that a randomly drawn $\hat{t}_1$ and a randomly drawm $\hat{t}_2$ from the $t$ distribution are at least as extreme as their observed values, but to test the hypothesis that $\beta_1 = 0$ _and_ $\beta_2 = 0$ what we really need is

$$
P(|T| > |\hat{t}_1| \text{  or  } |T| > |\hat{t}_2|).
$$

It is entirely possible that even if the $p$-value for $\hat{t}_1$ and the $p$-value for $\hat{t}_2$ are both below some significance threshold, the above probability is not. This is called the problem of _multiple hypothesis testing_ or _multiple comparisons_. There are simple ways to correct for this (e.g. by multiplying the $p$-values by the number of hypotheses to be tested, called a [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction)), though in the following section we will see that there is a much better way to compute a joint hypothesis over the coefficients, using an $F$-test.

## An $F$ test for joint hypotheses over the coefficients

In this subsection, we will describe a method for testing the hypothesis

$$
\begin{align}
&H_0: \beta_1=\beta_2=\cdots = \beta_p = 0\\
&H_a: \text{At least one $\beta_j \neq 0$} \hspace{10mm} (2)
\end{align}
$$

However, the methods presented here can easily be extended to more general hypotheses of the form

$$
\begin{align*}
&H_0: \beta_{j_1}= c_1, \; \beta_{j_2}= c_2, \cdots, = \beta_{j_k} = c_k\\
&H_a: \text{At least one $\beta_{j_l} \neq 0$}
\end{align*}
$$

where now $j_1,\dots,j_k \subseteq \{1,\dots,p\}$ are a subset of the features, and $c_1,\dots,c_k$ are any set of hypothesized values.

To test the hypothesis $(2)$, we need to first develop a reasonable test statistic. Let's suppose that the null hypothesis that $\beta_1=\cdots=\beta_p=0$ is true. In this case

$$
y_i = \beta_0 + \varepsilon_i.
$$

If this were the case, we would expect the responses $y_i$ to look like normally distributed noise around the constant value $\bar{y}$, and the predictions $\hat{y}_i = \hat{\boldsymbol{\beta}}\cdot \boldsymbol{x}_i$ shouldn't be much better than this. On the other hand, if the null hypothesis is not true -- and at least one of the coefficients from zero -- then the predictions $\hat{y}_i$ should be significantly better than the naive estimate $\bar{y}$.

To compare these two scenarios, we could compare the value $\|y_i - \bar{y}\|_2^2$ (which we have previously called the TSS) to the value of $\|y_i - \hat{y}_i\|_2^2$ (previously called the RSS). If the TSS is significantly larger than the RSS, then we would have evidence that the intercept-only model (corresponding to the null hypothesis $\beta_1=\cdots=\beta_p=0$) probably isn't true. One way to do this would be to look at the ratio

$$
\frac{\text{TSS}}{\text{RSS}}
$$

In practice, it is more standard however to modify this statistic a bit. Since $\text{TSS} = \text{RegSS} + \text{RSS}$, We could write the statistic as

$$
\frac{\text{TSS}}{\text{RSS}} = \frac{\text{RegSS} + \text{RSS}}{\text{RSS}} = 1+ \frac{\text{RegSS}}{RSS}.
$$

In this case, we could equivalently test whether $\text{RegSS}/\text{RSS}$ is significantly greater than zero. Importantly, it's possible to show that a) the RSS and RegSS are statistically independent, and 2) that under $H_0$, $\text{RegSS} = \|\hat{y}_i - \bar{y}\|_2^2 \sim \chi^2(p-1)$ and, and we showed in the previous section, that $\text{RSS} = \|y_i - \hat{y}_i\|_2^2 \sim \chi^2(n-p)$. With this known, we define the $F$ statistic

$$
\hat{F} = \frac{\|\hat{y}_i - \bar{y}\|_2^2/(p-1)}{\|y_i-\hat{y}_i\|_2^2/(n-p)}.
$$

This is exactly the ratio of two chi-squared distributions, each divided by their degrees of freedom. By definition, this will follow and $F(p-1,n-p)$ distribution. From this statistic, a p-value can be obtained by computing $P(F > \hat{F})$. Let's return to the fish toxicity example and actually compute this statistic.

```{code-cell}
y_bar = np.mean(y)

regss = np.sum((y_hat - y_bar)**2)
rss = np.sum((y-y_hat)**2)
F_hat = (regss/(X.shape[1]-1))/(rss/(X.shape[0]-X.shape[1]))
F_hat
```

This again matches the $F$-statistic entry from the `statsmodels` output! Finally, we can use this statistic to compute a $p$-value from the $F$ distribution.

```{code-cell}
from scipy.stats import f

f_dist = f(dfn=(X.shape[1]-1), dfd=(X.shape[0]-X.shape[1]))
p_val = (1-f_dist.cdf(F_hat))
p_val
```

Again we obtain the same $p$-value as the `statsmodels` output (zero). This means that we have confidence, at any significance threshold, that at least one of the coefficients is different from zero. In some settings, this is verification that the model we are studying is non-trivial: we've found evidence that _some_ variable is a non-trivial predictor of the response. However, if we needed to find out which variables it is, we would need to return to our $t$-tests and provide the appropriate correction for testing multiple hypotheses.

Of course, everything we've covered so far relies upon the fact that the Gaussian model is correct. This involves to assumptions (either of which failing to hold would make the tests derived here meangless): i) that the true model is actually linear, and ii) that the errors are normally distributed. Soon, we will introduce diagnostic tools that can be used to assess to what extent these assumptions actually appear to hold, which should always been used before interpreting the outputs of hypothesis tests.

## Proof of the fact that $(n-p)\hat{\sigma}^2/\sigma^2 \sim \chi^2(n-p)$
To begin, define the residual vector

$$
\boldsymbol{r} = \boldsymbol{y} - \hat{\boldsymbol{y}} = (\boldsymbol{I}-\boldsymbol{X}(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top})\boldsymbol{y} = \boldsymbol{Qy}
$$

where we define the matrix $\boldsymbol{Q} = \boldsymbol{I}-\boldsymbol{X}(\boldsymbol{X^\top X})^{-1}\boldsymbol{X^\top}$. Note that $\boldsymbol{Q}$ is a projection matrix onto the orthogonal complement of the column space of $\boldsymbol{X}$, which will be a subspace of dimension $n-p$. This means that

$$
(n-p)\hat{\sigma}^2 = \|\boldsymbol{r}\|_2^2 = \boldsymbol{y}^\top \boldsymbol{Q} \boldsymbol{y}.
$$

Now let's write the eigenvalue decomposition $\boldsymbol{Q} = \boldsymbol{VDV}^\top$, where $\boldsymbol{V}$ is an orthogonal matrix, meaning $\boldsymbol{V^\top V} = \boldsymbol{VV^\top} = \boldsymbol{I}$. Moreover, it is a fact that since $\boldsymbol{Q}$ is a projection onto a subspace of dimension $n-p$, we will also have that

$$
\boldsymbol{D} = \text{diag}(\underbrace{1,\dots, 1}_{\times (n-p)}, \underbrace{0,\dots, 0}_{\times p}).
$$

We can use the orthogonal matrix $\boldsymbol{V}$ to construct a transformed version of our residuals: $\boldsymbol{z} = \boldsymbol{V^\top r}$.

Importantly, this $\boldsymbol{p}$ has the same norm as $\boldsymbol{r}$, since

$$
\|\boldsymbol{z}\|_2^2 = \boldsymbol{r^\top V V^\top r} = \boldsymbol{r^\top r} = \|\boldsymbol{r}\|_2^2.
$$

Moreover, it's possible to show that $\boldsymbol{z} \sim N(0, \sigma^2 \boldsymbol{D})$, and so

$$
\begin{align*}
(n-p)\frac{\hat{\sigma}^2}{\sigma^2} &= \frac{1}{\sigma^2}\|\boldsymbol{r}\|_2^2 = \frac{1}{\sigma^2}\|\boldsymbol{z}\|_2^2\\
&=\frac{1}{\sigma^2}\sum_{j=1}^{n-p} z_i^2 = \sum_{j=1}^p \left(\frac{z_i}{\sigma}\right)^2
\end{align*}
$$

Since $z_i\sim N(0,\sigma^2)$, we have that $z_i/\sigma \sim N(0,1)$, and so $(n-p)\frac{\hat{\sigma}^2}{\sigma^2}$ is exactly the sum of $n-p$ squared independent standard normal random variables, and hence follows a $\chi^2(n-p)$ distribution.
