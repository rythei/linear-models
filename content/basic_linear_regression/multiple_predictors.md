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

# Linear regression with multiple predictors

So far, we've studied only "simple" linear regression, where we predict a response $y$ using a single predictor variable $x$ using a model of the form

$$
y_i = \alpha + \beta x_i + \varepsilon_i.
$$

However, it is frequently the case that we may observe multiple variables $x_{i1},\dots, x_{ip}$ whose relationship with $y_i$ we would like to model. In this case, a natural extension is to consider the model

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i.
$$

In this model,
