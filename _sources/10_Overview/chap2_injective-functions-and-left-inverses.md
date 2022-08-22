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

# Injective Functions and Left Inverses

An _injective_ function is a mapping from a set $A$ to a set $B$ such that: if $a'\neq a$ then $f(a)\neq f(a')$. 
That is, $f$ is injective if and only if it maps distinct elements of the domain $A$ to distinct elements of the co-domain $B$. 

Informally, if $f : A \rightarrow B$ is injective, then this means that $A$ must be ``smaller'' (not larger than) than $B$, since if $A$ were larger than $B$, then at least two elements of $A$ would have to map to the same element of $B$.
This intuition of larger/smaller is formally true for discrete sets and functions on discrete sets.
Although it is not precisely true for infinite sets (such as the set of integers or the set of real numbers or the set of points on the Euclidean plane), it is not a bad intuition.

In this section, we will investigate injective functions.
We will use the table operations we've seen before, as well as simple numerical examples.
In both cases, we will see that injective functions always have special complementary functions called _left inverses_.

In the next few sections of the workbook, we will work with datasets containing information about several western cities, which we will introduce now.

The `western_cities` dataset contains information about 251 cities located in the western United States.

```{code-cell}
from datasets import western_cities
western_cities.columns = ['City', 'Population', 'State']
western_cities.head()
```

We will also want a bit more information about these western states. Importantly, we will use information about the capitals of each of these states. This information is stored in the `state_capitals` dataset. 

```{code-cell}
from datasets import states as state_capitals
state_capitals = state_capitals[["State", "Capital"]]
state_capitals
```

## An example of an injective function

Recall our intuition that injective functions map "smaller" sets to "larger" ones. In the case of the `western_cities` dataset, we have three potential sets which we could work with: `City`, `Population` and `State`. Let's see how many unique values are in each.

```{code-cell}
print('City has %i unique values' % western_cities["City"].nunique())
print('Population has %i unique values' % western_cities["Population"].nunique())
print('State has %i unique values' % western_cities["State"].nunique())
```

We see that clearly `State` is the smallest set, while `City` and `Population` are much larger. 
Of course, this is because there are many cities within each state. 
Thus, to construct an injective function, let's choose `State` to be our domain, and either `Population` or `City` to be our co-domain. 
For this section, we will consider the function $f$ from `State` to `City` which maps each state to its capital. 
This function is already given to us in the `state_capitals` dataframe:

```{code-cell}
state_capitals
```

Clearly, this function is injective, since each distinct state is mapped to a distinct city. 
However, that this function is not _surjective_, since the set `Capital` is a strict subset of the set of all western cities `City`.
That is, there are some cities that are not the capital of a state. 

Since the function $f$ is injective, we've seen in class that it must have at least one complementary function called a _left inverse_. 
In our case, a left inverse is a function $g:$ `City` $\to$ `State`, such that $g\circ f$ is the identity function on the set `State`. 
In fact, since `City` is _strictly_ larger than `State` there must be more than one left inverse. 
In what follows, we will see a few examples of left inverses.

## Constructing left inverses

Let's think about what properties a function $g:$ `City` $\to$ `State` must have in order to be a left inverse for our function $f$ mapping each state to its capital. Since we need $g\circ f$ to be the identity function on `State`, we need that for any state $s$, $g(f(s)) = s$. In particular, we need that for each state capital, $g$ maps this capital back to the state that it is the capital of. Importantly, _it does not matter what $g$ does to cities which are not in the set `Capital`_. As we will see, it is for this reason that there will be several left inverses for $f$.

Let's use this characterization to give one example of a left inverse. Consider the function $g_1$: `City` $\to$ `State` mapping each city to the state that it is in. We can obtain this function straightforwardly from the `western_cities` data frame:

```{code-cell}
g1 = western_cities[["City", "State"]]
g1.head()
```

Of course, every state capital is in the state of which it is the capital, and so intuitively $g_1$ should be a valid left inverse for $f$. Let's verify this with code by computing $g_1\circ f$ using a left join. (Note: here we need to specify a value of `left_on` and `right_on` when we do the join, because the column names are different for the two tables.)

```{code-cell}
state_to_state1 = state_capitals.merge(g1, left_on="Capital",right_on="City", how="left")[["State_x", "State_y"]]
state_to_state1.columns = ["State", "(g1 o f)(State)"]
state_to_state1
```

Indeed, we see that $g_1\circ f$ is indeed the identity function on `State`, and so $g_1$ is a left inverse for $f$. As we said above, however, this is not the only left inverse we could have constructed: we only care about what the left inverse does on the elements of the _range_ of $f$, which in this case is the set `Capital`. 

To come up with another example, let's alter $g_1$ by changing some of the states that cities get mapped to. It doesn't matter which cities we change, as long as it isn't one of the 6 cities in the subset `Capital`. 

```{code-cell}
g2 = western_cities[["City", "State"]]
g2.loc[g2["City"] == "Tacoma", "State"] = "Colorado" # map Tacoma to Colorado
g2.loc[g2["City"] == "Pomona", "State"] = "Hawaii" # map Pomona to Hawaii
g2
```

Here we decided to map the city Tacoma (which is in Washington) to Colorado, and Pomona (which is in California) to Hawaii. We can verify that `g1` and `g2` are indeed different:

```{code-cell}
g1.equals(g2)
```

However, if we compute the composition $g_2\circ f$, we see that it still gives us the identity function on `State`, and therefore that $g_2$ is another valid left inverse for $f$:

```{code-cell}
state_to_state2 = state_capitals.merge(g2, left_on="Capital",right_on="City", how="left")[["State_x", "State_y"]]
state_to_state2.columns = ["State", "(g2 o f)(State)"]
state_to_state2
```

## A numerical example

To finish this section, we will give a simple numerical example of an injective function, which will complement the examples given with the data frame above.

Here, we consider a function $f:\mathbb{Z}\to\mathbb{R}$, where $\mathbb{Z} = \{\dots, -2,-1,0,1,2,\dots\}$ is the set of integers, and $\mathbb{R}$ is the set of real numbers. While $\mathbb{Z}$ and $\mathbb{R}$ are both infinite sets, we can intuitively see that $\mathbb{R}$ is 'larger', in the sense that there are many elements of $\mathbb{R}$ which are not in $\mathbb{Z}$ (this intuition of $\mathbb{R}$ being larger can in fact be formalized by proving that there is no surjective function from $\mathbb{Z}$ to $\mathbb{R}$). 

We define the function $f$ as follows: for any integer $z\in \mathbb{Z}$, let $f(z)$ be a random number in the interval $(z-1/2, z+1/2)$. Let's define a python function which does this.

```{code-cell}
import numpy as np
np.random.seed(0) # set random seed for reproducibility
r = np.random.uniform(low=-.5, high=.5)

def f(z):
    return z + r
```

Let's see a few examples of this function's output

```{code-cell}
print('f(-5) = %f' % f(-5))
print('f(2) = %f' % f(2))
print('f(14) = %f' % f(14))
```

This function is indeed injective: if $z\neq z'$, then the intersection of the sets $(z-1/2, z+1/2)$ and $(z'-1/2, z'+1/2)$ must be empty. Since $f$ is injective, it must have a left inverse, i.e. a function $g$ such that $g\circ f$ is equal to the identity function on the integers. How can we construct such a left inverse? We need a function $g$ such that for any integer $z$ and any value $f(z) \in (z-1/2, z+1/2)$, we have $g(f(z)) = z$. A natural way to construct such a function $g$ is simply by rounding to the nearest integer. To do this in python, we can use the `numpy` function `rint`. Let's define this function below:

```{code-cell}
def g(y):
    return np.rint(y)
```

Now we can verify on our examples that this does indeed give us a valid left inverse:

```{code-cell}
print('g(f(-5)) = %i' % g(f(-5)))
print('g(f(2)) = %i' % g(f(2)))
print('g(f(14)) = %i' % g(f(14)))
```

As expected, this gives us the identity function on the integers!

As we've seen, injective functions must always have left inverses, and in many cases have more than one of them. Intuitively, we can think about injective functions as being functions whose domain is "smaller" than their co-domain -- and it is this property that allows us to construct a variety of left inverses. As we will see in the next section, there is an analogous concept for functions whose domain is "larger" than their co-domain, and that this is related to the idea of surjective functions and right inverses.

## Idempotence and left inverses

One might be tempted to think that if we flipped the order of composition, we would also obtain an identity: namely that $f\circ g$ might be the identity function on $\mathbb{R}$. 
However, this cannot be the case: for a real number $x$, $g(x)$ gives the nearest integer, which $f$ is not guaranteed to map back to the same input $x$. 
On the other hand, $f\circ g$ does have another special property: it is always _idempotent_, meaning $(f\circ g)^2 = f\circ g$. 
This is easy to see this algebraically, using the fact that $g\circ f$ is the identity on the integers:
\begin{equation*}
(f\circ g)^2 = (f\circ g)\circ (f\circ g) = f\circ (g\circ f)\circ g = f\circ \text{Id}_{\mathbb{Z}}\circ g = f\circ g
\end{equation*}
We can also verify this property numerically for our example.

```{code-cell}
print('(f o g)(-5.21) = %f' % f(g(-5.21)))
print('(f o g)^2(-5.21) = %f' % f(g(f(g(-5.21)))))
print('')
print('(f o g)(4.68) = %f' % f(g(4.68)))
print('(f o g)^2(4.68) = %f' % f(g(f(g(4.68)))))
print('')
print('(f o g)(13.74) = %f' % f(g(13.74)))
print('(f o g)^2(13.74) = %f' % f(g(f(g(13.74)))))
```

