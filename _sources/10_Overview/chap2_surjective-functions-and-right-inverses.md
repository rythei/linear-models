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

# Surjective Functions and Right Inverses

A _surjective_ function is a mapping from a set $A$ to set $B$ such that: for every $b\in B$, there exists an $a\in A$ such that $f(a) = b$. 
That is, $f$ is surjective if every element of the co-domain $B$ is mapped to by some element of the domain $A$.

Informally, if $f : A \rightarrow B$ is surjective, then this means that $B$ must be ``smaller'' (not larger than) $A$, otherwise there would not be enough elements of $A$ to map to each distinct element of $B$.
Like with our intuition from injective functions, this is true for discrete sets and functions on discrete sets.
Although it is not precisely true for infinite sets (such as the set of integers or the set of real numbers or the set of points on the Euclidean plane), it is not a bad intuition.

In this section, we will investigate surjective functions. 
We will use the table operations we've seen before, as well as simple numerical examples.
In both cases, we will see that surjective functions always have special complementary functions called _right inverses_.

We will again work with the `western_cities` dataset

```{code-cell} 
from datasets import western_cities
western_cities.columns = ['City', 'Population', 'State']
western_cities.head()
```

We will also use again the mapping from states to state capitals that we used in the last notebook

```{code-cell}
from datasets import states as state_capitals
state_capitals = state_capitals[["State", "Capital"]]
state_capitals
```

## An example of a surjective function

As discussed above, our intuition about surjective functions tells us that for a function $f$ to be surjective, its co-domain needs to be "smaller" than its range. Let's look again at the size of each of the three sets contained in the `western_cities` data frame:

```{code-cell}
print('City has %i unique values' % western_cities["City"].nunique())
print('Population has %i unique values' % western_cities["Population"].nunique())
print('State has %i unique values' % western_cities["State"].nunique())
```

Again, `State` is much smaller than either `City` or `Population`. Therefore, we will consider a function $f$ which maps to the co-domain `State`, and for simplicity we will choose the domain to be `City`. Indeed, it is easy to see that the function 

```{code-cell}
city_to_state = western_cities[["City", "State"]]
city_to_state
```

mapping `City` to `State` is indeed surjective: every state in the set `State` is mapped to by some city in the set `City`. 

Since the function $f$ is surjective, we have a seen in lecture that it must have at least one complementary function called a _right inverse_. In our case, a right inverse is a function $g:$ `State` $\to$ `City`, such that $f\circ g$ is the identity function on the set `State`. In fact, since `City` is _strictly_ larger than `State` there must be more than one right inverse. In what follows, we will see a few examples of right inverses.

## Constructing right inverses

Let's think about what properties a function $g:$ `State` $\to$ `City` must have in order to be a right inverse for our function $f$ mapping each city to its state. Since we need $f\circ g$  to be the identity function on `State`, we need that for any state $s$, $f(g(s))=s$. In particular, we need that for each state, $g$ maps this state to some city which $f$ will map back to the same state. Since $f$ just maps each city to its state, $g(s)$ can output _any_ city in the state $s$ -- importantly, it does not matter which city in the state $g$ outputs. This is precisely why there will be more than one right inverse for $f$. 

Let's look at a few example of right inverses. First, let's consider the function $g_1:$ `State` $\to$ `City` which maps each state to its capital. 

```{code-cell}
g1 = state_capitals[["State", "Capital"]]
```

Intuitively $g_1$ should satisfy our condition for being a right inverse of $f$ , since every state capital of course belongs to the state of which it is the capital. However, we can also verify this by computing $f\circ g_1$ with a left merge:

```{code-cell}
state_to_state1 = g1.merge(city_to_state, left_on="Capital", right_on="City", how="left")[["State_x", "State_y"]]
state_to_state1.columns = ["State", "(f o g1)(State)"]
state_to_state1
```

Indeed, we see that $f\circ g_1$ is indeed the identity function on `State`, and so $g_1$ is a right inverse for $f$. As we said above, however, this is not the only right inverse we could have constructed: _any_ function $g$ which maps a state to some city in that state will suffice. For example, let's consider $g_2:$ `State` $\to$ `City`, which maps each state to its most populous city:

```{code-cell}
temp = western_cities.loc[western_cities.groupby("State")["Population"].idxmax()]
g2 = temp[["State", "City"]].reset_index(drop=True)
g2
```

Note that $g_2$ here is indeed a distinct functiom from $g_1$ (for example, Sacramento is the capital of California, but Los Angeles is the most populous). We can verify that $g_1$ and $g_2$ are different with the following:

```{code-cell}
g1.equals(g2)
```

On the other hand, we can easily verify that $g_2$ is indeed a valid right inverse for $f$:

```{code-cell}
state_to_state2 = g2.merge(city_to_state, on="City", how="left")[["State_x", "State_y"]]
state_to_state2.columns = ["State", "(f o g2)(State)"]
state_to_state2
```

As we expected $f\circ g_2$ does indeed give us the identity function on `State`, and thus $g_2$ is another distinct right inverse for $f$.

## A numerical example

To finish this section, we will give a simple numerical example of a surjective function, which will complement the examples given with the data frame above.

Here, we consider a function $f:\mathbb{R}\to\mathbb{Z}$, where $\mathbb{Z} = \{\dots, -2,-1,0,1,2,\dots\}$ is the set of integers, and $\mathbb{R}$ is the set of real numbers. While $\mathbb{Z}$ and $\mathbb{R}$ are both infinite sets, we can intuitively see that $\mathbb{R}$ is 'larger', in the sense that there are many elements of $\mathbb{R}$ which are not in $\mathbb{Z}$ (this intuition of $\mathbb{R}$ being larger can in fact be formalized by proving that there is no surjective function from $\mathbb{Z}$ to $\mathbb{R}$). 

We define the function $f$ as follows: for any integer $x\in \mathbb{R}$, let $f(x)$ be an integer obtained by rounding $x$ to the nearest integer. (Note this is essentially the same function we used as a left inverse in the previous section on injective functions).

```{code-cell}
import numpy as np

def f(x):
    return np.rint(x)
```

Let's see a few examples of this function's output

```{code-cell}
print('f(2.411) = %i' % f(2.411))
print('f(-4.89) = %i' % f(-4.89))
print('f(19.21) = %i' % f(19.21))
```

This function is indeed surjective: for any integer $z$, the real number $x = z-.1$ (for example) will always give $f(x)=z$. Of course, the real numbers $z+.1$ or $z$ or $z+.2119083$ would also satisfy this condition. Since $f$ is surjective, it must have a right inverse, i.e. a function $g$ such that $f\circ g$ is equal to the identity function on the integers. How can we construct such a right inverse? We need a function $g$ such that for any integer $z$ , $g(z)$ returns a real number such that $f(g(z)) = z$. As our argument above indicated, there are many such functions $g$ that would satisfy this! For example, we could consider the function $g_1(z)=z-.1$: 

```{code-cell}
def g1(z):
    return z-.1
```

Now we can verify on a few examples that this does indeed give us a valid right inverse:

```{code-cell}
print('f(g1(2)) = %i' % f(g1(2)))
print('f(g1(-4)) = %i' % f(g1(-4)))
print('f(g1(19)) = %i' % f(g1(19)))
```

Of course, we also could have considered the function $g_2(z) = z+.2119083$:

```{code-cell}
def g2(z):
    return z+.2119083
```

Then $f\circ g_2$ also gives us the identity function on the integers

```{code-cell}
print('f(g2(2)) = %i' % f(g2(2)))
print('f(g2(-4)) = %i' % f(g2(-4)))
print('f(g2(19)) = %i' % f(g2(19)))
```

In this section, we've looked at examples of surjective functions, and seen that surjective functions always come with at least one -- but often more than one -- special complementary functions called a _right inverses_. The concepts of injective/surjective functions, and left/right inverses will play an important roll in our study of linear algebra later on.

## Idempotence and right inverses

One might be tempted to think that if we flipped the order of composition, we would also obtain an identity: namely that $g\circ f$ might be the identity function on $\mathbb{R}$. 
However, this cannot be the case: for a real number $x$, $f(x)$ gives the nearest integer, which $g$ is not guaranteed to map back to the same input $x$. 
On the other hand, $g\circ f$ does have another special property: it is always _idempotent_, meaning $(g\circ f)^2 = g\circ f$. 
It is easy to see this algebraically, using the fact that $f\circ g$ is the identity on the integers:
\begin{equation*}
(g\circ f)^2 = (g\circ f)\circ (g\circ f) = g\circ (f\circ g)\circ f = g\circ \text{Id}_{\mathbb{Z}}\circ f = g\circ f
\end{equation*}
We can also verify this property numerically for our example.

```{code-cell}
print('(g2 o f)(-5.21) = %f' % g2(f(-5.21)))
print('(g2 o f)^2(-5.21) = %f' % g2(f(g2(f(-5.21)))))
print('')
print('(g2 o f)(4.68) = %f' % g2(f(4.68)))
print('(g2 o f)^2(4.68) = %f' % g2(f(g2(f(4.68)))))
print('')
print('(g2 o f)(13.74) = %f' % g2(f(13.74)))
print('(g2 o f)^2(13.74) = %f' % g2(f(g2(f(13.74)))))
```

