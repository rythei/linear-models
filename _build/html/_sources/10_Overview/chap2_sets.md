---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Sets and Set Algebra

In this section, we will cover basic aspects of sets in python. We will then use this to illustrate some of the simple properties that we’ve seen from the algebra of sets and the algebra of transformations.

* Introduction to sets and sets in python.

* Illustration of basic operations, including subset, superset, intersections, and unions.

* Illustration of more adanced notions, including: De Morgan's Laws; partitions versus non-partitions; distributive rule for intersection over union and distributive rule for unions over intersections; and how performing union/intersection with a given set is an idempotent operation.



## Introduction to Sets

Sets are collections of objects (without repetitions). Set theory dates back to the great German mathematician Georg Cantor (1845-1918) and forms the foundation of modern mathematics.

We can describe a set by listing all its elements explicitly, for example

$$
M := \{1,2,3,4,5,6\}
$$

defines a set $M$. It is convention to use capital letters (e.g., A, B, C) to represent sets. The objects in a set are called elements and are represented by lower case letters (e.g., x, y, z). 

* We say $x$ is an element in $X$, or short $x\in X$, if $x$ is contained in the set $X$. 
* Otherwise we say that $x$ is not an element of $X$, or short $x\not\in X$.

For example 2 is an element in $M$, i.e., $2\in M$. In contrast, 7 is not an element in $M$, i.e., $7\not\in M$. 

Our set $M$ contains 6 elements. We also say that the set $M$ has a cardinality of 6 and we can express this concisely as $|M|=6$.

* If a set is not infinite, then the cardinality of a set is a measure of the "number of elements" of the set.

Note, here we use the logical symbol $:=$ to define $M$ as an object that is equal to the set $\{1,2,3,4,5,6\}$. In contrast, the symbol $=$ expresses that the object on the left side is equal to the object on its right side. We will often be a bit sloppy and simply use $=$ to define a set, but it is good to keep in mind that there can be a difference between a definition and an equality. This is in particular import when we are writing code, e.g., `x=5` assigns the value 5 to `x` and this is different from the logical operation `x==5` which compares whether `x` is equal to 5. 

We can also define an infinite set, i.e., a set that has no last element as

$$
\mathbb{N}^* := \{1,2,3,...\}.
$$

This set contains all positive integers, i.e., it is the set of all natural numbers without 0. (We call $\{\}$ the set brackets, and the dots $...$ are called ellipsis.)

In many situations it would be tedious to list all elements. But, we can also define sets by stating properties that characterize its members. For example we can define $M$ more concisely as

$$
M := \{x \in \mathbb{N}^* | x\le 6\}.
$$

Here we say that the element $x$ is a member of $\mathbb{N}^*$ such that $x$ is smaller or equal to 6. When we define sets by stating properties we use typically the following notation 

$$
\{\text{membership}|\text{properties}\}
$$

or 

$$
\{\text{pattern}|\text{membership}\}.
$$

You have already seen an example for the first set notation, and here is an example for the second notation

$$
X := \{2n | n \in \mathbb{N}^* \}.
$$

This set contains all natural numbers that are divisible by 2.


## Sets in Python

Python has a rich number of build-in set objects. Like mathematical sets, Python sets have the following properties:

* Sets are unordered.
* Set elements are unique.

We can define a set in Python using the curly braces.

```python
M = {1,2,3,4,5,6}
print(M)
```

Let's check the type of this new object.

```python
type(M)
```

To compute the cardinality of this set, we can use the `len()` function.

```python
len(M)
```

We can also define a set by using the Python set function.

```python
set([1,2,3,4,5,6])
```

More compactly, we can create the set as

```python
{x for x in range(7) if x>0}
```

Now let's create a few more sets.

```python
X = {6,5,4,3,2,1}
Y = {1,2,3}
Z = {x for x in range(10) if x>0}
```

We can ask which of these sets is a subset of M. Recall, a set A is considered a subset of another set B if every element of A is in B.

```python
X.issubset(M)
```

```python
Y.issubset(M)
```

```python
Z.issubset(M)
```

Of course, Z is not a subset of M. But it is a superset, since every element of M is in Z, right?

```python
Z.issuperset(M)
```

Finally, we can also check whether two sets are equal.

```python
M == X
```

```python
M == Z
```

## Basic Operations on Sets

In order to work with sets we need to introduce some set operations first. To illustrate these operations, we define the sets $A=\{2,3,4,5\}$, $B=\{4,5,6\}$, and $\Omega=\{1,2,3,...9\}$.

Here is a summary of basic operations.

* Union: $A \cup B := \{x | x\in A \,\,\, \text{or} \,\,\, x\in B\} = \{x\in A | x\in B\}$.
  * The union $A \cup B$ consists of the elements that appear in $A$ or $B$ or both.
  * Example: $A \cup B := \{2,3,4,5,6\}$ 
 
 



* Intersection:  $A \cap B := \{x| x\in A \,\,\, \text{and} \,\,\, x\in B\}$.
 * The intersection $A \cap B$ consists of elements that appear in both A and B.
 * Example: $A \cap B := \{4,5\}$ 
            


* Complement:  $A^C := \{x \in \Omega | x\not\in A\}$, where $A \subseteq \Omega$.
 * The complement $A^C$ consists of all the elements of $\Omega$ that are not in A.
 * Example: $A^C := \{1,6,7,8,9\}$ 
 


* Relative Complement:  $A\backslash B := \{x \in A | x\not\in B\}$.
 * The relative complement $A\backslash B$ consists of all the elements in A but not in B.
 * Example: $A\backslash B := \{2,3\}$ 
 


We say two sets are disjoint if there are no element in common between the two sets, i.e., $D\cap E = \emptyset$. In our example A and B are not disjoint. 


Now let's try to repeat this example in Python. First we have to define the sets.

```python
omega = {x for x in range(10) if x>0}
print(omega)
A = {x for x in omega if x>1 and x<6}
print(A)
B = {x for x in omega if x>3 and x<7}
print(B)
```

First, we compute the union $A \cup B$.

```python
A.union(B)
```

Next we compute the intersection $A \cap B$

```python
A.intersection(B)
```

Next, we compute the relative complement $A \backslash B$.

```python
A.difference(B)
```

We can also use the difference function to compute the complement. 

```python
omega.difference(A)
```

## De Morgan's laws

Recall, that De Morgan's Laws relate the intersection and union of sets through complements. 

* $(A\cup B)^c = A^c \cap B^c$ (law of union)
* $(A\cap B)^c = A^c \cup B^c$ (law of intersection)


Let's try to verify these laws for our above example. 

```python
omega.difference(A.union(B))
```

This set is equal to

```python
(omega.difference(A)).intersection(omega.difference(B))
```
Hence, the law of unions holds for our example. You will verify the law of intersections for this example as a homework problem.


## Partitions
Given a set, one often wants to split it into representative pieces. A partition of a set is such a splitting of the elements of the set into non-empty subsets, in such a way that every element is included in exactly one subset. Here is an example. 

```python
A = {x for x in range(1,11)}
B = {1,3,5,7,9}
C = {2,4,6,8,10}
```

The set B and C partition the set A, i.e., the union of B and C contains all elements that are in A, while the intersection of B and C is the empty set.

```python
B.union(C) == A
```

```python
B.intersection(C) 
```

Partitions are important if want to count the elemens of sets. The cardinality of A is

```python
len(A)
```

Since B and C are partitions, we have $|A| = |B| + |C|$

```python
len(B) + len(C)
```

Note, that this does not work for the following two sets, since they do not perform a partition of A.

```python
B = {1,2,3,4,5,6,7}
C = {5,6,7,8,9,10}
```

Again, the union of B and C contains all elements that are in A.

```python
B.union(C) == A
```

But, the intersection yield a non-empty set.

```python
B.intersection(C) 
```

Now, things go wrong if we count the elements. 

```python
len(B) + len(C)
```

To get the correct answer, we need to subtract the intersection of A and B, i.e. $|A| = |B| + |C| - |B\cap C|$

```python
len(B) + len(C) - len(B.intersection(C))
```

## Distributive Rule


We have the following two rules:
* $A\cap (B\cup C) = (A\cap B) \cup (A\cap C)$ (intersection distributed over union)
* $A\cup (B\cap C) = (A\cup B) \cap (A\cup C)$ (union distributes over intersection)


Let's varify these rules for our example.

```python
A.intersection(B.union(C)) == (A.intersection(B)).union(A.intersection(C))
```

```python
A.union(B.intersection(C)) == (A.union(B)).intersection(A.union(C))
```

## Idempotence of unions

One property of the union operation is that it is idempotent.  We know that, e.g., $A \cup A = A$, but we can use this to get more interesting results.  In particular, this means that when we apply a union with a set multiple times, the result is the same as if we apply it once.  

Here, we will demonstrate this with the sets $A=\{1,2,\dots,20\}, B = \{21,22,\dots,40\}$.  We will observe that $A\cup B = (A\cup B)\cup B$.  

Similarly, you should be able to verify on your own that $A\cup B = A \cup (A\cup B)$.

```python
A = {x for x in range(1,21)}
print(A)
B = {x for x in range(21,41)}
print(B)
```

As we would expect, $A\cup B = \{1,2,\dots, 39, 40\}$:

```python
A_union_B = A.union(B)
print(A_union_B)
```

Now let's apply the union with $B$ again, and see if this changes the set.

```python
A_union_B_union_B = A_union_B.union(B)
print(A_union_B == A_union_B_union_B)
```

Indeed, the set is unchanged from applying the union with $B$ twice. In the homework, we will see that a similar idempotence property holds for intersections.
