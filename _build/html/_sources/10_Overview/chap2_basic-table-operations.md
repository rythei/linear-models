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


# Data Frames and Set Operations 

In this section, we will introduce data frames.  Topics to be covered include the following.

- Introduce basic properties of data frames.
- Show that many operations on data frames can be viewed as set operations, including how the union and intersection operations which we performed on sets correspond to outer and inner joins on data frames.
- Illustrate simple properties from the algebra of sets in data frames.

Let's start by considering rosters from three (fictional) statistics classes.

```{code-cell} 
import pandas as pd
from datasets import students
students
```


## Basic properties of a data frame

If the data frame is small, we can just print it out and look at the entire thing. However, it is often too big to do this, in which case we want to figure out basic properties of the data frame. Here are examples of how to do this.

We can determine the shape (number of rows and columns) of a data frame using the `.shape` command:

```{code-cell} 
students.shape
```

Here, we see that the `students` data frame has 41 rows and 5 columns. 

In many cases, the columns of a data frame will have meaningful names. We can inspect these names using the `.columns` command:

```{code-cell} 
students.columns
```

We can also inspect the first few rows of a data frame by using the `.head()` command:

```{code-cell} 
students.head()
```

Alternatively, we can also inspect the last few rows of a data frame by using the `.tail()` command:

```{code-cell} 
students.tail()
```

If we want to access a particular column of a data frame, we can do so using brackets `[]` and the column name of interest. (Note: we can also select a single column using the command `students['Major']`, however, this will not return a data frame back, but another object called a Series. We use the double brackets to make sure we get a data frame, which is easier to work with.)

```{code-cell} 
students[['Major']].head()
```

We can use the same method to select a subset of columns.

```{code-cell} 
students[['Major', 'Year']].head()
```


## Select and group by

When we have a dataframe, it is common to determine basic properties of the data. 
For example, we might want to find out how many students there are from each major.
We can do this with basic select and count operations, but we have to be careful, e.g., not to double count.
Let's illustrate this.

One of the most basic operations we can do on a database is to select certain rows. 
For example, we can separate out each class as its own table. 

```{code-cell} 
stat101 = students[students['Class'] == 'stat101']
stat102 = students[students['Class'] == 'stat102']
stat103 = students[students['Class'] == 'stat103']
```

For stat101, we have the following.

```{code-cell} 
stat101.head()
```

Similarly, for stat102, we have the following.

```{code-cell} 
stat102.head()
```

Now suppose we wanted to find out how many students there are from each major. 

We could try the following.

```{code-cell} 
students['Major'].value_counts()
```

However, if we look at the whole table, we realize that this doesn't give us the correct answer. 
The reason is that there are some students in multiple classes that are getting overcounted; the `.value_counts()` function simply counts the total number of rows within each major. 

Instead, we should first use the 'group by' operation. This operation _partitions_ the set of students into non-overlapping subsets. In our case, we get a different subset for each distinct major; since each student only has one major, these subsets will of course be non-overlapping. The code below first groups the rows by major and then, within each major, counts the number of unique student IDs.

```{code-cell} 
students.groupby('Major')['StudentID'].nunique()
```

Here,  the set of students has been partitioned Similarly, we can find the number of students in each year (Freshman, Sophomore, Junior, Senior) with the following:

```{code-cell} 
students.groupby('Year')['StudentID'].nunique()
```


## Merge and set operations

It is often fruitful to think of data frames as representing sets, where each row is an element, and where we can perform basic operations from the algebra of sets covered in class. 

Two of the most important operations we can do with sets are to take _unions_ and _intersections_. The union of two sets $A$ and $B$ is the new set $A \cup B = \{x \mid x\in A\text{ or } x\in B\}$, or, in words, the union is the set of all elements which are either in $A$ _or_ in $B$. Notice that taking the union of $A$ and $B$ always give us a set which is at least as big as both $A$ and $B$. The intersection of $A$ and $B$ is the set $A\cap B = \{x\mid x\in A \text{ and } x\in B\}$, in words, it's the set of all elements which are in both $A$ _and_ $B$. Notice that this will generally give us a smaller set that both $A$ and $B$. Below we illustrate the union and intersection using Venn diagrams.

![](figs/union-intersection.png)

In the language of database operations, unions are called _outer joins_, and intersections are called _inner joins_. It is easiest to see how this works through a few examples. Let us think of each of the classes (stat101, stat102 and stat103) as sets indexed by the column StudentID. 

First, let's print out the Student IDs of students in stat101:

```{code-cell} 
stat101[['StudentID']]
```

and stat102:

```{code-cell} 
stat102[['StudentID']]
```

Now, let's find all the students in stat101 $\cup$ stat102 with an outer join:

```{code-cell} 
stat101_union_102 = stat101.merge(stat102, how='outer', on='StudentID')
stat101_union_102[['StudentID']]
```

This code merges the tables stat101 and stat102, by taking every student that is in either one of these courses, and gives us a new (bigger) table back.

We can likewise find the set stat101 $\cap$ stat102 with the following inner join:

```{code-cell} 
stat101_intersection_102 = stat101.merge(stat102, how='inner', on='StudentID')
stat101_intersection_102[['StudentID']]
```

Note that to specify between an inner and outer join (i.e. an intersection or a union) using the Pandas `merge` function, we just need to use the option `how='inner'` or `how='outer'`.  The option `on='StudentID'` tells Pandas that we're using the column StudentID as the index set that we're taking the union/intersection on.


## Set complements (with data frames)

Another one of the basic operations one can do with sets is to take _complements_. To properly define the complement of a set $A$, we need to have a larger set $X$ for which $A\subseteq X$. Sometimes, this is simply all of the rows in the data frame. Then we can define $A^c = \{x\in X : x\not\in A\}$. We can use the select operations we learned earlier to calculate complements. 

For example, suppose we wanted to find stat$101^c$. To do this, we use the 'isin' function from numpy, which returns an array of True and False values describing whether each element in the first array is in the second array. For example, ```numpy.isin([1,2,3], [1,2])``` would return ```[True, True, False]```, since 1 and 2 are in ```[1,2]``` but 3 is not. We also use the function ```numpy.logical_not```, which finds the logical negation of each element in a boolean array. For example, ```numpy.logical_not([True, True, False]) = [False, False, True]```.

```{code-cell} 
import numpy

stat101_complement = students[numpy.logical_not(numpy.isin(students['StudentID'].values, stat101['StudentID'].values))]
stat101_complement
```

Similarly, we can use the below code to find the complement of the set ```freshman``` of all freshman students.

```{code-cell} 
freshman = students[students['Year'] == 'Freshman']
freshman_complement = students[numpy.logical_not(numpy.isin(students['StudentID'].values, freshman['StudentID'].values))]
freshman_complement
```


## Set associativity (with data frames)

Now that we've seen how to take unions and intersections, we can illustrate other properties, starting with the associative property. 

Recall that the _associative property_ (of unions) states that for sets $A,B$ and $C$, we have $(A\cup B)\cup C = A \cup (B\cup B)$; that is, it doesn't matter in which order we take unions.

In the cells below, we use outer joins to verify this property, by showing that (stat101 $\cup$ stat102) $\cup$ stat103 = stat101 $\cup$ (stat102 $\cup$ stat103). 

First, we compute stat 101 $\cup$ stat 102, by performing an outer join.

```{code-cell} 
stat101_union_102 = stat101.merge(stat102, how='outer', on='StudentID')
stat101_union_102[['StudentID']]
```

Next, we perform another outer join with stat103 to compute (stat101 $\cup$ stat102) $\cup$ stat103.

```{code-cell} 
stat101u102_union_103 = stat101_union_102.merge(stat103, how='outer', on='StudentID')
stat101u102_union_103[['StudentID']]
```

Next, we move on to computing the right-hand side. We start with (stat102 $\cup$ stat103):

```{code-cell} 
stat102_union_103 = stat102.merge(stat103, how='outer', on='StudentID')
stat102_union_103[['StudentID']]
```

And finally, we take an outer join with stat101 to find stat101 $\cup$ (stat102 $\cup$ stat103).

```{code-cell} 
stat102u103_union_101 = stat102_union_103.merge(stat101, how='outer', on='StudentID')
stat102u103_union_101[['StudentID']]
```

As we can see visually, the sets `stat101u102_union_103` and `stat102u103_union_101` are indeed equal! We can also check this using code:

```{code-cell} 
set(stat101u102_union_103['StudentID'].values) == set(stat102u103_union_101['StudentID'].values)
```

## Set distributivity (with data frames)

The next property which we will verify is the _distributive property_ of intersections over unions (there is also a distributive property of unions over intersection, which you will verify in the homework). This distribution property states that for sets $A,B$ and $C$, we have that $A\cap(B\cup C) = (A\cap B) \cup (A\cap C)$.

Here, we will demonstrate this property by showing that stat101 $\cap$ (stat102 $\cup$ stat103) = (stat101 $\cap$ stat102) $\cup$ (stat101 $\cap$ stat103). Noe that to do this, we need to use both inner and outer joins, to compute intersections and unions, respectively.

Let's start by computing the left-hand side, first by computing (stat102 $\cup$ stat103) with an outer join, and then taking an inner join with stat101.

```{code-cell} 
stat102_union_103 = stat102.merge(stat103, how='outer', on='StudentID')
stat101_intersection_102u103 = stat101.merge(stat102_union_103, how='inner', on='StudentID')
stat101_intersection_102u103[['StudentID']]
```

Next, let's compute the right-hand side. We do this by taking two inner joins to find the sets (stat101 $\cap$ stat102) and (stat101 $\cap$ stat103), and then taking an outer join to find the union of them.

```{code-cell} 
stat101_intersection_102 = stat101.merge(stat102, how='inner', on='StudentID')
stat101_intersection_103 = stat101.merge(stat103, how='inner', on='StudentID')
stat101n102_union_101n103 = stat101_intersection_102.merge(stat101_intersection_103, how='outer', on='StudentID')
stat101n102_union_101n103[['StudentID']]
```

As we can see visually, the sets `stat101_intersection_102u103` and `stat101n102_union_101n103` are indeed equal. However, we can again check this with code.

```{code-cell} 
set(stat101_intersection_102u103['StudentID'].values) == set(stat101n102_union_101n103['StudentID'].values)
```


## Set idempotence (with data frames)

In Section 2.1 on sets, we saw that the union operation was idempotent, meaning that for sets $A$ and $B$, we have that $A\cup B = (A\cup B) \cup B$. Another way to formalize this is to define the operation $G$, which takes in a set $A$ and returns the set $A\cup B$, namely: $G(A) = A \cup B$. Then we can equivalently state that for any set $A$, $G(G(A)) = G(A)$. Later, when we discuss functions, we will see that this means that $G$ is an identity function on its image (don't worry about this terminology for now). Here we will verify this idempotence property for unions using data frames and outer joins.

For this demonstration, we'll use the sets juniors101 (juniors in the class stat 101) and seniors102 (seniors in the class stat 102), which we define below.

```{code-cell} 
juniors101 = stat101[stat101['Year'] == 'Junior']
seniors102 = stat102[stat102['Year'] == 'Senior']
```

Let's first use an outer join to compute juniors101 $\cup$ seniors102.

```{code-cell} 
juniors101_u_seniors102 = juniors101.merge(seniors102, how='outer', on='StudentID')
juniors101_u_seniors102[['StudentID']]
```

Next, let's verify the idempotence property of the union with seniors, namely that juniors101 $\cup$ seniors102 = (juniors101 $\cup$ seniors102) $\cup$ seniors102:

```{code-cell} 
juniors101_u_seniors102_u_seniors102 = juniors101_u_seniors102.merge(seniors102, how='outer', on='StudentID')
juniors101_u_seniors102_u_seniors102[['StudentID']]
```

As we can see visually, the two sets consist of the same student ID are the same. We can also verify this using code:

```{code-cell} 
set(juniors101_u_seniors102_u_seniors102['StudentID'].values) == set(juniors101_u_seniors102['StudentID'].values)
```

In the homework, we will see that a similar idempotence property also holds for intersections.
