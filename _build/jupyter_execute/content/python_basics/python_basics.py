#!/usr/bin/env python
# coding: utf-8

# # The Basics
# 
# 
# ## Printing out variables
# 
# You can think about variables as a container that stores a value. In Python it is really easy to define a variable and assign a value to it.

# In[1]:


s = "hello world" # assign "hello world" to variable b


# Here, the variable `s` has been assigned a string. A string is a sequence of characters that the computer understands to be text. We will come back to think in detail about how to
# use and manipulate strings below.
# 
# ```{note}
# It is good practice to comment each line of your code and you can do so by writing a comment after the "#" symbol. Comments will greatly help you to understand your code that you may have written a few month ago, and it will be also appreciated by others who you share your code with.
# ```
# 
# You can use the `print()` function to ask the computer to print whatever has been stored in a variable.

# In[2]:


print(s)


# The print functions often comes in handy to print some intermediate results that can help to debug your code.
# 
# 
# 
# 
# 
# ## Different Types of Variables
# 
# ### Numbers
# 
# Next, we assign numeric values (integers and floats) to the variables `a`, `b`, and `c`.

# In[3]:


a = 1 # assign the integer 1 to variable a
b = 2.5	# assign 2 to variable a
c = 3.2	# assign 3 to variable c


# You can also add variables (as you would expect).

# In[4]:


print('a+b:', a+b) # add b to a
print('a+c:', a+c) # add c to a


# Similar you can subtract, multiply, and divide variables. You can try this yourself.

# In[5]:


# multiply a and b
# subtract c from a


# You can see what type a variable is, which is often helpful to diagnose problems.

# In[6]:


print('a is of type', type(a))
print('b is of type', type(b))
print('c is of type', type(c))


# ### Strings
# 
# 
# Strings are text things

# In[7]:


my_name = 'Michael Mahoney'
print('My name is', my_name)
print('my_name is of type', type(my_name))


# The `+` operator will concatenate two strings.

# In[8]:


my_name = 'Michael'
print('Just first name:', my_name)
my_name += 'Mahoney'
print('Concatenated:', my_name)


# We can also use Python's slicing operator on strings:

# In[9]:


print('Index 1 of my_name is', my_name[1])
print('The first 2 characters of my_name are', my_name[:2])
print('The last 3 characters of my_name are', my_name[-3:])


# ### Lists
# 
# Lists are just a list of things that usually can be operated on in limited ways. I.e., they are ***NOT*** mathematical vectors in a vector space. We will see later how we can create data types backed by lists, but contain additional operations, that are a much better representation for a vector.

# In[10]:


my_first_list = [11,22,33]
print('my_first_list:', my_first_list)
print('my_first_list is of type', type(my_first_list))


# The `+` operator concatenates two lists. Note that concatenation using `+` is non-destructive, i.e., the original list does not change.

# In[11]:


my_first_list + [44]
print('my_first_list after bad concat:', my_first_list)


# Let's try this again. We can use `+=` to create a temporary variable equivalent to `my_first_list + [44]`, then assign that back to `my_first_list`, consistent with the use of `+=` for numbers.

# In[12]:


my_first_list += [44]
print('my_first_list after good concat:', my_first_list)


# We can also use `.append()` to mutate (change) a list. While using `+` creates a new list, `.append` does not create a new list, and instead changes an existing list by adding an item to it.

# In[13]:


my_first_list.append(55)
my_first_list.append(66)
print('my_first_list after append:', my_first_list)


# We can use `len()` to get the length of a list.

# In[14]:


print('The length of my_first_list is', len(my_first_list))


# We can also use the indexing operator on lists.

# In[15]:


print('The first element of my_first_list is', my_first_list[0])


# We can also use the slicing operators on lists.
# 
# **Note:** when slicing ([:]), the last element is NOT included in the slice.

# In[16]:


print('The second through fourth elements of my_first_list are', my_first_list[1:4])


# We can also leave the number before the colon blank to slice from the start.

# In[17]:


print('The first through fourth elements of my_first_list are also', my_first_list[:4])


# Note how this is equivalent to:

# In[18]:


print('The first through fourth elements of my_first_list are', my_first_list[0:4])


# We can also leave the number after the colon blank to slice until the end.

# In[19]:


print('The third through last elements of my_first_list are', my_first_list[2:6])


# Note how this is equivalent to:

# In[20]:


print('The third through last elements of my_first_list are also', my_first_list[2:])


# ### Tuples
# 
# Tuples are similar to lists. However, there are some subtle differences. The main difference is that tuples are [IMMUTABLE](https://en.wikipedia.org/wiki/Immutable_object). This means, that after creating a tuple, you cannot modify its elements, nor can you append to or delete from it.
# 
# Try to figure out how to use a list first, then once you're used to the concept, you'll gradually realize some use cases make more sense as a tuple.

# In[21]:


my_first_tuple = (1, 2, 3)
print('my_first_tuple:', my_first_tuple)
print('my_first_tuple is of type', type(my_first_tuple))


# What happens if we try to modify a tuple?
# 
# ```
# my_first_tuple[0] = 5
# ```
# 
# However, you can still concatenate a tuple with another tuple. This is because doing so creates a new tuple.

# In[22]:


my_second_tuple = my_first_tuple + (4, 5, 6)
print('my_second_tuple:', my_second_tuple)


# You can also use the `+=` operator as a shortcut

# In[23]:


my_first_tuple += (4, 5, 6)
print('my_first_tuple after concat:', my_first_tuple)


# ### Dictionaries
# 
# Dictionaries: key-value pairs. (The key is the "word", the value is the "definition".) We won't spend as much time on this, but it's a good data structure to know.

# In[24]:


my_first_dict = {'name':'Michael', 'score1':100}
print('my_first_dict:', my_first_dict)
print('my_first_dict is of type', type(my_first_dict))


# Similar to lists, we can index into dictionaries to modify and retrieve values.

# In[25]:


my_first_dict['score1'] = 99
print('my score1 after modification:', my_first_dict['score1'])


# Unlike lists, you can "index" into a key that does not exist yet! What this does, is you can create a new key in the dictionary and assign a value to it right away. If they key exists, you will overwrite it as you would expect.
# 
# *Note:* the order of the `key:value` pairs do not matter in a dictionary, and is not guaranteed. This means, you could have `score2` appearing before `score1` when printing, even though you added `score2` "later".

# In[26]:


my_first_dict['score2'] = 90
print('my_first_dict after adding new key:', my_first_dict)
print('my score2:', my_first_dict['score2'])


# You can also retrieve a list of keys and values separately from the dictionary. This is potentially useful if you want to iterate over all the keys or all the values.

# In[27]:


print('Keys in my_first_dict:', my_first_dict.keys())
print('Values in my_first_dict', my_first_dict.values())


# Note that they are of weird types, `dict_keys` and `dict_values`, instead of something familiar like a `list`.

# In[28]:


print('Type of my_first_dict.keys():', type(my_first_dict.keys()))
print('Type of my_first_dict.values():', type(my_first_dict.values()))


# We can turn this into a list by using `list()` if you require a list. However, if you are only using `.keys()` or `.values()` for iteration, don't worry, because `dict_keys` and `dict_values` are still perfectly iterable.

# In[29]:


print('List of keys in my_first_dict:', list(my_first_dict.keys()))
print('List of values in my_first_dict:', list(my_first_dict.values()))


# ## String formatting
# 
# As you may or may not have noticed above, if you pass a comma separated list of things to the print function `print`, it will print them all with a space between.

# In[30]:


my_variable = 10
template = 'The value of my variable is {}'
print(template, my_variable)


# Hm, that looks a bit weird. What we did above was ignore the silly `{}`, and have Python naturally add the space.

# In[31]:


template2 = 'The value of my variable is'
print(template2, my_variable)


# But what if we want to insert the value of a variable to the middle of a string? Or what if we don't want spaces before and after variables? This is where the `{}` comes in. We can use it in conjunction with the `.format` method of strings to insert variables wherever we want in a string.

# In[32]:


s = template.format(my_variable)
print(s)


# We can use this new power to fix our grammar: adding a period to the end of the sentence!

# In[33]:


template3 = 'The value of my variable is {}.'
print(template3.format(my_variable))


# Supposed we wanted to concatenate first and last names, but with a space in between. We can use the `+` operator like before, but this might be a bit annoying and error-prone

# In[34]:


first = 'Michael'
last = 'Mahoney'
full1 = first + ' ' + last
print(full1)


# We can make this cleaner using string formatting. If we have multiple placeholders (the `{}` things), Python will insert the variables passed to `.format()` in order.

# In[35]:


full2 = '{} {}'.format(first, last)
print(full2)


# However, we can also mix up the order a bit! Simply specify which argument you want in which placeholder. **Note:** zero indexing as usual.

# In[36]:


full3 = '{1}, {0}'.format(first, last)
print(full3)


# You can also have repeats.

# In[37]:


full_x2 = '{0} {1} {0} {1}'.format(first, last)
print(full_x2)


# You can also use string formatting to print out numbers in a way that is visually easier to digest and parse. Specify the number format you want after a `:` within the `{}`. The `8` before the `.` means that you want the entire number to be $8$ characters wide, padded with spaces at the start. The number after the `.` specifies how many decimal places you want. If there aren't that many decimal places, it will be zero padded at the end. The `'f'` represents that this should be displayed as a floating point number.

# In[38]:


pi_to_8decimals = 3.14159265
print('pi is approximately equal to {:.3f}'.format(pi_to_8decimals))
print('pi is approximately equal to {:8.3f}'.format(pi_to_8decimals))
print('pi is approximately equal to {:8.5f}'.format(pi_to_8decimals))
print('pi is (not) approximately equal to {:.13f}'.format(pi_to_8decimals))


# ## Importing Python Packages
# 
# Import the `math` library.  This is one of many libraries.

# In[39]:


import math


# The following will result in an error since the variable still ins't defined.
# 
# ```
# pi
# ```
# 
# But it is defined in the math library, so we can access it this way.

# In[40]:


math.pi


# In[41]:


print('pi is roughly: {:.30f}'.format(math.pi))
print(' e is roughly: {:.30f}'.format(math.e))
print()
print('pi is roughly: {:.10f}'.format(math.pi))
print(' e is roughly: {:.10f}'.format(math.e))


# ## Loops
# 
# For loops. Very important to do repeated, similar calculation.s Be careful with indentations.

# In[42]:


for counter in [1,2,3,4]:
    print("{}. Still in the first loop".format(counter))

print("Out of the first loop")


# You can also use `range()` to generate an [iterable](https://treyhunner.com/2018/02/python-range-is-not-an-iterator/) of items to iterate through without typing every one of them yourself.
# 
# This is helpful when you want to iterate through a bunch of things.

# In[43]:


for x in range(0, 9, 2):
    print("In second loop: {}".format(x))

print("Finished and out of second loop")


# Another kind of loop is a list comprehension. This is a powerful Python trick. A list comprehension allows us to quickly iterate through every item in a list to create a new list based on the original list.
# 
# **Tip:** We can use `list(range())` to create quickly create a list of integers.

# In[44]:


xs = list(range(5))
print('xs:', xs)


# The most basic use case of a list comprehension is similar to a call to `map()`, in that we are creating a new item for each item in the original list.
# 
# Here, we are mapping $x$ to $x^2$ for every item `x` in the list `xs`.

# In[45]:


ys = [x**2 for x in xs]
print('ys:', ys)


# The above is equivalent to the following for loop:

# In[46]:


y = []
for x in xs:
     y.append(x**2)
print('ys:', y)


# Notice how the list comprehension is much shorter, and more ["Pythonic"](https://docs.python-guide.org/writing/style/).

# We can also use list comprehension to act as a filter. In this use case, we are building a new list using items from the original list that meet a certain criteria.

# In[47]:


evens = [x for x in xs if x%2==0]
print('evens:', evens)


# Lastly, we can combine mapping and filtering. This will apply the filter first, and only add a mapped version of the original item if the original item passes the filter.
# 
# **Note:** Only the original item has to pass the filter, not the mapped version. In this example, we will add  $3$ to each even item. Note how none of the resulting items are even. They don't have to be. Only the original list items needed to be even.

# In[48]:


ys = [x+3 for x in xs if x%2==0]
print('ys:', ys)


# ## Non-Commutativity of Floats
# 
# Careful: different ways to sum numbers give different answers. This won't be too much of an issue for us, but always check what every step of your computation does.

# In[49]:


values = [ 0.1 ] * 10

sum1 = sum(values)

sum2 = 0.0
for i in values:
    sum2 += i

sum3 = math.fsum(values)

print('{:12}: {}'.format('Input values', values))
print('{:12}: {:.20f}'.format('sum()', sum1))
print('{:12}: {:.20f}'.format('for-loop', sum2))
print('{:12}: {:.20f}'.format('math.fsum()', sum3))


# ## Functions
# 
# Functions are a good way to reuse code.

# In[50]:


def get_circumference(r):
    c = 2 * math.pi * r
    return c

def get_area(r):
    """
    This is a function to compute the area of a circle of radius r
    """
    a = math.pi * r * r
    return a


# Let's try the functions we just wrote.

# In[51]:


r1 = 10
c1 = get_circumference(r1)
a1 = get_area(r1)

print('For a radius of {:.3f}, the circumference is {:.3f} and the area is {:.3f}'.format(r1, c1, a1))


# Because we already defined the function, we don't have to write the code again even ifwe wanted to calculate for a different circle!

# In[52]:


r2 = 20
c2 = get_circumference(r2)
a2 = get_area(r2)

print('For a radius of {:.3f}, the circumference is {:.3f} and the area is {:.3f}'.format(r2, c2, a2))


# In[53]:


def get_bounding_box_area(r):
    bba = 2**2 * r**2
    return bba

for r in range(0,21):
    print('{}:\t{:10.3f}\t{}:\t{:10.3f}\t{}:\t{:10.3f}\t{}:\t{:10.3f}'.format(
        'Radius', r,
        'Circumference', get_circumference(r),
        'Area', get_area(r),
        'BoundingBoxArea', get_bounding_box_area(r)
    ))


# ## Lambda functions
# 
# The syntax:
# 
# ```
# new_function = lambda x : do_stuff(x)
# ```
# 
# is equivalent to the syntax
# 
# ```
# def new_function(x):
#     return do_stuff(x)
# ```
# 
# However, one benefit of the former syntax is that it doesn't require us to name the new function, i.e. the following by itself is valid Python code:
# 
# ```
# lambda x : do_stuff(x)
# ```
# 
# Another benefit of this syntax is that it can be written on one line. One effect this has is to make it easier to use functions as arguments to other functions (so called "higher-order functions"). For example, one can write
# 
# ```
# higher_order_function(lambda x : do_stuff(x))
# ```
# 
# instead of
# 
# ```
# def new_function(x):
#     return do_stuff(x)
# 
# higher_order_function(new_function)
# ```
# 
# This can be helpful in cases where the definition of the new function is short, especially when we know that we won't use the new function more than once so that giving it a name isn't useful. (Of course if the new function's definition is short, but we know that we will use it more than once, we can still give it a name using the code `new_function = lambda x : do_stuff(x)`.)
# 
# A common example is when we have a function which takes more than one argument, e.g. `bivariate_function(x1, x2)`, and we want to define functions from it using "partial evaluation", e.g.
# 
# ```
# def new_function(x):
#     return bivariate_function(x1=x, x2=5)
# ```
# 
# Using lambdas, we can do this in one line:
# 
# ```
# lambda x : bivariate_function(x1=x, x2=5)
# ```
# 
# thus allowing us to use it in our higher-order function without needing to give it its own name:
# 
# ```
# higher_order_function(lambda x : bivariate_function(x1=x, x2=5))
# ```
# 
# In the case that the function definition needs to be longer than one line, lambda functions are less useful.
