# NumPy Tutorial: Arrays, Broadcasting, and Vectorization

Welcome to this tutorial on NumPy, focusing on arrays, broadcasting, dot multiplication, and vectorization techniques that eliminate for-loops to speed up machine learning algorithms like logistic regression. This tutorial is designed to help you understand these concepts through explanations and progressively challenging multiple-choice questions.

---

## Table of Contents

1. [NumPy Arrays](#section1)
2. [Array Reshaping](#section2)
3. [Broadcasting](#section3)
4. [Element-wise Operations](#section4)
5. [Dot Product and Matrix Multiplication](#section5)
6. [Vectorization Techniques](#section6)
7. [Putting It All Together](#section7)

---

<a name="section1"></a>
## 1. NumPy Arrays

### Understanding NumPy Arrays

NumPy arrays are the core of the NumPy library. They are similar to Python lists but provide additional functionality for numerical computations.

- **Creation**: You can create a NumPy array using `np.array()`.
- **Dimensions**: Arrays can be one-dimensional, two-dimensional, or multi-dimensional.
- **Attributes**: Important attributes include `.shape` (dimensions of the array) and `.dtype` (data type of the elements).

**Example:**

```python
import numpy as np

# Creating a one-dimensional array
a = np.array([1, 2, 3])
print(a.shape)  # Output: (3,)

# Creating a two-dimensional array
b = np.array([[1, 2], [3, 4]])
print(b.shape)  # Output: (2, 2)
```

### Multiple-Choice Questions

**Question 1**

Which of the following correctly creates a two-dimensional NumPy array with shape `(2, 3)`?

A. `np.array([[1, 2, 3], [4, 5, 6]])`  
B. `np.array([1, 2, 3, 4, 5, 6])`  
C. `np.array([[1, 2], [3, 4], [5, 6]])`  
D. `np.array([[[1, 2, 3]], [[4, 5, 6]]])`

**Question 2**

What is the shape of the array created by `np.zeros((4,))`?

A. `(4,)`  
B. `(1, 4)`  
C. `(4, 1)`  
D. `(4, 0)`

### Proceed or Review?

If you're confident with NumPy arrays, proceed to the next section. If not, consider reviewing this section.

---

<a name="section2"></a>
## 2. Array Reshaping

### Understanding Reshaping

Reshaping allows you to change the dimensions of an array without changing its data.

- **`reshape()` Method**: Used to give a new shape to an array.
- **Total Elements**: The total number of elements must remain the same.

**Example:**

```python
# Original array with 6 elements
a = np.array([1, 2, 3, 4, 5, 6])

# Reshaping to (2, 3)
b = a.reshape(2, 3)
print(b.shape)  # Output: (2, 3)
```

### Multiple-Choice Questions

**Question 3**

Suppose `x` is an array of shape `(8,)`. Which of the following is a valid reshape?

A. `x.reshape(2, 2, 2)`  
B. `x.reshape(4, 4)`  
C. `x.reshape(1, 8)`  
D. `x.reshape(2, 3)`

**Question 4**

Given `a = np.array([[1, 2], [3, 4]])`, what does `a.reshape(-1)` return?

A. A flattened array `[1, 2, 3, 4]`  
B. An array of shape `(2, 2)`  
C. An error due to ambiguous dimensions  
D. An array of shape `(1, 4)`

### Proceed or Review?

If you're comfortable with array reshaping, move on to the next section. Otherwise, revisit the examples.

---

<a name="section3"></a>
## 3. Broadcasting

### Understanding Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes.

- **Rules of Broadcasting**:
    1. If the arrays do not have the same rank, prepend the shape of the lower-rank array with ones until both shapes have the same length.
    2. Arrays are compatible in a dimension if they are equal or one of them is one.
    3. Arrays can be broadcast together if they are compatible in all dimensions.

**Example:**

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

b = np.array([1, 2, 3])    # Shape: (3,)

# Broadcasting b to shape (2, 3)
c = a + b
print(c)
```

### Multiple-Choice Questions

**Question 5**

Given `a.shape = (3, 4)` and `b.shape = (1, 4)`, what is the shape of `c = a + b`?

A. `(3, 4)`  
B. `(1, 4)`  
C. `(3, 1)`  
D. Broadcasting not possible due to incompatible shapes

**Question 6**

Which of the following is **not** a rule of broadcasting?

A. Arrays must have the same number of dimensions.  
B. Arrays are compatible in a dimension if one of them is one.  
C. The smaller array is "stretched" to match the larger array.  
D. Broadcasting is done element-wise.

### Proceed or Review?

Feel ready to tackle more on broadcasting? Proceed ahead. If not, take another look at the rules.

---

<a name="section4"></a>
## 4. Element-wise Operations

### Understanding Element-wise Operations

Element-wise operations are performed on arrays of the same shape or broadcasted to the same shape.

- **Addition (`+`)**: Adds corresponding elements.
- **Subtraction (`-`)**: Subtracts corresponding elements.
- **Multiplication (`*`)**: Multiplies corresponding elements.
- **Division (`/`)**: Divides corresponding elements.

**Example:**

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise multiplication
c = a * b  # Output: array([4, 10, 18])
```

### Multiple-Choice Questions

**Question 7**

Given `a.shape = (3, 4)` and `b.shape = (4, 1)`, which operation correctly computes `c[i][j] = a[i][j] * b[j]` without using loops?

A. `c = a * b.T`  
B. `c = a * b`  
C. `c = np.dot(a, b)`  
D. `c = a.T * b`

**Question 8**

Which operation is used for element-wise multiplication in NumPy?

A. `np.multiply(a, b)`  
B. `np.dot(a, b)`  
C. `a @ b`  
D. `np.cross(a, b)`

### Proceed or Review?

Confident with element-wise operations? Let's move forward. Otherwise, review the examples provided.

---

<a name="section5"></a>
## 5. Dot Product and Matrix Multiplication

### Understanding Dot Product

The dot product is a fundamental operation in linear algebra.

- **Vectors**: For 1D arrays, `np.dot(a, b)` computes the inner product.
- **Matrices**: For 2D arrays, `np.dot(a, b)` performs matrix multiplication.

**Example:**

```python
# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # Output: 32

# Matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
matrix_product = np.dot(A, B)
```

### Multiple-Choice Questions

**Question 9**

Given `a.shape = (12288, 150)` and `b.shape = (150, 45)`, what is the shape of `c = np.dot(a, b)`?

A. `(12288, 45)`  
B. `(150, 150)`  
C. `(12288, 150)`  
D. Operation not possible due to shape mismatch

**Question 10**

Which function performs matrix multiplication in NumPy?

A. `np.dot(a, b)`  
B. `a * b`  
C. `np.multiply(a, b)`  
D. `np.cross(a, b)`

### Proceed or Review?

Ready to delve into vectorization? If not, take time to revisit dot products.

---

<a name="section6"></a>
## 6. Vectorization Techniques

### Understanding Vectorization

Vectorization involves rewriting code to use array operations instead of explicit loops.

- **Advantages**:
    - **Speed**: Vectorized code runs faster because it's optimized in low-level languages.
    - **Readability**: Code is often shorter and easier to understand.
- **Techniques**:
    - Use NumPy functions that operate on entire arrays.
    - Replace Python loops with array operations.

**Example Without Vectorization:**

```python
# Without vectorization
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.zeros(3)

for i in range(3):
    c[i] = a[i] + b[i]
```

**Example With Vectorization:**

```python
# With vectorization
c = a + b
```

### Multiple-Choice Questions

**Question 11**

Which of the following is a benefit of vectorization in NumPy?

A. Slower execution time  
B. Increased memory usage without any performance gain  
C. Faster computations by leveraging optimized C code  
D. It makes the code more complex and harder to read

**Question 12**

How would you vectorize the following code snippet?

```python
for i in range(len(a)):
    c[i] = a[i] * b[i]
```

A. `c = np.dot(a, b)`  
B. `c = a * b`  
C. `c = np.multiply(a, b)`  
D. Both B and C

### Proceed or Review?

Feeling confident about vectorization? If so, proceed to the final section.

---

<a name="section7"></a>
## 7. Putting It All Together

### Application in Machine Learning

Vectorization is crucial in machine learning algorithms for efficiency.

- **Example**: Logistic regression cost function

Without vectorization:

```python
for i in range(m):
    z[i] = np.dot(w, x[i]) + b
    a[i] = 1 / (1 + np.exp(-z[i]))
```

With vectorization:

```python
z = np.dot(w, x.T) + b
a = 1 / (1 + np.exp(-z))
```

### Multiple-Choice Questions

**Question 13**

Why is vectorization important in implementing machine learning algorithms?

A. It allows using for-loops for better control  
B. It slows down computation but uses less memory  
C. It speeds up computation by utilizing optimized numerical libraries  
D. It is only useful for small datasets

**Question 14**

Consider `X` is a matrix of shape `(m, n_x)` where `m` is the number of examples. If we use row vectors for features, what is the correct shape of `X`?

A. `(n_x, m)`  
B. `(m, n_x)`  
C. `(m, 1)`  
D. `(1, n_x)`

### Proceed or Review?

If you've successfully answered these questions, you have a solid understanding of how these concepts fit into machine learning.

---

## Final Thoughts

Congratulations on completing this tutorial! By mastering arrays, broadcasting, and vectorization in NumPy, you're well-equipped to write efficient machine learning code.

---

**Note:** To check your understanding, revisit the multiple-choice questions and attempt to answer them without looking back at the explanations. If you find any areas challenging, consider reviewing that section before moving on.