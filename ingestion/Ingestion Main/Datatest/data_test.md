VIETNAM GENERAL CONFEDERATION OF LABOR

TON DUC THANG UNIVERSITY
FACULTY OF INFORMATION TECHNOLOGY

![Logo of Ton Duc Thang University](page_1_layout_ocr_rupp_254_212_129_76.png)

Tạ Chấn Nam - 52500174

MIDTERM ESSAY

LINEAR ALGEBRA FOR IT

HO CHI MINH CITY, 2025
VIETNAM GENERAL CONFEDERATION OF LABOR

TON DUC THANG UNIVERSITY
FACULTY OF INFORMATION TECHNOLOGY

![Logo of Ton Duc Thang University](page_2_layout_ocr_chhe_254_206_129_77.png)

Tạ Chấn Nam - 52500174

MIDTERM ESSAY

APPLIED CALCULUS FOR IT

Advised by
Mr. Tran Ha Son
HO CHI MINH CITY, 2025
i

# ACKNOWLEDGEMENT

I would like to express our sincere gratitude to Mr. Tran Ha Son, our instructor and mentor, for his valuable guidance and support throughout the mid-term report of our report about solving linear algebra logical question with relative knowledge. He is very helpful and patient in providing us with constructive feedback and suggestions to improve our work. Thanks to his friendly and harmonious personality, All of linear algebra lessons are very attractive and funny. I have learned a lot precious knowledge from his expertise, also experience in logical thinking, developed mindset and deeply understand about core value of study. I am very honored and privileged to have him as our teacher and supervisor.

*Ho Chi Minh city, 22nd December 2025.*

Author

*(Signature and full name)*

**Nam**

Tạ Chấn Nam
ii

## DECLARATION OF AUTHORSHIP

I hereby declare that this is our own report and is guided by Mr. Tran Ha Son; The content research and results contained herein are central and have not been published in any form before. The data in the tables for analysis, comments and evaluation are collected by the main author from different sources, which are clearly stated in the reference section.

In addition, the report also uses some comments, assessments as well as data of other authors, other organizations with citations and annotated sources.

**If something wrong happens, I’ll take full responsibility for the content of my report.** Ton Duc Thang University is not related to the infringing rights, the copyrights that I give during the implementation process (if any).

*Ho Chi Minh city, 22nd December 2025*

*Author*

*(Signature and full name)*

***Nam***
Tạ Chấn Nam
iii

# ABSTRACT

The goal of this report is represent and explain method to solve logical question related to linear algebra by used fundamental knowledge about them. The report include 2 Chapter:

**Chapter 1**: This chapter has 2 part. Part 1 primarily introduce to basic knowledge of linear algebra will be used to solve questions consist of Definition, Formula…. Next part will represent and explain detailed step to solve following question

- Question 1: Given the matrix A = $$ \begin{bmatrix} 1 & 2 & -1 \\ 2 & 2 & 1 \\ 1 & 2 & a \end{bmatrix} $$. Find all values of $a$ for which det( A)=0.

- Question 2: Solve the following system of linear equations by using Gaussian Elimination method.

a)
$$
\begin{cases}
x + 5y - 2z = 4 \\
3x - y + z = 3 \\
5x + y - 2z = 4
\end{cases}
$$
b)
$$
\begin{cases}
x + 3 - z = 3 \\
x - 2y + 2z = 4 \\
2x + y + z = 7
\end{cases}
$$

- Question 3: Let $v_1=(1;1;1)$, $v_2=(2;-5;1)$, $v_3=(3;0;5)$. Show that the set B=$\{v_1, v_2, v_3\}$ is a basis of $R^3$.

- Question 4: Find a matrix P that diagonalizes $A = \begin{bmatrix} 1 & 2 & 2 \\ 2 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}$

- Question 5: Let $S = \{v_1=(2;4;3), v_2 = (2;4;2), v_3= (-6;4;2)\}$. Find the coordinate vector of $v = (54, 12, 9)$ relative to S.
iv

- Question 6: Use the Gram-Schmidt orthonormalization process to transform $S = \{v_1 = (4;-8;8), v_2 = (8;8;4), v_3 = (8;-4;-8)\}$ for $R^3$ into an orthonormal basis.

- Question 7: Consider the vector space R3 with two bases:

$\varepsilon = \{\varepsilon_1, \varepsilon_2, \varepsilon_3\}$ in which $\varepsilon_1 = (1, 0, 0)$ , $\varepsilon_2 = (0, 1, 0)$ , $\varepsilon_3 = (0, 0, 1)$

$\theta = \{\theta_1, \theta_2, \theta_3\}$ in which $\theta_1 = (1, 1, 0)$ , $\theta_2 = (0, 1, 1)$ , $\theta_3 = (1, 0, 1)$

a) Find the transition matrix from the basis $\varepsilon$ to the basis $\theta$.

b) Find the transition matrix from the basis $\theta$ to the basis $\varepsilon$

**Chapter 2**: Show the correct answer of given question.
v

## TABLE OF CONTENT

**CHAPTER 1. SOLUTION** . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ....... 1
1.1 Introduction. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ....... 1
1.2 Detailed Steps. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .......
1

# CHAPTER 1. SOLUTION

## 1.1 Introduction.

- *Linear Algebra.*

Linear Algebra is the branch of mathematics that focuses on the study of vectors, vector spaces, matrices, and linear transformations. It deals with linear equations, linear functions, and their representations through matrices and determinants. It has a wide range of applications in Physics and Mathematics. It is the basic concept for machine learning and data science.

- *Matrix.*

In mathematics, a matrix (pl.: matrices) is a rectangular array of numbers or other mathematical objects with elements or entries arranged in rows and columns, usually satisfying certain properties of addition and multiplication.

- *Diagonal Matrix.*

In linear algebra, Diagonals Matrix is a square matrix that all elements except the main diagonal are zero.

- *Identity Matrix.*

An identity Matrix is a square matrix whose all diagonal elements are equal to 1 and the rest of the elements are zero.
2

- *Inverese Matrix.*

The inverse of a matrix is a square matrix that, when multiplied by itself, results in the identity matrix $I$.

A Matrix has its inverse if the determination of matrix non-zero

The inverse of a Matrix "A", denoted as $A^{-1}$.

$$ A \times A^{-1} = A^{-1} \times A = 1 $$

- *System of linear Equations.*

In mathematics, a system of linear equations (or linear system) is a collection of two or more linear equations involving the same variables.

For example, consider A general system of m linear equations with n unknowns and coefficients can be written as:

$$
\begin{cases}
a_{11}x_1 + a_{21}x_2 + \dots + a_{1n}x_n = b_1 \\
a_{12}x_1 + a_{11}x_2 + \dots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \dots + a_{mn}x_n = b_m
\end{cases}
$$

- *Augmented Matrices.*

Augmented Matrices are two matrices combined using their column values. Thus, if we have m columns in the first matrix and n columns in the second matrix, then in the Augmented Matrices we have (m + n) columns. Augmented Matrices is used to solve simple linear equations. An Augmented Matrices has the same number of rows as there are variables in the given linear equations.

An Augmented Matrices is a means to solve simple linear equations. The coefficients and constant values of the linear equations are represented as a matrix, referred to as an Augmented Matrices. In simple terms, the
3

Augmented Matrices is the combination of two simple matrices along the columns. If there are m columns in the first matrix and n columns in the second matrix, then there would be m + n columns in the Augmented Matrices.

Consider 3 Matrix:

Coefficient Matrix A = $$ \begin{bmatrix} a_1 & b_1 & ... & c_1 \\ a_2 & b_2 & ... & c_2 \\ a_3 & b_3 & ... & c_3 \end{bmatrix} $$

Constant Matrix B = $$ \begin{bmatrix} d_1 \\ d_2 \\ d_3 \end{bmatrix} $$

Variable Matrix X = $$ \begin{bmatrix} x \\ y \\ z \end{bmatrix} $$

The Augmented Matrices M is calculated as.

$$ M = (A \times X | B) $$

- **Row Echelon Form.**

of a matrix simplifies solving systems of linear equations, understanding linear transformations, and working with matrix equations.

A matrix is in Row Echelon form if it has the following properties:

1. Zero Rows at the Bottom: If there are any rows that are completely filled with zeros they should be at the bottom of the matrix.

2. Leading 1s: In each non-zero row, the first non-zero entry (called a leading entry) can be any non-zero number. It does not have to be 1.
4

3. Staggered Leading 1s: The leading entry in any row must be to the right of the leading entry in the row above it.

Example about a Row Echelon Form:

$$ A = \begin{bmatrix} 1 & 2 & -1 & 4 \\ 0 & 4 & 0 & 3 \\ 0 & 0 & 1 & 2 \end{bmatrix} $$

- *Linear Combination.*

Given a set of vectors $v_1, v_2, \dots, v_n$ in a vector space, a linear combination of these vectors is an expression of the form:

$$ w = c_1v_1 + c_2v_2 + \dots + c_nv_n $$

Where $c_1, c_2, \dots, c_n$ are scalars (real numbers, complex numbers, etc.).

Example of Linear Combination:

Consider 2 vector:

$$ v_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, v_2 = \begin{bmatrix} 3 \\ 4 \end{bmatrix} $$

A linear combination of $v_1$ and $v_2$ would be:

$$ w = c_1v_1 + c_2v_2 = c_1 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + c_2 \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{pmatrix} c_1 + 3c_2 \\ 2c_1 + 4c_2 \end{pmatrix} $$

- *Coordinate Vector.*

In a vector space, any vector can be written as a linear combination of a basis. The coefficients of the linear combination are called the coordinates of the vector with respect to the basis.

Let $S$ be a finite-dimensional linear space. Let $C = (c_1, c_2, ..., c_n)$ be a basis for $S$. For any $s \in S$, take the unique set of $k$ scalars $v_1, ..., v_n$ such that

$$ s = c_1v_1 + \dots + c_nv_n $$

Then, the $n \times 1$ vector
5

$$[s]_B = \begin{bmatrix} v_1 \\ \vdots \\ v_n \end{bmatrix}$$

is called the coordinate vector of $C$ with respect to the basis.

- **Gaussian Elimination.**

Gaussian elimination is a row reduction algorithm for solving linear systems. It involves a series of operations on the Augmented Matrices (which includes both coefficients and constants) to simplify it into a row echelon form or reduced row echelon form. This method can also help in determining the rank, determinant and inverse of matrices. Gaussian elimination is a method for solving systems of equations in matrix form.

Elementary Row Operations:

1. Interchanging Rows: Swap two rows.

2. Multiplying a Row by a Scalar: Multiply all elements of a row by a non-zero number.

3. Adding a Scalar Multiple of One Row to Another: Add or subtract a multiple of one row.

Categories of Linear Equation Systems:

1. Consistent Independent System: Has exactly one solution.

2. Consistent Dependent System: Has infinite solutions.

3. Inconsistent System: Has no solution.

- **Gauss-Jordan Elimination.**

Gauss-Jordan elimination method is a row reduction algorithm. This is an advanced version of Gaussian elimination but instead of eliminating just
6

element under main diagonal line, it’s also eliminate upper part to become a diagonal matrix or identity matrix.

- *Determinant.*

In mathematics, the determinant is a scalar-valued function of the entries of a square matrix. The determinant of a matrix A is commonly denoted det(A), det A.

- *Rule of Sarrus.*

The Rule of Sarrus is a mnemonic device for computing the determinant of a 3×3 matrix named after the French mathematician Pierre Frédéric Sarrus.

Consider a 3 × 3 Matrix.

$$ M = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} $$

Write out the first two columns of the matrix to the right of the third column, giving five columns in a row. Then add the products of the main diagonals and its parallel diagonals and going from top to bottom and subtract the products of the sub diagonals and its parallel diagonals going from bottom to top. This yields.

$$ Det(A) = \begin{vmatrix} a & b & c \\ d & e & f \\ g & h & i \end{vmatrix} = (aei + dhc + bfg) - (ceg + fha + bdi) $$

- *Laplace Expansion.*

In linear algebra, The Laplace Expansion, also called Cofactor expansion is an expression of the determinant of an $n \times n$ matrix $B$ as a weighted sum of minors, which are the determinants of some $(n-1) \times (n-1)$ submatrices of $B$. Specifically, for every $i$, the *Laplace expansion along the $i^{th}$ row* is the equality.
7

We have 2 way to perform Laplace Expansion and their General Formula.

1. Expansion along to column | 2. Expansion along to row
Det(B) = $$ \sum_{j=1}^{n} (-1)^{i+j} b_{i,j} m_{i,j} $$ | Det(B) = $$ \sum_{i=1}^{n} (-1)^{i+j} b_{i,j} m_{i,j} $$

where $b_{i,j}$ is the entry of the $i^{th}$ row and $j^{th}$ column of $B$, and $m_{i,j}$ is the determinant of the submatrix obtained by removing the $i^{th}$ row and the $j^{th}$ column of $B$. Similarly, the *Laplace expansion along the $j^{th}$ column* is the equality.

The Laplace Expansion and The Rule of Sarrus also can convert to each others.

$$
\begin{aligned}
Det(A) &= \begin{vmatrix} a & b & c \\ d & e & f \\ g & h & i \end{vmatrix} = (aei + dhc + bfg) - (ceg + hfa + bdi) \\
&\iff aei + dhc + bfg - ceg - hfa - bdi \\
&\iff a \times (ei - hf) + d \times (bi - hc) + g \times (bf - ce) \\
&\iff a \times \begin{vmatrix} e & f \\ h & i \end{vmatrix} + d \times \begin{vmatrix} b & c \\ h & i \end{vmatrix} + g \times \begin{vmatrix} b & c \\ h & i \end{vmatrix}
\end{aligned}
$$

- **The Crammer’s Rule.**

In linear algebra, Cramer's rule is an explicit formula for the solution of a system of linear equations with as many equations as unknowns, valid whenever the system has a unique solution. It expresses the solution in terms of the determinants of the (square) coefficient matrix and of matrices obtained from it by replacing one column by the column vector of right-sides of the equations

Consider a system of n linear equations for n unknowns, represented in matrix multiplication form as follows:
8

Ax=B

where the n × n matrix A has a nonzero determinant, and the vector x= (x₁, x₂, . . . ,xₙ)ᵀ is the column vector of the variables. Then the theorem states that in this case the system has a unique solution, whose individual values for the unknowns are given by:

$$x_i = \frac{\det(A_i)}{\det(A)} = i = 1,..., n$$

Where A is the matrix formed by replacing the i-th column of A by the column vector b

- *Linear Independence and Dependence.*

In a vector space, a set of vectors is said to be linearly independent if no vector in the set can be expressed as a linear combination of the other vectors in the set.

A set of vectors {v₁, v₂, . . . ,vₙ} is linearly independent if the equation: The expression c₁v₁ + c₂v₂ + . . . + cₙvₙ = 0 has only the trivial solution c₁ = c₂= . . . = cₙ = 0.

In contrast, if there exist non-zero scalars c₁, c₂, ..., cₙ such that the equation above holds, then the set of vectors is linearly dependent.

- *Vector Space.*

a vector space (also called a linear space) is a set whose elements, often called vectors, can be added together and multiplied ("scaled") by numbers called scalars. The concept of vector spaces is fundamental for linear algebra, together with the concept of matrices, which allows computing in vector spaces. This provides a concise and synthetic way for manipulating and studying systems of linear equations.

Vector spaces generalize Euclidean vectors, which allow modeling of physical quantities have not only a magnitude, but also a direction
9

Vector spaces are characterized by their dimension, which, roughly speaking, specifies the number of independent directions in the space. This means that for two vector spaces over a given field and with the same dimension, the properties that depend only on the vector-space structure are exactly the same.

- **Basis of Matrix.**

In mathematics, a set B of elements of a vector space V is called a basis if it was satisfied 2 privileged condition.

1. Every element of v can be written in a unique way as a finite linear combination of the elements of B. The coefficients of this linear combination are called the components or coordinates of the vector with respect to B. The elements of a basis are called the basis vectors.

2. The elements of B are linearly independent, and every element of v is a linear combination of the elements of B. In other words, a basis is a linearly independent spanning set.

- *Change of Basis.*

Change of basis transtition matrix is coordinate transtition matrix of a vector from this basis to other... Transition matrix $P_{S \to T}$ is coordinate matrix P along to S.

To find the change of basis transition Matrix, we need to find coordinate vector T with basis S, then write the matrix follow to a rule.

Consider 3 Matrix:

$$ u_1 = (a, d, g), u_2 = (b, e, h), u_3 = (c, f, i) $$
10

$$P = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}$$

$$v_1 = (A, D, G), v_2 = (B, E, H), v_3 = (C, F, I)$$

$$T = \begin{bmatrix} A & B & C \\ D & E & F \\ G & H & I \end{bmatrix}$$

Unknown Variable Matrix $X = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$

Each vector of T would be the constant matrix to Matrix P or $(P \times X | T_{v_1}), ..., (P \times X | T_{v_3})$ and thus the corefficient matrix is the set of variable vector computed in each case of Matrix P with $T_{v_1}, ..., T_{v_3}$.

- **Characteristic Polynomial.**

In linear algebra, the characteristic polynomial of a square matrix is a polynomial which is invariant under matrix similarity and has the eigenvalues as roots. It has the determinant and the trace of the matrix among its coefficients. The characteristic polynomial of an endomorphism of a finite-dimensional vector space is the characteristic polynomial of the matrix of that endomorphism over any basis (that is, the characteristic polynomial does not depend on the choice of a basis)

Consider an matrix $A_{n \times n}$, identity matrix $I_{n \times n}$ and the eigenvalue $\lambda$. The characteristic polynomial of A denoted by $p_A(\lambda)$ is the polynomial defined by

$$p_A(t) = \det(\lambda I - A)$$

where I denoted $n \times n$ identity matrix.

The characteristic equation, also known as the determinant equation, is the equation obtained by equating the characteristic polynomial to zero.
11

Defined by

$$ Det(A - \lambda I) = \vec{0} $$

- *Eigenvalues and Eigenvectors.*

Eigenvalues and eigenvectors are fundamental concepts in linear algebra, used in various applications such as matrix diagonalization, stability analysis, and data analysis. They are associated with a square matrix and provide insights into its properties.

Eigenvalues are unique scalar values linked to a matrix. They indicate how much an eigenvector gets stretched or compressed during the transformation. The eigenvector's direction remains unchanged unless the eigenvalue is negative, in which case the direction is simply reversed.

The equation for eigenvalue and eigenvalue are given by

$$ Av = \lambda v $$

where, A is the given square matrix, v is associated eigenvector and $\lambda$ is scalar eigenvalue.

Eigenvectors are non-zero vectors that, when multiplied by a matrix, only stretch or shrink without changing direction. The eigenvalue must be found first before the eigenvector. For any square matrix A of order n × n, the eigenvector is a column matrix of size n × 1. This is known as the right eigenvector, as matrix multiplication is not commutative.

The equation for eigenvalue and eigenvector are given by

$$ Av = \lambda v $$

where, A is the given square matrix, v is the eigenvector and $\lambda$ is any scalar multiple.
12

- *Algebraic and Geometric Multiplicity.*

The Algebraic Multiplicity is the number of times a specific eigenvalue appears in the polynomial equation that is produced by characteristic equation denoted by m<sub>a</sub>(λ).

The Geometric Multiplicity tells us the number of the linearly independent eigenvectors associated with the eigenvalue. To create a eigenspace which is a set of eigenvectors corresponding to the particular eigenvalue along with the zero vector denoted by m<sub>g</sub>(λ).

- *Matrix Diagonalization.*

Matrix diagonalization is the process of reducing a square matrix into its diagonal form using a similarity transformation. Not all matrices are diagonalizable. A matrix is diagonalizable if it has no defective eigenvalues, meaning each eigenvalue's geometric multiplicity is equal to its algebraic multiplicity.

A Diagonals matrix of an any matrix is the result of the combinimation theorical knowledges mentioned above include Characteristic Polynomial and Equation, Eigenvalues and Eigenvectors, Algebraic and Geometric Multiplicity, Inverese Matrix, Identity Matrix... defined by a general formula

$$ D = P^{-1}AP $$

Where, A is the given square matrix, P is a modal matrix which is consist of eigenvectors created by eigenvalues found form characteristic equation, P<sup>-1</sup> is the inverse matrix of P and D is the diagonals matrix of A.
13

- **Gram-Schmidt orthonormalization process.**

In mathematics, particularly linear algebra and numerical analysis, the Gram–Schmidt process or Gram-Schmidt algorithm is a way of finding a set of two or more vectors that are perpendicular to each other.

By technical definition, it is a method of constructing an orthonormal basis from a set of vectors in an inner product space, most commonly the Euclidean space R<sup>n</sup> equipped with the standard inner product. The Gram–Schmidt process takes a finite, linearly independent set of vectors S=(u<sub>1</sub>,...,u<sub>k</sub>) for k ≤ n and generates an orthogonal set S′=(f<sub>1</sub>,...,f<sub>k</sub>) that spans the same k-dimensional subspace of R<sup>n</sup> as S.

General Formula to find orthogonal basis:

f<sub>1</sub> = u<sub>1</sub>

f<sub>2</sub> = u<sub>2</sub> − $\frac{u_2 \times f_1}{\|f_1\|^2} \times f_1$

f<sub>3</sub> = u<sub>3</sub> − $\frac{u_3 \times f_1}{\|f_1\|^2} \times f_1$ − $\frac{u_3 \times f_2}{\|f_2\|^2} \times f_2$

$\Rightarrow f_k = u_k - \frac{u_k \times f_1}{\|f_1\|^2} \times f_1 - \frac{u_k \times f_2}{\|f_2\|^2} \times f_2 - ... - \frac{u_k \times f_{(k-1)}}{\|f_{(k-1)}\|^2} \times f_{(k-1)}$

Orthonormal basis is the Orthogonal basis after normalized by formula:

Normalized S′ = $\left(\frac{f_1}{\|f_1\|}, ..., \frac{f_k}{\|f_k\|}\right)$.

- **Rank of Matrix.**

The Rank of a Matrix is the maximum number of linearly independent rows or columns in a matrix denoted by ρ(A) or rank(A). It essentially determines the dimensionality of the vector space formed by the rows or columns of the matrix. If a matrix has all rows with zero elements, then the rank of a matrix is said to be zero.
14

## 

 1.2 Detailed Steps.

- 

 **Question 1.**

Method 1:

 By Using *the rule of Sarrus* in matrix A, we have a single-variable equation in term of “a” to find *determinant*. Given that the determinant of A equal to zero. We can generate the expression in term of “a” to find its real value

Step 1. Find the expression for the Determinant in term of “a”.

$$ det(A) = \begin{vmatrix} 1 & 2 & -1 \\ 2 & 2 & 1 \\ 1 & 2 & a \end{vmatrix} $$
$$ = (1 \times 2 \times a + 2 \times 2 \times (-1) + 2 \times 1 \times 1) - ((-1) \times 2 \times 1 + 1 \times 2 \times 1 + 2 \times 2 \times a) $$
$$ = (2a - 2) - 4a $$
$$ = -2a - 2 $$

Step 2. From the Determinant expression in term of “a” has been just found, we set it equal to zero then find the real value of “a”.

$$ det(A) = 0 \iff -2a - 2 \implies a = -1 $$

Method 2:

 By using the *Laplace expansion* method. Firstly, we eliminate value in column 1 with *elementary row operations* and try to convert it to zero then follow along to the *Laplace formula*.
15

Step 1. Find the expression in term of “a” through Laplace formula.

$$ det(A) = \begin{vmatrix} 1 & 2 & -1 \\ 2 & 2 & 1 \\ 1 & 2 & a \end{vmatrix} \xrightarrow[(row_3 - row_1)]{(row_2 - 2row_1)} \begin{vmatrix} 1 & 2 & -1 \\ 0 & -2 & 3 \\ 0 & 0 & a+1 \end{vmatrix} $$

$$ = 1 \times \begin{vmatrix} -2 & 3 \\ 0 & a+1 \end{vmatrix} + 0 \times \begin{vmatrix} 2 & -1 \\ 0 & a+1 \end{vmatrix} + 0 \times \begin{vmatrix} 2 & -1 \\ -2 & 3 \end{vmatrix} $$

$$ = -2a - 2 $$

Step 2. From the determinant expression in term of “a” has been just found, we set it equal to zero then find the real value of “a”.

$$ det(A) = 0 \Leftrightarrow -2a - 2 \Rightarrow a = -1 $$

- **Question 2.**

Method 1:

Firstly we turn the *system of linear equations* into its *augmented matrices* form. Then by using *Gaussian elimination* method, we convert it to *Row electron form* with *elementary row operations*. Finally turn it back *system of linear equations* form and find *categories of linear equation systems* thus conclude the solution of the System of Linear Equations.

a)

Step 1. Convert the system of linear equations to augmented matrices and find its echelon form of matrix A.

$$ [AX \mid B] = \left[ \begin{array}{ccc|c} 1 & 5 & -2 & 4 \\ 3 & -1 & 1 & 3 \\ 5 & 1 & -2 & 4 \end{array} \right] $$

$$ \left[ \begin{array}{ccc|c} 1 & 5 & -2 & 4 \\ 3 & -1 & 1 & 3 \\ 5 & 1 & -2 & 4 \end{array} \right] \xrightarrow[(row_3 - 5row_1)]{(row_2 - 3row_1)} \left[ \begin{array}{ccc|c} 1 & 5 & -2 & 4 \\ 0 & -16 & 7 & -9 \\ 0 & -24 & 8 & -16 \end{array} \right] $$

$$ \left[ \begin{array}{ccc|c} 1 & 5 & -2 & 4 \\ 0 & -16 & 7 & -9 \\ 0 & -24 & 8 & -16 \end{array} \right] \xrightarrow{\frac{row_3}{4}} \left[ \begin{array}{ccc|c} 1 & 5 & -2 & 4 \\ 0 & -16 & 7 & -9 \\ 0 & -6 & 2 & -4 \end{array} \right] $$
16

$$
\left[
\begin{array}{ccc|c}
1 & 5 & -2 & 4 \\
0 & -16 & 7 & -9 \\
0 & -6 & 2 & -4
\end{array}
\right]
\xrightarrow{\substack{(3row_2 - 8row_3) \\ (row_3 \leftrightarrow row_2)}}
\left[
\begin{array}{ccc|c}
1 & 5 & -2 & 4 \\
0 & -6 & 2 & -4 \\
0 & 0 & 5 & -5
\end{array}
\right]
$$

$\Rightarrow$ The system of linear equations is a consistent independent so it has exactly one solution.

Step 2. Turn eliminated matrix back to its system of linear equations and find the solution.

$$
\begin{cases}
x + 5y - 2z = 4 \\
-6y + 2z = -4 \\
5z = 5
\end{cases}
\Rightarrow
\begin{cases}
x = 1 \\
y = 1 \\
z = 1
\end{cases}
$$

b)

Step 1. Convert the system of linear equations to augmented matrices and find its echelon form

$$
[AX \mid B] = \left[
\begin{array}{ccc|c}
1 & 3 & -1 & 3 \\
1 & -2 & 2 & 4 \\
2 & 1 & 1 & 7
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|c}
1 & 3 & -1 & 3 \\
1 & -2 & 2 & 4 \\
2 & 1 & 1 & 7
\end{array}
\right]
\xrightarrow{\substack{(row_2 - row_1) \\ (row_3 - 2row_1)}}
\left[
\begin{array}{ccc|c}
1 & 3 & -1 & 3 \\
0 & -5 & 3 & 1 \\
0 & -5 & 3 & 1
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|c}
1 & 3 & -1 & 3 \\
0 & -5 & 3 & 1 \\
0 & -5 & 3 & 1
\end{array}
\right]
\xrightarrow{(row_3 - row_2)}
\left[
\begin{array}{ccc|c}
1 & 3 & -1 & 3 \\
0 & -5 & 3 & 1 \\
0 & 0 & 0 & 0
\end{array}
\right]
$$

$\Rightarrow$ The system of linear equations is consistent dependent so it has infinite solutions. We consider an arbitrary value and assign it to an unknown arbitrary variable in the system of linear equations called T.

Step 2. Find the solution.

$$
\begin{cases}
x + 3y - z = 3 \\
-5y + 3z = 1 \\
z = T
\end{cases}
\Rightarrow
\begin{cases}
x = \frac{-18 - 4T}{5} \\
y = \frac{-1 + 3T}{5} \\
z = T
\end{cases}
$$
17

Method 2:

First of all, we need to turn *the linear system* into its *augmented matrices*, then we need to check whether *the linear system* is *independent* or not by *the determinant*. If it is linear independent system , we can using *the Crammer’s rule* to infer the solution of the linear system.

a)

Step 1. Find the Determinant of A:
consider [AX | B] = $$ \begin{bmatrix} 1 & 5 & -2 & | & 4 \\ 3 & -1 & 1 & | & 3 \\ 5 & 1 & -2 & | & 4 \end{bmatrix} $$
$$ \det(A) = \begin{vmatrix} 1 & 5 & -2 \\ 3 & -1 & 1 \\ 5 & 1 & -2 \end{vmatrix} $$
$$ = (2 + 25 - 6) - (10 + 1 - 30) = 40 $$

Step 2. Because the Determinant is non-zero so it has unique solution so we can applying Cramer’s Rule successively substituting B into the columns of matrix A, we can find the Determinant of $A_x$, $A_y$, $A_z$.
$$ \det(A_x) = \begin{vmatrix} 4 & 5 & -2 \\ 3 & -1 & 1 \\ 4 & 1 & -2 \end{vmatrix} $$
$$ = (8 - 6 + 20) - (8 + 4 - 30) = 40 $$

$$ \det(A_y) = \begin{vmatrix} 1 & 4 & -2 \\ 3 & 3 & 1 \\ 5 & 4 & -2 \end{vmatrix} $$
$$ = (-6 - 24 + 20) - (-30 + 4 - 24) = 40 $$

$$ \det(A_z) = \begin{vmatrix} 1 & 5 & 4 \\ 3 & -1 & 3 \\ 5 & 1 & 4 \end{vmatrix} $$
$$ = (-4 + 12 + 75) - (-20 + 60 + 3) = 40 $$
18

Step 3. Find the solution of the linear system of equations along to Cramer’s formula.

$$
\begin{cases}
x = \frac{\det(A_x)}{\det(A)} = \frac{40}{40} = 1 \\
y = \frac{\det(A_y)}{\det(A)} = \frac{40}{40} = 1 \\
z = \frac{\det(A_z)}{\det(A)} = \frac{40}{40} = 1
\end{cases}
$$

b)

Step 1. Find the determinant of A:

consider $[A \mid B] = \left[\begin{array}{ccc|c} 1 & 3 & -1 & 3 \\ 1 & -2 & 2 & 4 \\ 2 & 1 & 1 & 7 \end{array}\right]$

$\det(A) = \left|\begin{array}{ccc} 1 & 3 & -1 \\ 1 & -2 & 2 \\ 2 & 1 & 1 \end{array}\right|$

$= (-2 - 1 + 12) - (4 + 2 + 3) = 0$

Step 2. Because A already isn’t linear independent so we can not use The Cramer’s Rule to find the solution for the system of linear equations. Therefore, we should be use the Gaussian elimination method alternatively mentioned at method 1.

Step 3. According to method 1 the solution of this system of linear equations is

$$
\begin{cases}
x + 3y - z = 3 \\
-5y + 3z = 1 \\
z = T
\end{cases}
\implies
\begin{cases}
x = \frac{-18 - 4T}{5} \\
y = \frac{-1 + 3T}{5} \\
z = T
\end{cases}
$$
19

- **Question 3.**

Method:

We need two condition to prove set B is a *Basis* of R$^3$. Firstly, B must be a *linear independent matrix*, we can prove it through using Determinant. Secondly, by using *Gaussian elimination* to convert B to its *row echelon form* then compare the number of non-zero row in matrix B (rank(B)) with the number of dimension in R space(dim( R$^3$ )). if both of them is equal so we can conclude B is a basis of R$^3$.

Step 1. Find the determinant of B by using the rule of Sarrus.

$$ det(B) = \begin{vmatrix} 1 & 2 & 3 \\ 1 & -5 & 0 \\ 1 & 1 & 5 \end{vmatrix} $$
$$ = (1 \times (-5) \times 5 + 1 \times 1 \times 3 + 2 \times 0 \times 1) - (3 \times (-5) \times 1 + 1 \times 0 \times 1 + 2 \times 1 \times 5) $$
$$ = -17 $$

Because the determinant of matrix is non-zero so it's linear independent. Thus, we satisfied first condition

Step 2. Find rank of matrix B

$$ B = \begin{bmatrix} 1 & 2 & 3 \\ 1 & -5 & 0 \\ 1 & 1 & 5 \end{bmatrix} \xrightarrow{\substack{(row_2 - row_1) \\ (row_3 - row_1)}} \begin{bmatrix} 1 & 2 & 3 \\ 0 & -7 & -3 \\ 0 & -1 & 2 \end{bmatrix} $$
$$ \begin{bmatrix} 1 & 2 & 3 \\ 0 & -7 & -3 \\ 0 & -1 & 2 \end{bmatrix} \xrightarrow{(row_2 - 7row_3)} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & -17 \\ 0 & -1 & 2 \end{bmatrix} $$
$$ \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & -17 \\ 0 & -1 & 2 \end{bmatrix} \xrightarrow{(row_2 \leftrightarrow row_3)} \begin{bmatrix} 1 & 2 & 3 \\ 0 & -1 & 2 \\ 0 & 0 & -17 \end{bmatrix} $$

rank(B) = dim(R$^3$) = 3, because matrix B has satisfied both need condition so we can conclude that B is the basis of R$^3$
20

- **Question 4.**

Method:

Firstly, we consider identity matrix I, *eigenvalue* $\lambda$ then construct a *characteristic equation* between matrix A and $\lambda I$. After that we find value of $\lambda$ and determine *algebraic multiplicity* and with each value of $\lambda$ we alternate it into characteristic equation and using the **Gaussian elimination** to find unknown variable and determine *geometric multiplicity*. If *algebraic multiplicity* equal to *geometric multiplicity*, we conclude that matrix A is diagonalizable. Therefore, we can infer *eigenvectors*. The set of result eigenvectors P = ($v_1$, $v_2$, $v_3$) will be the matrix which is able to diagonalize A.

Step 1. Find eigenvalue and algebraic of multiplicity.

$$P_A(\lambda) = Det(A - \lambda I) = 0$$

$$\Leftrightarrow \begin{vmatrix} 1-\lambda & 2 & 2 \\ 2 & 1-\lambda & 1 \\ 0 & 0 & 1-\lambda \end{vmatrix} = 0$$

$$\Leftrightarrow ((1-\lambda)^3 + 2 \times 0 \times 2 + 2 \times 1 \times 0) - (2 \times (1-\lambda) \times 0 + 1 \times 0 \times (1-\lambda) + 2 \times 2 \times (1-\lambda))$$

$$= 0$$

$$\Leftrightarrow (1-\lambda)^3 - 4 + 4\lambda = 0$$

$$\Leftrightarrow 1 - 3\lambda + 3\lambda^2 - \lambda^3 - 4 + 4\lambda = 0$$

$$\Leftrightarrow -\lambda^3 + 3\lambda^2 + \lambda - 3 = 0$$

$$\Rightarrow \begin{cases} \lambda_1 = -1 \\ \lambda_2 = 3 \\ \lambda_3 = 1 \end{cases} \Rightarrow m_a(\lambda) = 1$$
21

Step 2. Find eigenvectors and geometric multiplicity.

$$ \lambda_1 = -1 $$

$$ E(1) = [(A - \lambda_1 I)X \mid 0] = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 1 \\ 0 & 0 & 2 \end{bmatrix} $$

$$ \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 1 \\ 0 & 0 & 2 \end{bmatrix} \xrightarrow{(row_2 - row_1)} \begin{bmatrix} 2 & 2 & 2 \\ 0 & 0 & -1 \\ 0 & 0 & 2 \end{bmatrix} $$

$$ \begin{bmatrix} 2 & 2 & 2 \\ 0 & 0 & -1 \\ 0 & 0 & 2 \end{bmatrix} \xrightarrow{(row_3 + 2row_2)} \begin{bmatrix} 2 & 2 & 2 \\ 0 & 0 & -1 \\ 0 & 0 & 0 \end{bmatrix} = \begin{cases} 2x + 2y + 2z = 0 \\ -z = 0 \end{cases} $$

We consider $y = T \Rightarrow \begin{cases} x = -T \\ y = T \\ z = 0 \end{cases}$

$$ E(1) = (-T, T, 0 \mid T \in R) = T \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix} \Rightarrow E(1) = span(v_1), m_g(\lambda_1) = 1 $$

$$ \lambda_2 = 3 $$

$$ E(2) = [(A - \lambda_2 I)X \mid 0] = \begin{bmatrix} -2 & 2 & 2 \\ 2 & -2 & 1 \\ 0 & 0 & -2 \end{bmatrix} $$

$$ \begin{bmatrix} -2 & 2 & 2 \\ 2 & -2 & 1 \\ 0 & 0 & -2 \end{bmatrix} \xrightarrow{(row_2 + row_1)} \begin{bmatrix} -2 & 2 & 2 \\ 0 & 0 & 3 \\ 0 & 0 & -2 \end{bmatrix} $$

$$ \begin{bmatrix} -2 & 2 & 2 \\ 0 & 0 & 3 \\ 0 & 0 & -2 \end{bmatrix} \xrightarrow{(3row_3 + 2row_2)} \begin{bmatrix} -2 & 2 & 2 \\ 0 & 0 & 3 \\ 0 & 0 & 0 \end{bmatrix} = \begin{cases} 2x + 2y + 2z = 0 \\ 3z = 0 \end{cases} $$

We consider $y = T \Rightarrow \begin{cases} x = T \\ y = T \\ z = 0 \end{cases}$

$$ E(2) = (T, T, 0 \mid T \in R) = T \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} \Rightarrow E(2) = span(v_2), m_g(\lambda_2) = 1 $$
22

$$ \lambda_3 = 1 $$

E(3) = [(A - $\lambda_3$I)X | 0] = $\begin{bmatrix} 0 & 2 & 2 \\ 2 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$

$\begin{bmatrix} 0 & 2 & 2 \\ 2 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix} \xrightarrow{(row_2 \leftrightarrow row_1)} \begin{bmatrix} 2 & 0 & 1 \\ 0 & 2 & 2 \\ 0 & 0 & 0 \end{bmatrix} = \begin{cases} 2x + z = 0 \\ 2y + 2z = 0 \end{cases}$

We consider x = T $\Rightarrow$ $\begin{cases} x = T \\ y = 2T \\ z = -2T \end{cases}$

E(3) = (T, 2T, -2T | T $\in$ R) = T $\begin{bmatrix} 1 \\ 2 \\ -2 \end{bmatrix}$ $\Rightarrow$ E(3) = span(v$_3$), m$_g(\lambda_3)$ = 1

Step 3. Find matrix P which diagonalize A.

P=(v$_1$, v$_2$, v$_3$) = $\begin{bmatrix} -1 & 1 & 1 \\ 1 & 1 & 2 \\ 0 & 0 & -2 \end{bmatrix}$

Because both value of algebraic multiplicity and geometric multiplicity are equal so matrix A is diagonalizable. Then to ensure that P is really the matrix which is diagonalize A, we consider the result of the diagonalization matrix general formula, if the multiply between P, A and the inverse matrix of P was a diagonal matrix called D so the matrix P which have just found is correct answer

D=P$^{-1}$AP

= $\begin{bmatrix} -\frac{1}{2} & \frac{1}{2} & \frac{1}{4} \\ \frac{1}{2} & \frac{1}{2} & \frac{3}{4} \\ 0 & 0 & -\frac{1}{2} \end{bmatrix} \times \begin{bmatrix} 1 & 2 & 2 \\ 2 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix} \times \begin{bmatrix} -1 & 1 & 1 \\ 1 & 1 & 2 \\ 0 & 0 & -2 \end{bmatrix}$

= $\begin{bmatrix} -1 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 1 \end{bmatrix}$
23

- **Question 5.**

Method:

The main method to solve this question is find *coordinate vector* $v$ relative to S is build a *system of linear equations* around them and turn it into *augmented matrices* form. Then by using *Gaussian elimination method* to convert matrix to its row electron form and determine the *categories of linear equation system*. If it was a independent consistent system, the solution would be the *coordinate vector* of v relative to S, unless there is no *coordinate vector* of v relative to S.

Step 1. Find echelon form of augmented matrices.

$$ [SX \mid v] = \begin{bmatrix} 2 & 2 & 6 & \mid & 54 \\ 4 & 4 & 4 & \mid & 12 \\ 3 & 2 & 2 & \mid & 9 \end{bmatrix} $$

$$ \begin{bmatrix} 2 & 2 & 6 & \mid & 54 \\ 4 & 4 & 4 & \mid & 12 \\ 3 & 2 & 2 & \mid & 9 \end{bmatrix} \xrightarrow[\frac{row_1}{2}]{\frac{row_2}{4}} \begin{bmatrix} 1 & 1 & -3 & \mid & 27 \\ 1 & 1 & 1 & \mid & 3 \\ 3 & 2 & 2 & \mid & 9 \end{bmatrix} $$

$$ \begin{bmatrix} 1 & 1 & -3 & \mid & 27 \\ 1 & 1 & 1 & \mid & 3 \\ 3 & 2 & 2 & \mid & 9 \end{bmatrix} \xrightarrow[(row_3 - 3row_1)]{(row_2 - row_1)} \begin{bmatrix} 1 & 1 & -3 & \mid & 27 \\ 0 & 0 & 4 & \mid & -24 \\ 0 & -1 & 11 & \mid & -72 \end{bmatrix} $$

$$ \begin{bmatrix} 1 & 1 & -3 & \mid & 27 \\ 0 & 0 & 4 & \mid & -24 \\ 0 & -1 & 11 & \mid & -72 \end{bmatrix} \xrightarrow{(row_2 \leftrightarrow row_3)} \begin{bmatrix} 1 & 1 & -3 & \mid & 27 \\ 0 & -1 & 11 & \mid & -72 \\ 0 & 0 & 4 & \mid & -24 \end{bmatrix} $$

$\Rightarrow$ The system of linear equations is a consistent independent system so it has exactly one solution.

Step 2. Find solution of augmented matrix.

$$ \begin{cases} x + y - 3z = 27 \\ -y + 11z = -72 \\ 4z = -24 \end{cases} \Rightarrow \begin{cases} x = 3 \\ y = 6 \\ z = -6 \end{cases} \Rightarrow coordinate vector  [v]_s = \begin{bmatrix} 3 \\ 6 \\ -6 \end{bmatrix} $$
24

- **Question 6.**

Method:

Firstly, we need to check if the *orthonormal basis* of S is available, a basis could orthonormalized if it was *linear independent*. By using *the rule of Sarrus* or *the Laplace expansion*, we will determine whether it is linear independent through the determinant. Finally, with *Gram-Schmidt orthonormalization process* We can find a set of vectors that are perpendicular to each other in matrix S with it General Formula and construct its *orthogonal matrix* and normalize it.

Step 1. Check the determinant of matrix

$$ det(B) = \begin{vmatrix} 4 & 8 & 8 \\ -8 & 8 & -4 \\ 8 & 4 & -8 \end{vmatrix} $$

$$ = (4 \times 8 \times (-8) + (-8) \times 4 \times 8 + 8 \times (-4) \times 8) $$
$$ - (8 \times 8 \times 8 + 4 \times (-4) \times 4 + (-8) \times 8 \times (-8)) $$

$$ = -1728 $$

$\Rightarrow$ Basis S is linear independent so we can conclude it has a orthogonal basis called $S' = (f_1, f_2, f_3)$.

Step 2. Find $f_1, f_2, f_3$ in $S'$

$f_1 = row_1(S) = (4, -8, 8)$

$f_2 = row_2(S) - \frac{row_2(S) \times f_1}{\|f_1\|^2} \times f_1$

$$ \frac{row_2(S) \times f_1}{\|f_1\|^2} = \frac{(8, 8, 4) \times (4, -8, 8)}{\left(\sqrt{4^2 + (-8)^2 + 8^2}\right)^2} = 0 $$

$\Rightarrow f_2 = row_2(S) = (8, 8, 4)$
25

$$ f_3 = row_2(S) - \frac{row_3(S) \times f_1}{\|f_1\|^2} \times f_1 - \frac{row_3(S) \times f_2}{\|f_2\|^2} \times f_2 $$

$$ \frac{row_3(S) \times f_1}{\|f_1\|^2} = \frac{(8, -4, -8) \times (4, -8, 8)}{\left(\sqrt{4^2 + (-8)^2 + 8^2}\right)^2} = 0 $$

$$ \frac{row_3(S) \times f_2}{\|f_2\|^2} = \frac{(8, -4, -8) \times (8, 8, 4)}{\left(\sqrt{8^2 + 8^2 + 4^2}\right)^2} = 0 $$

$$ \Rightarrow f_3 = row_3(S) = (8, -4, 8) $$

$$ \Rightarrow Orthogonal basis of  S = S' = \begin{bmatrix} 4 & 8 & 8 \\ -8 & 8 & -4 \\ 8 & 4 & -8 \end{bmatrix} $$

Step 3. Normalized orthogonal basis, we divide each vector in orthogonal basis with their norm.

Orthonormal basis of $S' = \left( \frac{f_1}{\|f_1\|}, \frac{f_2}{\|f_2\|}, \frac{f_3}{\|f_3\|} \right) = \begin{bmatrix} \frac{1}{3} & \frac{2}{3} & \frac{2}{3} \\ -\frac{2}{3} & \frac{2}{3} & -\frac{1}{3} \\ \frac{2}{3} & \frac{1}{3} & -\frac{2}{3} \end{bmatrix}$

- **Question 7.**

Method 1:

By using *Gauss-Jordan elimination* method we construct an *augmented matrices* by placing the vectors of the target basis $\varepsilon$ on the left and the vectors of the source basis $\theta$ on the right, forming $[\varepsilon \mid \theta]$. Then, by applying *elementary row operations*, we transform the left side into the *identity matrix* I. When the left side becomes I, the resulting matrix on the right side will be the transition matrix from $\varepsilon$ to $\theta$ and conversely or $[\varepsilon \mid \theta]$ will converted to $[I \mid P_{\varepsilon \to \theta}]$ and reversely.

a)

Step 1. Transit basis from $\varepsilon$ to $\theta$

$$ P_{\varepsilon \to \theta} = [\varepsilon \mid \theta] = \begin{bmatrix} 1 & 0 & 0 & 1 & 0 & 1 \\ 0 & 1 & 0 & 1 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 & 1 \end{bmatrix} $$
26

$$
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
0 & 1 & 0 & 1 & 1 & 0 \\
0 & 0 & 1 & 0 & 1 & 1
\end{array}
\right]
\xrightarrow{(row_2 - row_1)}
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
0 & 0 & 1 & 0 & 1 & 1
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
0 & 0 & 1 & 0 & 1 & 1
\end{array}
\right]
\xrightarrow{(row_3 - row_2)}
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
1 & -1 & 1 & 0 & 0 & 2
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
1 & -1 & 1 & 0 & 0 & 2
\end{array}
\right]
\xrightarrow{\frac{row_3}{2}}
\left[
\begin{array}{ccc|ccc}
1 & 0 & 1 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} & 0 & 0 & 1
\end{array}
\right]
$$

<!-- layout: kevh, ccig -->
$$
\left[
\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & 0 & 1 \\
-1 & 1 & 0 & 0 & 1 & -1 \\
\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} & 0 & 0 & 1
\end{array}
\right]
\xrightarrow[\substack{(row_1 - row_3)}]{(row_2 + row_3)}
\left[
\begin{array}{ccc|ccc}
\frac{1}{2} & \frac{1}{2} & -\frac{1}{2} & 1 & 0 & 0 \\
-\frac{1}{2} & \frac{1}{2} & \frac{1}{2} & 0 & 1 & 0 \\
\frac{1}{2} & -\frac{1}{2} & \frac{1}{2} & 0 & 0 & 1
\end{array}
\right]
$$

$$
\Rightarrow P_{\varepsilon \to \theta} =
\left[
\begin{array}{ccc}
\frac{1}{2} & \frac{1}{2} & -\frac{1}{2} \\
-\frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\
\frac{1}{2} & -\frac{1}{2} & \frac{1}{2}
\end{array}
\right]
$$

b)

Step 1. Convert matrix $\varepsilon$ in $[\theta \mid \varepsilon]$ to identity matrix.

$$
P_{\theta \to \varepsilon} = [\theta \mid \varepsilon] =
\left[
\begin{array}{ccc|ccc}
1 & 0 & 1 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 & 1 & 0 \\
0 & 1 & 1 & 0 & 0 & 1
\end{array}
\right]
$$

Step 2. Because the matrix $\varepsilon$ already identity matrix so we can say that.

$$
P_{\theta \to \varepsilon} = \theta =
\left[
\begin{array}{ccc}
1 & 0 & 1 \\
1 & 1 & 0 \\
0 & 1 & 1
\end{array}
\right]
$$

Another proof to conclude $P_{\varepsilon \to \theta}$ and $P_{\theta \to \varepsilon}$ is the change of basis matrix from $\varepsilon$ to $\theta$ and reversely is the inverse matrix of $P_{\varepsilon \to \theta}$ is $P_{\theta \to \varepsilon}$
27

Method 2:

By transiting each vectors in matrix basis $\theta$ to basis $\varepsilon$. The resulting set of vectors in basis $\theta$ **a** after transited will be the the change of basis matrix from $\theta$ to $\varepsilon$ and reversely.

a)

Step 1. Find vectors in $\theta$ to $\varepsilon$ through solution of the augmented matrix constructed by $\theta$ and each vectors in $\varepsilon$

$$[\thetaX \mid column_1(\varepsilon)] = \begin{bmatrix} 1 & 0 & 1 & | & 1 \\ 1 & 1 & 0 & | & 0 \\ 0 & 1 & 1 & | & 0 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 1 \\ 1 & 1 & 0 & | & 0 \\ 0 & 1 & 1 & | & 0 \end{bmatrix} \xrightarrow{(row_2 - row_1)} \begin{bmatrix} 1 & 0 & 1 & | & 1 \\ 0 & 1 & -1 & | & -1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 1 \\ 0 & 1 & -1 & | & -1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix} \xrightarrow{(row_3 - row_2)} \begin{bmatrix} 1 & 0 & 1 & | & 1 \\ 0 & 1 & -1 & | & -1 \\ 0 & 0 & 2 & | & 1 \end{bmatrix}$$

$$= \begin{cases} x + z = 1 \\ y - z = -1 \\ 2z = 1 \end{cases} \implies \begin{cases} x = \frac{1}{2} \\ y = -\frac{1}{2} \\ z = \frac{1}{2} \end{cases} = v_1$$

$$[\thetaX \mid column_2(\varepsilon)] = \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 1 & 1 & 0 & | & 1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 1 & 1 & 0 & | & 1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix} \xrightarrow{(row_2 - row_1)} \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 1 \\ 0 & 1 & 1 & | & 0 \end{bmatrix} \xrightarrow{(row_3 - row_2)} \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 1 \\ 0 & 0 & 2 & | & -1 \end{bmatrix}$$

$$= \begin{cases} x = 0 \\ y - z = 1 \\ 2z = -1 \end{cases} \implies \begin{cases} x = \frac{1}{2} \\ y = \frac{1}{2} \\ z = -\frac{1}{2} \end{cases} = v_2$$
28

$$[\theta X \mid column_3(\varepsilon)] = \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 1 & 1 & 0 & | & 0 \\ 0 & 1 & 1 & | & 1 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 1 & 1 & 0 & | & 0 \\ 0 & 1 & 1 & | & 1 \end{bmatrix} \xrightarrow{(row_2 - row_1)} \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 0 \\ 0 & 1 & 1 & | & 1 \end{bmatrix}$$

$$\begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 0 \\ 0 & 1 & 1 & | & 1 \end{bmatrix} \xrightarrow{(row_3 - row_2)} \begin{bmatrix} 1 & 0 & 1 & | & 0 \\ 0 & 1 & -1 & | & 0 \\ 0 & 0 & 2 & | & 1 \end{bmatrix}$$

$$= \begin{cases} x + z = 0 \\ y - z = 0 \\ 2z = 1 \end{cases} \Rightarrow \begin{cases} x = -\frac{1}{2} \\ y = \frac{1}{2} \\ z = \frac{1}{2} \end{cases} = v_3$$

Step 2. $$P_{\varepsilon \to \theta} = (v_1, v_2, v_3) = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} & -\frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \end{bmatrix}$$

b)

Step 1. Find vectors in $\varepsilon$ to $\theta$ through solution of the augmented matrix constructed by $\varepsilon$ and each vectors in $\theta$.

$$[\varepsilon X \mid column_1(\theta)] = \begin{bmatrix} 1 & 0 & 0 & | & 1 \\ 0 & 1 & 0 & | & 1 \\ 0 & 0 & 1 & | & 0 \end{bmatrix} \Rightarrow \begin{cases} x = 1 \\ y = 1 = u_1 \\ z = 0 \end{cases}$$

$$[\varepsilon X \mid column_2(\theta)] = \begin{bmatrix} 1 & 0 & 0 & | & 0 \\ 0 & 1 & 0 & | & 1 \\ 0 & 0 & 1 & | & 1 \end{bmatrix} \Rightarrow \begin{cases} x = 0 \\ y = 1 = u_2 \\ z = 1 \end{cases}$$

$$[\varepsilon X \mid column_3(\theta)] = \begin{bmatrix} 1 & 0 & 0 & | & 1 \\ 0 & 1 & 0 & | & 0 \\ 0 & 0 & 1 & | & 1 \end{bmatrix} \Rightarrow \begin{cases} x = 1 \\ y = 0 = u_3 \\ z = 1 \end{cases}$$

Step 2. $$P_{\theta \to \varepsilon} = (u_1, u_2, u_3) = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$$
29

## CHAPTER 2. RESULT

### 2.1 Question 1.

Given the matrix $A = \begin{bmatrix} 1 & 2 & -1 \\ 2 & 2 & 1 \\ 1 & 2 & a \end{bmatrix}$. Find all values of $a$ for which $\det(A)=0$.

$a = -1$

### 2.2 Question 2.

Solve the following system of linear equations by using Gaussian Elimination method.

a)
$$
\begin{cases}
x + 5y - 2z = 4 \\
3x - y + z = 3 \\
5x + y - 2z = 4
\end{cases}
$$

b)
$$
\begin{cases}
x + 3 - z = 3 \\
x - 2y + 2z = 4 \\
2x + y + z = 7
\end{cases}
$$

a) The system of linear equations has only one solution
$x = 1, y = 1, z = 1$

b) The system of linear equations has infinite solution
$x = \frac{-18-4T}{5}, y = \frac{-1+3T}{5}, z = T$

### 2.3 Question 3.

Let $v_1=(1;1;1), v_2=(2;-5;1), v_3=(3;0;5)$. Show that the set $B = (v_1, v_2, v_3)$ is a basis of $R^3$.
30

$$
\begin{cases} *Matrix B is linear independent* \\ rank(B) = dim(R^3)=3 \end{cases} \implies B * is a basis of * R^3
$$

## 2.4 Question 4.

Find a matrix P that diagonalize $A = \begin{bmatrix} 1 & 2 & 2 \\ 2 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}$

$P = \begin{bmatrix} -1 & 1 & 1 \\ 1 & 1 & 2 \\ 0 & 0 & -2 \end{bmatrix}$

## 2.5 Question 5.

Let $S = \{v_1=(2;4;3), v_2 = (2;4;2), v_3= (-6;4;2)\}$. Find the coordinate vector of $v= (54, 12, 9)$ relative to S.

$$[v]_s = \begin{bmatrix} 3 \\ 6 \\ -6 \end{bmatrix}$$

## 2.6 Question 6.

Use the Gram-Schmidt orthonormalization process to transform the basis $S = \{v_1 = (4;-8;8), v_2 = (8;8;4), v_3 = (8;-4;-8)\}$ for $R^3$ into an orthonormal basis.

orthonormal basis of $S = \begin{bmatrix} \frac{1}{3} & \frac{2}{3} & \frac{2}{3} \\ -\frac{2}{3} & \frac{2}{3} & -\frac{1}{3} \\ \frac{2}{3} & \frac{1}{3} & -\frac{2}{3} \end{bmatrix}$
31

## 

 2.7 Question 7.

Sales in Each Quarter

<table>
    <tr>
        <th>Quarter</th>
        <th>Total Sales</th>
    </tr>
<tr>
        <td>Q1</td>
<td>$20,000</td>
    </tr>
<tr>
        <td>Q2</td>
<td>$24,000</td>
    </tr>
<tr>
        <td>Q3</td>
<td>$27,500</td>
    </tr>
<tr>
        <td>Q4</td>
<td>$32,500</td>
    </tr>
</table>

Consider the vector space R3 with two bases:

$$ \varepsilon = \{\varepsilon_1, \varepsilon_2, \varepsilon_3\}  in which  \varepsilon_1 = (1, 0, 0), \varepsilon_2 = (0, 1, 0), \varepsilon_3 = (0, 0, 1) $$

$$ \theta = \{\theta_1, \theta_2, \theta_3\}  in which  \theta_1 = (1, 1, 0), \theta_2 = (0, 1, 1), \theta_3 = (1, 0, 1) $$

a) Find the transition matrix from the basis $\varepsilon$ to the basis $\theta$.

b) Find the transition matrix from the basis $\theta$ to the basis $\varepsilon$.

a)
$$ P_{\varepsilon \to \theta} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} & -\frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \end{bmatrix} $$

b)
$$ P_{\theta \to \varepsilon} = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix} $$
32

# REFFERENCES

Clay, A. (2015). Introduction to linear algebra. University of Manitoba: https://adamjclay.github.io/linear_notes.pdf

Kin, E. (2025). Introduction to linear algebra: https://elijahkin.github.io/teaching/math240.pdf

Ricardo, H. (2009). *A modern introduction to linear algebra*. Press CRC.
Boby, M. a. K. (n.d.). The Sarrus Rule.docx. Scribd: https://fr.scribd.com/document/629312168/The-Sarrus-Rule-docx

Wikipedia contributors. (2025, December 2). *Laplace expansion*. Wikipedia. https://en.wikipedia.org/wiki/Laplace_expansion

Wikipedia contributors. (2025, December 14). *Cramer’s rule*. Wikipedia. https://en.wikipedia.org/wiki/Cramer%27s_rule

Khrushchev, S. (2024). Gauss-Jordan Elimination. In *Classroom companion: economics* (pp. 1–83). https://doi.org/10.1007/978-3-031-68682-5_1
33
