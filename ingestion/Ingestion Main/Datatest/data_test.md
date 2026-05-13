VIETNAM GENERAL CONFEDERATION OF LABOR

![0_image_0.png](0_image_0.png)

TON DUC THANG UNIVERSITY
FACULTY OF INFORMATION TECHNOLOGY
Tạ Chấn Nam - 52500174 MIDTERM **ESSAY**
LINEAR ALGEBRA FOR IT
HO CHI MINH CITY, 2025 VIETNAM GENERAL CONFEDERATION OF LABOR

![1_image_0.png](1_image_0.png)

TON DUC THANG UNIVERSITY
FACULTY OF INFORMATION TECHNOLOGY
Tạ Chấn Nam - 52500174 MIDTERM **ESSAY**
APPLIED CALCULUS FOR IT
Advised by Mr. Tran Ha Son HO CHI MINH CITY, 2025

# Acknowledgement

I would like to express our sincere gratitude to Mr. Tran Ha Son, our instructor and mentor, for his valuable guidance and support throughout the midterm report of our report about solving linear algebra logical question with relative knowledge. He is very helpful and patient in providing us with constructive feedback and suggestions to improve our work. Thanks to his friendly and harmonious personality, All of linear algebra lessons are very attractive and funny. I have learned a lot precious knowledge from his expertise, also experience in logical thinking, developed mindset and deeply understand about core value of study. I am very honored and privileged to have him as our teacher and supervisor. 

Ho Chi Minh city, 22nd December 2025. 

Author
(Signature and full *name)*
Nam Tạ Chấn Nam i

# Declaration Of Authorship

I hereby declare that this is our own report and is guided by Mr. Tran Ha Son; The content research and results contained herein are central and have not been published in any form before. The data in the tables for analysis, comments and evaluation are collected by the main author from different sources, which are clearly stated in the reference section. 

In addition, the report also uses some comments, assessments as well as data of other authors, other organizations with citations and annotated sources. 

If something wrong happens, Ill take full responsibility for the content of my **report.** Ton Duc Thang University is not related to the infringing rights, the copyrights that I give during the implementation process (if any). 

Ho Chi Minh city, 22nd December *2025* Author
(Signature and full *name)*
Nam Tạ Chấn Nam

# Abstract

The goal of this report is represent and explain method to solve logical question related to linear algebra by used fundamental knowledge about them. The report include 2 Chapter:
Chapter 1: This chapter has 2 part. Part 1 primarily introduce to basic knowledge of linear algebra will be used to solve questions consist of Definition, Formula. Next part will represent and explain detailed step to solve following question

Question 1: Given the matrix $mathbf{A}=begin{bmatrix}1&2&-1 2&2&1 1&2&aend{bmatrix}$. Find all values of $a$. 
. Find all values of a

for which det( A)=0. 

- Question 2: Solve the following system of linear equations by using Gaussian Elimination method. 

a)
b)

$$left{begin{array}{l l}{X+5y-2z=4} {3x-mathrm{y}+z=3} {5x+y-2z=4}end{array}right.$$
x + 5y 2z = 4
3x y + z = 3
5x + y 2z = 4
$$left{begin{array}{l l}{x+3-z=3} {x-2y+2z=4} {2x+y+z=7}end{array}right.$$
x + 3 z = 3
x 2y + 2z = 4
2x + y + z = 7
- Question 3: Let v1= 1;1;1 , v2= 2;5;1 , v3= 3;0;5 . Show that the set B= v1
, v2 v3 is a basis of R
3
. 

 $ $ Question 4: Find a matrix P that diagonalizes $ A=begin{bmatrix}1&2&2 2&1&1 0&0&1end{bmatrix}$. 
1 2 2
2 1 1
0 0 1
- Question 5: Let S = v1= 2;4;3 , v2 = 2;4;2 , v3= 6;4;2 . 

Find the coordinate vector of v = 54, 12, 9 relative to S.

- Question 6: Use the Gram-Schmidt orthonormalization process to transform S = v1 = 4;-8;8 , v2 = 8;8;4 , v3 = 8;-4;8 the for R
3 into an orthonormal basis. 

- Question 7: Consider the vector space R3 with two bases:
ε = ε1
, ε2
, ε3 in wihich ε1 = 1, 0, 0 , ε2 = 0, 1, 0 , 
ε3 = 0, 0, 1 θ = θ1
, θ2
, θ3 in wihich θ1 = 1, 1, 0 , θ2 = 0, 1, 1 , 
θ3 = 1, 0, 1 a) Find the transition matrix from the basis ε to the basis θ. b) Find the transition matrix from the basis θ to the basis ε Chapter 2: Show the correct answer of given question.

# Table Of Content

| CHAPTER 1. SOLUTION |
|-----------------------|
| 1.1 Introduction. |
| 1.2 Detailed Steps. |
| CHAPTER 2. RESULT |
| 2.1 Question 1. |
| 2.2 Question 2. |
| 2.3 Question 3. |
| 2.4 Question 4. |
| 2.5 Question 5. |
| 2.6 Question 6. |
| 2.7 Question 7. |
| REFFERENCES |

# Chapter 1. Solution

## 1.1 Introduction. - Linear Algebra.

Linear Algebra is the branch of mathematics that focuses on the study of vectors, vector spaces, matrices, and linear transformations. It deals with linear equations, linear functions, and their representations through matrices and determinants. It has a wide range of applications in Physics and Mathematics. It is the basic concept for machine learning and data science. 

## - Matrix.

In mathematics, a matrix (pl.: matrices) is a rectangular array of numbers or other mathematical objects with elements or entries arranged in rows and columns, usually satisfying certain properties of addition and multiplication. 

## - Diagonal Matrix.

In linear algebra, Diagonals Matrix is a square matrix that all elements except the main diagonal are zero. 

## - Identity Matrix.

An identity Matrix is a square matrix whose all diagonal elements are equal to 1 and the rest of the elements are zero.

## - Inverese Matrix.

The inverse of a matrix is a square matrix that, when multiplied by itself, results in the identity matrix I. 

A Matrix has its inverse if the determination of matrix non-zero The inverse of a Matrix "A", denoted as A
1
. 

A A
1 = A
1 A = 1

#### - System Of Linear Equations.

In mathematics, a system of linear equations (or linear system) is a collection of two or more linear equations involving the same variables. 

For example, consider A general system of m linear equations with n unknowns and coefficients can be written as:

a11x1+a21x2++a1nxn=b1
a12x1 + a11x2++a2nxn=b2
$$left{begin{array}{l l}{a_{11}x_{1}+a_{21}x_{2}+cdots+a_{1n}x_{n}=b_{1}} {a_{12}x_{1}+a_{11}x_{2}+cdots+a_{2n}x_{n}=b_{2}} {vdots} {a_{m1}x_{1}+a_{m2}x_{2}+cdots+a_{m n}x_{n}=b_{m}}end{array}right.$$

## Am1X1+Am2X2++Amnxn=Bm - Augmented Matrices.

Augmented Matrices are two matrices combined using their column values. Thus, if we have m columns in the first matrix and n columns in the second matrix, then in the Augmented Matrices we have (m + n)
columns. Augmented Matrices is used to solve simple linear equations. An Augmented Matrices has the same number of rows as there are variables in the given linear equations. 

An Augmented Matrices is a means to solve simple linear equations. The coefficients and constant values of the linear equations are represented as a matrix, referred to as an Augmented Matrices. In simple terms, the Augmented Matrices is the combination of two simple matrices along the columns. If there are m columns in the first matrix and n columns in the second matrix, then there would be m + n columns in the Augmented Matrices. 

 ## Coefficient Matrix A = $begin{bmatrix}mathbf{a}_{1} mathbf{a}_{2} mathbf{a}_{3}end{bmatrix}$ Constant Matrix B = $begin{bmatrix}mathbf{d}_{1} mathbf{d}_{2} mathbf{d}_{3}end{bmatrix}$ Variable Matrix X = $begin{bmatrix}mathbf{X} mathbf{y} mathbf{z}end{bmatrix}$
Consider 3 Matrix:

Coefficient Matrix A =
Constant Matrix B =
d1
d2
d3
Variable Matrix X =
x
y
z
$$begin{array}{r l r l}{mathbf{b}_{1}}&{{}dots}&{mathbf{c}_{1}} {mathbf{b}_{2}}&{{}dots}&{mathbf{c}_{2}} {mathbf{b}_{3}}&{{}dots}&{mathbf{c}_{3}}end{array}$$
a1 b1
... 
a2 b2
... 
a3 b3
... 
c1
c2
c3
The Augmented Matrices M is calculated as. 

$$mathbf{M}=(mathbf{A}timesmathbf{X}|mathbf{B})$$
M = A X B

## - Row Echelon Form.

of a matrix simplifies solving systems of linear equations, understanding linear transformations, and working with matrix equations. 

A matrix is in Row Echelon form if it has the following properties:
1. Zero Rows at the Bottom: If there are any rows that are completely filled with zeros they should be at the bottom of the matrix. 

2. Leading 1s: In each non-zero row, the first non-zero entry (called a leading entry) can be any non-zero number. It does not have to be 1.

3. Staggered Leading 1s: The leading entry in any row must be to the right of the leading entry in the row above it. 

Example about a Row Echelon Form:

$$mathbf{A}={begin{bmatrix}1&2&-1&4 0&4&0&3 0&0&1&2end{bmatrix}}$$
A =
1 2 1
0 4 0
0 0 1
4
3
2

#### - Linear Combination.

Given a set of vectors v1
, v2
, . . . , vn in a vector space, a linear combination of these vectors is an expression of the form:
w = c1v1 + c2v2 + . . . + cnvn Where c1
, c2
, . . . , cn are scalars (real numbers, complex numbers, etc.). 

Example of Linear Combination: Consider 2 vector:

$$mathbf{v}_{1}=begin{bmatrix}1 2end{bmatrix}, mathbf{v}_{2}=begin{bmatrix}3 4end{bmatrix}$$
v1 =
1
2
, v2 =
3
4
A linear combination of v1 and v2 would be:

$$mathbf{w}=mathbf{c}_{1}mathbf{v}_{1}+mathbf{c}_{2}mathbf{v}_{2}=mathbf{c}_{1}left[{begin{matrix}1 2end{matrix}}right]+mathbf{c}_{2}left[{begin{matrix}3 4end{matrix}}right]=left({begin{matrix}mathbf{c}_{1}+3mathbf{c}_{2} 2mathbf{c}_{1}+4mathbf{c}_{2}end{matrix}}right)$$
2 + c2
3
4 =
c1 + 3c2
2c1 + 4c2

## - Coordinate Vector.

In a vector space, any vector can be written as a linear combination of a basis. The coefficients of the linear combination are called the coordinates of the vector with respect to the basis. 

Let S be a finite-dimensional linear space. Let C = c1
, c2
, ..., cn be a basis for S. For any s S , take the unique set of k scalars v1
, ..., vn such that

$$mathbf{s}=mathbf{c}_{1}mathbf{v}_{1}+ldots+mathbf{c}_{mathrm{{n}}}mathbf{v}_{mathrm{{n}}}$$
s = c1v1 + . . . + cnvn 
Then, the n 1 vector

vn

$$[mathbf{s}]_{mathrm{{B}}}=$$

s B =

![11_image_0.png](11_image_0.png)

v1

## - Gaussian Elimination. - Gauss-Jordan Elimination.

is called the coordinate vector of C with respect to the basis. 

Gaussian elimination is a row reduction algorithm for solving linear systems. It involves a series of operations on the Augmented Matrices (which includes both coefficients and constants) to simplify it into a row echelon form or reduced row echelon form. This method can also help in determining the rank, determinant and inverse of matrices. Gaussian elimination is a method for solving systems of equations in matrix form. Elementary Row Operations:
1. Interchanging Rows: Swap two rows. 

2. Multiplying a Row by a Scalar: Multiply all elements of a row by a non-zero number. 

3. Adding a Scalar Multiple of One Row to Another: Add or subtract a multiple of one row. 

Categories of Linear Equation Systems:
1. Consistent Independent System: Has exactly one solution. 

2. Consistent Dependent System: Has infinite solutions. 

3. Inconsistent System: Has no solution. 

Gauss-Jordan elimination method is a row reduction algorithm. This is an advanced version of Gaussian elimination but instead of eliminating just element under main diagonal line, its also eliminate upper part to become a diagonal matrix or identity matrix. 

## - Determinant.

In mathematics, the determinant is a scalar-valued function of the entries of a square matrix. The determinant of a matrix A is commonly denoted det(A), det A. 

## - Rule Of Sarrus.

The Rule of Sarrus is a mnemonic device for computing the determinant of a 33 matrix named after the French mathematician Pierre Frédéric Sarrus. 

Consider a 3 3 Matrix. 

$$mathbf{M}=begin{bmatrix}mathbf{a}&mathbf{b}&mathbf{c} mathbf{d}&mathbf{e}&mathbf{f} mathbf{g}&mathbf{h}&mathbf{i}end{bmatrix}$$
M =
a b c
d e f
g h i
Write out the first two columns of the matrix to the right of the third column, giving five columns in a row. Then add the products of the main diagonals and its parallel diagonals and going from top to bottom and subtract the products of the sub diagonals and its parallel diagonals going from bottom to top. This yields. 

$$mathrm{Det(A)=left|begin{array}{l l l}{mathrm{a}}&{mathrm{b}}&{mathrm{c}} {mathrm{d}}&{mathrm{e}}&{mathrm{f}} {mathrm{g}}&{mathrm{h}}&{mathrm{i}}end{array}right|=(mathrm{aei,+dhc,+bfg)-(ceg+fha+bdi)}$$
= (aei + dhc + bfg) (ceg + fha + bdi)

## - Laplace Expansion.

In linear algebra, The Laplace Expansion, also called Cofactor expansion is an expression of the determinant of an n n matrix B as a weighted sum of minors, which are the determinants of some (n 1) (n 1)
submatrices of B. Specifically, for every i, the Laplace expansion *along* the i th row is the equality.

We have 2 way to perform Laplace Expansion and their General Formula. 

1. Expansion along to column 2. Expansion along to row $$text{Det(B)}=sum_{text{j}=1}^{text{n}}left(-1right)^{text{i}+text{j}}text{b}_{text{i},text{j}}text{m}_{text{i},text{j}}$$ $$text{Det(B)}=sum_{text{i}=1}^{text{n}}left(-1right)^{text{i}+text{j}}text{b}_{text{i},text{j}}text{m}_{text{i},text{j}}$$
(1)
i + j bi, jmi, j
where bi, j is the entry of the i th row and j th column of B, and mi, j is the determinant of the submatrix obtained by removing the i th row and the j th column of B. Similarly, the Laplace expansion along the j th *column* is the equality. 

The Laplace Expansion and The Rule of Sarrus also can convert to each others. 

Det(A) = a b c d e f g h i a e f h i + d b c
= (aei + dhc + bfg) (ceg + hfa + bdi)
aei + dhc + bfg ceg hfa bdi
a(ei hf) + d(bi hc) + g(bf ce)
h i + g 
b c
h i 

## - The CrammerS Rule.

In linear algebra, Cramers rule is an explicit formula for the solution of a system of linear equations with as many equations as unknowns, valid whenever the system has a unique solution. It expresses the solution in terms of the determinants of the (square) coefficient matrix and of matrices obtained from it by replacing one column by the column vector of right-sides of the equations Consider a system of n linear equations for n unknowns, represented in matrix multiplication form as follows:

# Ax=B

where the n n matrix A has a nonzero determinant, and the vector x= x1
, x2
, . . . ,xn T
is the column vector of the variables. Then the theorem states that in this case the system has a unique solution, whose individual values for the unknowns are given by:

xi =
$$mathbf{x}_{mathrm{i}}={frac{operatorname*{det}(A_{i})}{operatorname*{det}(mathbf{A})}}=,i=1,!!ldots,n$$
det(A) = i = 1,..., n Where A is the matrix formed by replacing the i-th column of A by the column vector b

## - Linear Independence And Dependence.

In a vector space, a set of vectors is said to be linearly independent if no vector in the set can be expressed as a linear combination of the other vectors in the set. 

A set of vectors {v1
, v2
, . . . ,vn} is linearly independent if the equation:
The expression c1v1 + c2v2 + . . . + cnvn = 0 has only the trivial solution c1 = c2= . . . = cn = 0. 

In contrast, if there exist non-zero scalars c1
, c2
, ..., cn such that the equation above holds, then the set of vectors is linearly dependent. 

## - Vector Space.

a vector space (also called a linear space) is a set whose elements, often called vectors, can be added together and multiplied ("scaled") by numbers called scalars. The concept of vector spaces is fundamental for linear algebra, together with the concept of matrices, which allows computing in vector spaces. This provides a concise and synthetic way for manipulating and studying systems of linear equations. 

Vector spaces generalize Euclidean vectors, which allow modeling of physical quantities have not only a magnitude, but also a direction Vector spaces are characterized by their dimension, which, roughly speaking, specifies the number of independent directions in the space. This means that for two vector spaces over a given field and with the same dimension, the properties that depend only on the vector-space structure are exactly the same. 

## - Basis Of Matrix.

In mathematics, a set B of elements of a vector space V is called a basis if it was satisfied 2 privileged condition. 

1. Every element of v can be written in a unique way as a finite linear combination of the elements of B. The coefficients of this linear combination are called the components or coordinates of the vector with respect to B. The elements of a basis are called the basis vectors. 

2. The elements of B are linearly independent, and every element of v is a linear combination of the elements of B. In other words, a basis is a linearly independent spanning set. 

## - Change Of Basis.

Change of basis transtition matrix is coordinate transtition matrix of a vector from this basis to other Transition matrix PST is coordinate matrix P along to S. 

To find the change of basis transition Matrix, we need to find coordinate vector T with basis S, then write the matrix follow to a rule. 

Consider 3 Matrix:
u1 = a, d, g , u2 = b, e, h , u3 = c, f, i

$$mathbf{}D$$

a b c

x y z

P =
$$begin{array}{l l}{e}&{f} {mathrm{h}}&{i}end{array}$$
d e f
g h i
v1 = A, D, G , v2 = B, E, H , v3 = C, F, I

$$T{equiv};{binom{A}{D}}$$
T=
$$0,,G),,V_{2}$$
$$begin{array}{c c}{{B}}&{{C}} {{E}}&{{F}} {{H}}&{{I}}end{array}$$
A B C
D E F
G H I
Unknowned Variable Matrix X =
Each vector of T would be the constant matrix to Matrix P or P X Tv1
, , P X Tv3 and thus the corefficient matrix is the set of variable vector computed in each case of Matrix P with Tv1
, , Tv3
. 

## - Characteristic Polynomial.

In linear algebra, the characteristic polynomial of a square matrix is a polynomial which is invariant under matrix similarity and has the eigenvalues as roots. It has the determinant and the trace of the matrix among its coefficients. The characteristic polynomial of an endomorphism of a finite-dimensional vector space is the characteristic polynomial of the matrix of that endomorphism over any basis (that is, the characteristic polynomial does not depend on the choice of a basis)
Consider an matrix An n, identity matrix In n and the eigenvalue λ The characteristic polynomial of A denoted by pA(λ) is the polynomial defined by pA(t) = det(λI A)
where I denoted n n identity matrix. 

The characteristic equation, also known as the determinant equation, is the equation obtained by equating the characteristic polynomial to zero.

# Det(A Λi) = 0

## - Eigenvalues And Eigenvectors.

Eigenvalues and eigenvectors are fundamental concepts in linear algebra, used in various applications such as matrix diagonalization, stability analysis, and data analysis. They are associated with a square matrix and provide insights into its properties. 

Eigenvalues are unique scalar values linked to a matrix. They indicate how much an eigenvector gets stretched or compressed during the transformation. The eigenvectors direction remains unchanged unless the eigenvalue is negative, in which case the direction is simply reversed. 

The equation for eigenvalue and eigenvalue are given by Av = λv where, A is the given square matrix, v is associated eigenvector and λ is scalar eigenvalue. 

Eigenvectors are non-zero vectors that, when multiplied by a matrix, only stretch or shrink without changing direction. The eigenvalue must be found first before the eigenvector. For any square matrix A of order n n, the eigenvector is a column matrix of size n 1. This is known as the right eigenvector, as matrix multiplication is not commutative. 

The equation for eigenvalue and eigenvector are given by Av = λv where, A is the given square matrix, v is the eigenvector and λ is any scalar multiple.

## - Algebraic And Geometric Multiplicity.

The Algebraic Multiplicity is the number of times a specific eigenvalue appears in the polynomial equation that is produced by characteristic equation denoted by ma( λ ) . 

The Geometric Multiplicity tells us the number of the linearly independent eigenvectors associated with the eigenvalue. To create a eigenspace which is a set of eigenvectors corresponding to the particular eigenvalue along with the zero vector denoted by mg( λ ). 

## - Matrix Diagonalization.

Matrix diagonalization is the process of reducing a square matrix into its diagonal form using a similarity transformation. Not all matrices are diagonalizable. A matrix is diagonalizable if it has no defective eigenvalues, meaning each eigenvalues geometric multiplicity is equal to its algebraic multiplicity. 

A Diagonals matrix of an any matrix is the result of the combinimation theorical knowledges mentioned above include Characteristic Polynomial and Equation, Eigenvalues and Eigenvectors, Algebraic and Geometric Multiplicity, Inverese Matrix, Identity Matrix defined by a general formula

## D = P 1Ap

Where, A is the given square matrix, P is a modal matrix which is consist of eigenvectors created by eigenvalues found form characteristic equation, P
1 is the inverse matrix of P and D is the diagonals matrix of A.

## - Gram-Schmidt Orthonormalization Process.

In mathematics, particularly linear algebra and numerical analysis, the GramSchmidt process or Gram-Schmidt algorithm is a way of finding a set of two or more vectors that are perpendicular to each other. 

By technical definition, it is a method of constructing an orthonormal basis from a set of vectors in an inner product space, most commonly the Euclidean space R
n equipped with the standard inner product. The GramSchmidt process takes a finite, linearly independent set of vectors S= u1
,,uk for k n and generates an orthogonal set S= f1
,,fk that spans the same k-dimensional subspace of R
n as S. 

General Formula to find orthogonal basis:
f1 = u1

$$mathbf{f}_{2}=mathbf{u}_{2}-{frac{mathbf{u}_{2}timesmathbf{f}_{1}}{|mathbf{f}_{1}|^{2}}}timesmathbf{f}_{1}$$
f2 = u2 
u2f1
f1
2 f1
f3 = u3 
$$mathrm{f}_{3}=mathrm{u}_{3}-,{frac{mathrm{u}_{3}timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}timesmathrm{f}_{1}-,{frac{mathrm{u}_{3}timesmathrm{f}_{2}}{|mathrm{f}_{2}|^{2}}}timesmathrm{f}_{2}$$
2 f1
u3f2
f2
2 f2
$$Longrightarrowmathrm{f}_{mathrm{k}}=mathrm{u}_{mathrm{k}}-{frac{mathrm{u}_{mathrm{k}}timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}timesmathrm{f}_{1}-{frac{mathrm{u}_{mathrm{k}}timesmathrm{f}_{2}}{|mathrm{f}_{2}|^{2}}}timesmathrm{f}_{2}-ldots-{frac{mathrm{u}_{mathrm{k}}timesmathrm{f}_{(mathrm{k}-1)}}{|mathrm{f}_{(mathrm{k}-1)}|^{2}}}timesmathrm{f}_{(mathrm{k}-1)}$$
2 f2...
2 f(k1)
Orthonormal basis is the Orthogonal basis after normalized by formula:

$${mathrm{NormalizedS^{prime}=}}left({frac{mathrm{f_{1}}}{|mathrm{f_{1}}|}},...,{frac{mathrm{f_{k}}}{|mathrm{f_{k}}|}}right).$$
Normalized S=
f1
f1
,, fk
fk
. 

## - Rank Of Matrix.

The Rank of a Matrix is the maximum number of linearly independent rows or columns in a matrix denoted by ρ(A) or rank(A). It essentially determines the dimensionality of the vector space formed by the rows or columns of the matrix. If a matrix has all rows with zero elements, then the rank of a matrix is said to be zero.

## 1.2 Detailed Steps. - Question 1.

Method 1: By Using the rule of *Sarrus* in matrix A, we have a single-variable equation in term of "a" to find *determinant*. Given that the determinant of A equal to zero. We can generate the expression in term of "a" to find its real value Step 1. Find the expression for the Determinant in term of "a". 

$$operatorname*{det}(mathbf{A})={begin{vmatrix}1&2&-1 2&2&1 1&2&mathbf{a}end{vmatrix}}$$
det(A) =
1 2 1
2 2 1
1 2 a 
= (1 2 a + 2 2 (1) + 2 1 1) 
$$begin{array}{r l}{={}}&{{}(1times2timesmathrm{a}+2times2times(-1),+2times1times1)} {-}&{{}} {-}&{{}((-1)times2times1+1times2times1+2times2timesmathrm{a})}end{array}$$
((1) 2 1 + 1 2 1 + 2 2 a) 
$$begin{array}{r l}{={}}&{{}(2mathbf{a}-2)-4mathbf{a}}end{array}$$
= (2a 2) 4a 
$$mathbf{partial}=mathbf{partial}-2mathbf{a}-2mathbf{partial}$$
= 2a 2
Step 2. From the Determinant expression in term of "a" has been just found, we set it equal to zero then find the real value of "a". 

det(A) = 0 2a 2 a = 1 Method 2: By using the Laplace *expansion* method. Firstly, we eliminate value in column 1 with elementary row *operations* and try to convert it to zero then follow along to the Laplace formula.

Step 1. Find the expression in term of "a" through Laplace formula. 

$$operatorname*{det}(mathbf{A})={begin{vmatrix}1&2&-1 2&2&1 1&2&mathbf{a}end{vmatrix}}{begin{array}{l}{(mathrm{row}_{2}-2mathrm{row}_{1})} {mathrm{row}_{3}-mathrm{row}_{1})} {mathrm{ab}}end{array}}{begin{vmatrix}1&2&-1 0&-2&3 0&0&mathbf{a}+1end{vmatrix}}$$ $$=1times{begin{vmatrix}-2&3 0&mathbf{a}+1end{vmatrix}}+ 0times{begin{vmatrix}2&-1 0&mathbf{a}+1end{vmatrix}}+0times{begin{vmatrix}2&-1 -2&3end{vmatrix}}$$

1 2 1
0 2 3
0 0 a+1 
0 a + 1 + 0 
2 1 
2 3 
$$mathbf{partial}=-2mathbf{a}-2$$
= 2a 2
Step 2. From the determinant expression in term of "a" has been just found, we set it equal to zero then find the real value of "a". 

det(A) = 0 2a 2 a = 1

Method 1:
Firstly we turn the system of linear *equations* into its augmented *matrices* form. Then by using Gaussian *elimination* method, we convert it to Row electron *form* with elementary row *operations*. Finally turn it back *system* of linear *equations* form and find categories of linear equation *systems* thus conclude the solution of the System of Linear Equations. 

a)
Step 1. Convert the system of linear equations to augmented matrices and find its echelon form of matrix A. 

$$begin{bmatrix}mathbf{A}mathbf{X}&mathbf{|}&mathbf{B}end{bmatrix}=begin{bmatrix}1&5&-2&4 3&-1&1 5&1&-2&4end{bmatrix}.$$
AX | B =
1 5 2
3 1 1
5 1 2
4
3
4
$${begin{bmatrix}1&5&-2 3&-1&1 5&1&-2end{bmatrix}}{begin{bmatrix}4 3 4end{bmatrix}} (mathrm{row}_{2}-3mathrm{row}_{1}) (mathrm{row}_{3}-5mathrm{row}_{1})end{bmatrix}}{begin{bmatrix}1&5&-2 0&-16&7 0&-24&8end{bmatrix}} {begin{bmatrix}4 -9 -16end{bmatrix}}$$

1 5 2
0 16 7
0 24 8
9 
16
$${begin{bmatrix}1&5&-2 0&-16&7 0&-24&8end{bmatrix}} {begin{bmatrix}4 -9 -16end{bmatrix}} {begin{bmatrix}1&5&-2 0&-16&7 0&-6&2end{bmatrix}} {begin{bmatrix}4 -9 -4end{bmatrix}}$$

4 1 5 2
0 16 7
0 6 2
9 
$${begin{bmatrix}1&5&-2 0&-16&7 0&-6&2end{bmatrix}}{begin{bmatrix}4 -9 -4end{bmatrix}}{begin{bmatrix}1&5&-2 0&-6&2 0&0&5end{bmatrix}}{begin{bmatrix}4 -4 -5end{bmatrix}}$$

1 5 2
0 6 2
0 0 5
4 
5
 The system of linear equations is a consistent independent so it has exactly one solution. 

Step 2. Turn eliminated matrix back to its system of linear equations and find the solution. 

$$left{{begin{array}{l}{mathrm{x,+5y-2z=4}} {-6,mathrm{y+2z=-4implies}} {5mathrm{z=5}}end{array}}right.left{{begin{array}{l}{mathrm{x=1}} {mathrm{y=1}} {mathrm{z=1}}end{array}}right.$$
x + 5y 2z = 4 
6 y + 2z = 4
5z = 5

x = 1
y = 1
z = 1
b)
Step 1. Convert the system of linear equations to augmented matrices and find its echelon form

$$[{mathrm{AX}}quad|quadmathrm{B}]=begin{bmatrix}1&3&-1 1&-2&2 2&1&1end{bmatrix}begin{bmatrix}3 4 7end{bmatrix}$$
AX | B =
1 3 1
1 2 2
2 1 1
3
4
7
 $begin{bmatrix}1&3&-11&-2&22&1&1end{bmatrix}$ $begin{bmatrix}1&3&-10&-5&30&-5&3end{bmatrix}$ $implies$ The svstq
1 3 1
1 2 2
2 1 1
$begin{bmatrix}3 4 7end{bmatrix}:begin{pmatrix}text{row}_2-text{row}_1 longrightarrow text{(row}_3-2text{row}_1)end{bmatrix}begin{bmatrix}1 0 0end{bmatrix}$ $begin{bmatrix}3 1 1end{bmatrix}:begin{pmatrix}text{row}_3-text{row}_2 longrightarrow 0end{bmatrix}begin{bmatrix}1 0 0end{bmatrix}$ n of linear equatior
(row2 row1)
(row3 2row1)
1 3 1
0 5 3
0 5 3
(row3 row2) 1 3 1
$$begin{array}{r r r}{{3}}&{{-1}} {{-5}}&{{3}} {{-5}}&{{3}}end{array}left|begin{array}{c}{{3}} {{1}} {{1}}end{array}right|$$
1 3 1
0 5 3
0 5 3
3
1
1

$$begin{array}{r l}{3}&{{}-1} {-5}&{3} {0}&{0}end{array}left|begin{array}{l}{3} {1} {0}end{array}right|$$
0 5 3
0 0 0
3
1
0

 The system of linear equations is consistent dependent so it has infinite solutions. We consider an arbitrary value and assign it to an unknown arbitrary variable in the system of linear equations called T. 

Step 2. Find the solution. 

x + 3y z = 3 
5y + 3z = 1
$${left{begin{array}{l l}{mathrm{x,+3y-z=3}} {-5mathrm{y+3z=1}} {mathrm{z=T}}end{array}right.}Longrightarrow{left{begin{array}{l l}{mathrm{x,={frac{-18-4T}{5}}}} {mathrm{y={frac{-1+3T}{5}}}} {mathrm{z=T}}end{array}right.}$$

x =
184T
5
y =
1+3T
5
z = T
Method 2:
First of all, we need to turn the linear *system* into its augmented *matrices*, then we need to check whether the linear system is *independent* or not by the *determinant*. If it is linear independent system , we can using the Crammers *rule* to infer the solution of the linear system. 

a)

Step 1. Find the Determinant of A:
# Find the Determinant of $mathbf{A}$: . 
$${begin{array}{r l}{{mathrm{consider}}[mathrm{AX}quad|quadmathrm{B}]=}{left[{begin{matrix}1&5&-2&4 3&-1&1 5&1&-2end{matrix}}right|left.{begin{matrix}4 3 4end{matrix}}right]} {operatorname*{det}(mathrm{A})=}{left|{begin{matrix}1&5&-2 3&-1&1 5&1&-2end{matrix}}right|} {operatorname*{det}(2+25-6)-(10+1-30)=}40end{array}}right|}$$
1 5 2
3 1 1
5 1 2
4
3
4
= (2 + 25 6) (10 + 1 30) =40
Step 2. Because the Determinant is non-zero so it has unique solution so we can applying Cramers Rule successively substituting B into the columns of matrix A, we can find the Determinant of Ax

 $;$ Determinant is non-2 we can applying Cram B into the columns of . 
, Ay
, Az
. 

$$begin{split}det(mathbf{A_{x}})&=begin{vmatrix}4 3 4end{vmatrix} &=(xi)end{split}$$

det(Ax
$$={left|begin{matrix}{4}&{5}&{-2}{3}&{-1}&{1}{4}&{1}&{-2}end{matrix}right|}$$ $$=(8-6+20)-(8+4-30)=40$$
= (8 6 + 20) (8 + 4 30) = 40
det(Ay
$$={left|begin{array}{l l l}{1}&{4}&{-2} {3}&{3}&{1} {5}&{4}&{-2}end{array}right|}$$ $$=(-6-24+20)-(-30+4-24)=40$$
$$operatorname*{det}(mathrm{A_{y}})=$$
= (6 24 + 20) (30 + 4 24) = 40
$$={left|begin{matrix}1&5&4 3&-1&3 5&1&4end{matrix}right|}$$ $$=(-4+12+75)-(-20+60+3)=40$$
$$operatorname*{det}(mathrm{A_{z}})=$$
= (4 + 12 + 75) (20 + 60 + 3) = 40
Cramers formula. 

Step 3. Find the solution of the linear system of equations along to

x =
$$begin{array}{l}{mathrm{{dot{x}=frac{operatorname*{det}(A_{x})}{operatorname*{det}(A)}=frac{40}{40}=1}}} {mathrm{{y=frac{operatorname*{det}(A_{y})}{operatorname*{det}(A)}=frac{40}{40}=1}}} {mathrm{z=frac{operatorname*{det}(A_{z})}{operatorname*{det}(A)}=frac{40}{40}=1}}end{array}$$
det(A) =
40
40 = 1
y =
det(A) =
40
40 = 1
z =
det(A) =
40
40 = 1 b)
Step 1. Find the determinant of A:

$${begin{array}{l}{{mathrm{consider}}[mathrm{A}quad|quadmathrm{B}]=left[{begin{matrix}1&3&-1&3 1&-2&2&4 2&1&1&7end{matrix}}right]} {operatorname*{det}(mathrm{A})=left|{begin{matrix}1&3&-1 1&-2&2 2&1&1end{matrix}}right|}end{array}}$$
1 3 1
1 2 2
2 1 1
3
4
7
$$=(-2-1+12)-(4+2+3)=0$$
= (2 1 + 12) (4 + 2 + 3) =0
Step 2. Because A already isnt linear independent so we can not use The Cramers Rule to find the solution for the system of linear equations. Therefore, we should be use the Gaussian elimination method alternatively mentioned at method 1. 

Step 3. According to method 1 the solution of this system of linear equations is

$${left{begin{array}{l l}{mathrm{x,+3y-z=3}} {-5mathrm{y+3z=1}} {mathrm{z=T}}end{array}right.}Longrightarrow{left{begin{array}{l l}{mathrm{x,={frac{-18-4T}{5}}}} {mathrm{y={frac{-1+3T}{5}}}} {mathrm{z=T}}end{array}right.}$$

x =
184T
5
y =
1+3T
5
z = T
Method:
We need two condition to prove set B is a *Basis* of R
3
. Firstly, B must be a linear inderpendent *matrix*, we can prove it through using Determinant. Secondly, by using Gaussian *elimination* to convert B to its row echelon form then compare the number of non-zero row in matrix B (rank(B)) with the number of dimension in R space(dim( R
3
)). if both of them is equal so we can conclude B is a basis of R
3
. 

Step 1. Find the determinant of B by using the rule of Sarrus. 

$$operatorname*{det}(mathbf{B})=begin{vmatrix}1&2&3 1&-5&0 1&1&5end{vmatrix}$$
det(B) =
1 2 3
1 5 0
1 1 5 
= (1 (5) 5 + 1 1 3 + 2 0 1 
$$begin{array}{r l}{={}}&{{}(1times(-5)times5+1times1times3+2times0times1} {-}&{{}} {-}&{{}(3times(-5)times1+1times0times1+2times1times5)}end{array}$$
(3 (5) 1 + 1 0 1 + 2 1 5) 
$$mathbf{partial}=mathbf{partial}-17$$
= 17
Because the determinant of matrix is non-zero so its linear independent. Thus, we satisfied first condition Step 2. Find rank of matrix B

B = 1 2 3 1 5 0 1 1 5 (row2 row1) (row3 row1) 1 2 3 0 7 3 0 1 2
$$begin{bmatrix}1 1 1end{bmatrix}$$
$$mathbf{B}$$
$$begin{bmatrix}1 0 0end{bmatrix}$$

1 2 3
0 7 3
0 1 2

(row2 7row3) 1 2 3
0 0 17
0 1 2
$${begin{array}{r l}{2}&{3} {0}&{-17} {-1}&{2}end{array}}{left[begin{array}{l l}{(mathrm{row}_{2}longleftrightarrowmathrm{row}_{3})} {longrightarrow} {0}&{-1} {0}&{0}end{array}begin{array}{l l}{2}&{3} {2} {-17}end{array}right]}$$
$$begin{bmatrix}1 0 0end{bmatrix}$$

(row2 row3) 1 2 3
0 1 2
0 0 17
rank(B) = dim(R
3
) = 3, because matrix B has satisfied both need condition so we can conclude that B is the basis of R
3

Method:
Firstly, we consider identity matrix I, e*igenvalue* λ then construct a characteristic *equation* between matrix A and λI. After that we find value of λ and determine algebraic *multiplicity* and with each value of λ we alternate it into characteristic equation and using the Gaussian elimination to find unknown variable and determine geometric multiplicity. If algebraic *multiplicity* equal to geometric *multiplicity,* we conclude that matrix A is diagonalizable. Therefore, we can infer eigenvectors. The set of result eigenvectors P = v1,, v2
, v3 will be the matrix which is able to diagonalize A. 

Step 1. Find eigenvalue and algebraic of multiplicity. 

$mathbf{v}$
PA( λ) = Det(A λI) = 0

$$Longleftrightarrow{left|begin{array}{l l l}{1!-!lambda}&{2}&{2} {2}&{1!-!lambda}&{1} {0}&{0}&{1!-!lambda}end{array}right|}=0$$

1λ 2 2
2 1λ 1
0 0 1λ 
= 0
$$begin{array}{l}{{Leftrightarrow((1-lambda)^{,3}+2times0times2+2times1times0)}} {{-}} {{qquad(2times(1-lambda)times0+1times0times(1-lambda)+2times2times(1-lambda))}}end{array}$$
(2 (1 λ) 0 + 1 0 (1 λ) + 2 2 (1 λ)) 
$$mathbf{bar{theta}}=0$$
= 0
$$begin{array}{r l}{Longleftrightarrow}&{{}(1-lambda)^{,3}-4+4,lambda=0}end{array}$$
 (1 λ)
3 4 + 4 λ = 0
$$Leftrightarrow 1-3,lambda,+,3,lambda^{2}-lambda^{3}-4+4,lambda=0$$
 1 3 λ + 3 λ
2 λ
3 4 + 4 λ = 0
$$iff -,lambda^{3}+3,lambda^{2}+,lambda-3=0$$
 λ
3 + 3 λ
2 + λ 3 = 0
$$Longrightarrow{begin{cases}lambda_{,1}=-1 lambda_{,2}=3 lambda_{,3}=1end{cases}}Longrightarrowoperatorname{m_{a}}(,lambda,)=1$$

λ 1= 1
λ 2 = 3
λ 3= 1
 ma( λ ) = 1
Step 2. Find eigenvectors and geometric multiplicity. 

$$lambda_{,,1}=-1$$

λ 1= 1
$$begin{array}{l}text{E(1)}=[(text{A}-lambda_{1}text{I})text{X}mid0]=begin{bmatrix}2&2&2 2&2&1 0&0&2end{bmatrix} begin{bmatrix}2&2&2 2&2&1 0&0&2end{bmatrix}begin{bmatrix}text{row}_{2}-text{row}_{1} rightarrow end{bmatrix}begin{bmatrix}2&2&2 0&0&-1 0&0&2end{bmatrix} begin{bmatrix}2&2&2 0&0&-1 0&0&2end{bmatrix}begin{bmatrix}text{row}_{3}+2text{row}_{2} rightarrow end{bmatrix}begin{bmatrix}2&2&2 0&0&-1 0&0&0end{bmatrix}=begin{bmatrix}2text{x}+2text{y}+2text{z}=0 -text{z}=0end{bmatrix} begin{bmatrix}text{x}=-text{T} text{y}=text{T} text{z}=0end{bmatrix}end{array}$$
=
2x + 2y + 2z = 0 
z = 0
$$mathrm{E(1)=(neg T,T,0mid Tin R)=Tleft[begin{array}{l}{{-1}} {{1}} {{0}}end{array}right]Rightarrow E(1)=operatorname{span}(v_{1}),m_{g}(lambda_{1})=1$$
 E(1) = span(v1), mg( λ 1) = 1
$$lambda_{2}=3$$
λ 2= 3
E(2) = (A λ 2I )X | 0 = 2 2 2 2 2 1 0 0 2 2 2 2 2 2 1 0 0 2 (row2 + row1) 2 2 2 0 0 3 0 0 -2 2 2 2 0 0 3 0 0 -2 (3row3 + 2row2) 2 2 2 0 0 3 0 0 0 = We consider y = T x = T y = T z = 0 E(2) = (T, T, 0 | T R) = T 1 1 0
2x + 2y +2z = 0
3z = 0
 E(2) = span(v2), mg( λ 2) = 1
λ 3= 1

$$mathrm{E}(3)=[(mathrm{A}-lambda_{3}mathrm{I})mathrm{X}mid0]=begin{bmatrix}0&2&2 2&0&1 0&0&0end{bmatrix}$$
E(3) = (A λ 3I)X | 0 =
0 2 2
2 0 1
0 0 0
$begin{bmatrix}0&2&2 2&0&1 0&0&0end{bmatrix}xrightarrow{text{(row}_{2}leftrightarrowtext{row}_{1})}begin{bmatrix}2&0&1 0&2&2 0&0&0end{bmatrix}=begin{bmatrix}2text{x}+text{z}=0 2text{y}+2text{z}=0end{bmatrix}$ We consider $text{x}=text{T}impliesbegin{cases}text{x}=text{T} text{y}=2text{T} text{z}=-2text{T}end{cases}$ $text{E}(3)=(text{T},2text{T},,text{-}2text{T}midtext{T}intext{R})=text{T}begin{bmatrix}1 2 -2end{bmatrix}Rightarrowtext{E}(3)=text{span}(text{v}_{3}),,,text{m}_{text{g}}(lambda_{3})=1$
 E(3) = span(v3), mg( λ 3) = 1
Step 3. Find matrix P which diagonalize A. 

$$mathbf{P}=left(mathbf{v}_{1,},mathbf{v}_{2,},mathbf{v}_{3,}right)={left[begin{array}{l}{-1} {1} {0}end{array}right]}$$
P= v1,, v2,, v3, =
$$begin{array}{l l}{{left[begin{array}{l l}{1}&{1} {1}&{2} {0}&{-2}end{array}right]}}end{array}$$
1 1 1
1 1 2
0 0 2
Because both value of algebraic multiplicity and geometric multiplicity are equal so matrix A is diagonalizable. Then to ensure that P is really the matrix which is diagonalize A, we consider the result of the diagonalization matrix general formula, if the multiply between P, A and the inverse matrix of P was a diagonal matrix called D so the matrix P which have just found is correct answer D=P
1AP 

$$begin{array}{l}{{vdotsquadquadfrac{1}{4}}} {{vdotsquadquadfrac{3}{4}}} {{0quad-frac{1}{2}]}}end{array}timesbegin{bmatrix}1&2&2 2&1&1 0&0&1end{bmatrix}timesbegin{bmatrix}-1&1&1 1&1&2 0&0&-2end{bmatrix}$$
$$begin{array}{c}{{-frac{1}{2}}} {{frac{1}{2}}} {{0}}end{array}$$
$$mathbf{Sigma}=mathbf{Sigma}$$

1 1 1
1 1 2
0 0 2 
$$={left[begin{array}{l}{-1} {0} {0}end{array}right]}$$
=
$$begin{array}{r l}{|}&{{}0|} {|}&{{}0|} {|}&{{}1|}end{array}$$
1 0 0
0 3 0
0 0 1
Method: The main method to solve this question is find coordinate *vector* v relative to S is build a system of linear *equations* around them and turn it into augmented *matrices* form. Then by using Gaussian elimination method to convert matrix to its row electron *form* and determine the categories of linear equation *system*. If it was a independent consistent system, the solution would be the coordinate *vector* of v relative to S, unless there is no coordinate *vector* of v relative to S. 

Step 1. Find echelon form of augmented matrices. 

$$[mathrm{SX}quad|quadmathrm{v}]={begin{bmatrix}2&2 4&4 3&2end{bmatrix}}$$
SX | v =
$$begin{array}{c|c}{{6}}&{{54}} {{4}}&{{12}} {{2}}&{{9}}end{array}$$
2 2 6
4 4 4
3 2 2
54
12
9
$${begin{bmatrix}1&1&-3 0&0&4 0&-1&11end{bmatrix}}{begin{array}{l}{(mathrm{row}_{2}longleftrightarrowmathrm{row}_{3})} {-24} {-72}end{array}} {begin{bmatrix}1&1&-3 0&-1&11 0&0&4end{bmatrix}}{begin{array}{l}{-72} {-24}end{array}}$$

(row2 row3) 1 1 3
0 1 11
0 0 4
27 
72 
24
 The system of linear equations is a consistent independent system so it has exactly one solution. 

Step 2. Find solution of augmented matrix. 

$${left{begin{array}{l l}{mathrm{x+y-3z=27}} {-mathrm{y+11z=-72}} {4mathrm{z=-24}}end{array}right.}Longrightarrow{left{begin{array}{l l}{mathrm{x=3}} {mathrm{y=6}} {mathrm{z=-6}}end{array}right.}Longrightarrow{mathrm{coordinatevector}}[mathrm{v}]_{mathrm{s}}={left[begin{array}{l}{3} {6} {-6}end{array}right]}$$
 coordinate vector v s =
3
2 2 6 4 4 4 3 2 2
$${begin{array}{r}{{frac{operatorname{row_{2}}}{4}}} {{frac{operatorname{row_{1}}}{2}}} end{array}}begin{bmatrix}1&1&-3&27 1&1&1&3 3&2&2&9end{bmatrix}$$ $${begin{array}{r}{7} {0} {10} {-1}end{array}}begin{bmatrix}1&1&-3&27 0&0&4&-24 0&-1&11&-72end{bmatrix}$$
$$left|begin{array}{l}{{54^{circ}}} {{12}} {{9 .}}end{array}right.$$
$$begin{bmatrix}2&2 4&4 3&2end{bmatrix}$$
$$begin{array}{l l l}{left[1right.}&{1}&{-3} {left[1right.}&{1}&{1} {left[3right.}&{2}&{2}end{array}right]}end{array}$$

## 
27
3 9
1 1 3 1 1 1 3 2 2

1 1 3 0 0 4 0 1 11
27 
24 72 - Question 6.

Method: Firstly, we need to check if the orthonormal *basis* of S is available, a basis could orthonormalized if it was linear *independent*. By using the rule of Sarrus or the Laplace *expansion*, we will determine whether it is linear independent through the determinant. Finally, with Gram-Schmidt orthonormalization *process* We can find a set of vectors that are perpendicular to each other in matrix S with it General Formula and construct its orthogonal *matrix* and normalize it. 

$$mathrm{Step1.}$$

Step 1. Check the determinant of matrix

$$mathrm{Checkthe}$$
$$operatorname*{det}(mathbf{B})={left|begin{array}{l l l}{4}&{8}&{8} {-8}&{8}&{-4} {8}&{4}&{-8}end{array}right|}$$
det(B) =
4 8 8 
8 8 4
8 4 8 
= (4 8 (8) + (8) 4 8 + 8 (4) 8) 
$$begin{array}{r l}{={}}&{{}(4times8times(-8)+(-8)times4times8+8times(-4)times8)} {-}&{{}} {-}&{{}(8times8times8+4times(-4)times4+(-8)times8times(-8))}end{array}$$
(8 8 8 + 4 (4) 4 +( 8) 8 (8)) 
$$mathbf{partial}=mathbf{partial}-1728$$
= 1728
 Basis S is linear independent so we can conclude it has a orthogonal basis called S
= f1
,f2
, f3 . 

$$2.$$
Step 2. Find f1
,f2
, f3 in S

# Find $mathrm{f_1,f_2,:f_3}$ in $mathrm{S^n}$. 
$$mathbf{f}_{1}=mathbf{r}mathbf{o}mathbf{w}_{1}(mathbf{S})=(4,-8,8)$$
f1 =row1(S) = 4, 8, 8
$$mathrm{f}_{2}=!mathrm{row}_{2}(mathrm{S})-{frac{mathrm{row}_{2}(mathrm{S})timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}times!mathrm{f}_{1}$$
f2 =row2(S)
row2(S) f1
f1
2 f1
$${frac{mathrm{row}_{2}(mathrm{S})timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}={frac{(8,8,4)times(4,-8,8)}{left({sqrt{4^{2}+(-8)^{2}+8^{2}}}right)^{2}}}=0$$
2 =
8, 8, 4 4,8, 8
2+(8)2+8
2
2 =0
$$Longrightarrowmathrm{f}_{2}=mathrm{row}_{2}(mathrm{S})=(8,8,4)$$
 f2 =row2(S) = 8, 8, 4
$$mathrm{f}_{3}=mathrm{row}_{2}(mathrm{S})-{frac{mathrm{row}_{3}(mathrm{S})timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}timesmathrm{f}_{1}-{frac{mathrm{row}_{3}(mathrm{S})timesmathrm{f}_{2}}{|mathrm{f}_{2}|^{2}}}timesmathrm{f}_{2}$$
2 f1
row3(S)f2
f2
2 f2
$${frac{mathrm{row}_{3}(mathrm{S})timesmathrm{f}_{1}}{|mathrm{f}_{1}|^{2}}}={frac{(8,-4,-8)times(4,-8,8)}{left({sqrt{4^{2}+(-8)^{2}+8^{2}}}right)^{2}}}=0$$
2 =
8, 4, 8 4,8, 8
2+(8)2+8
2
2 =0
$${frac{mathrm{row}_{3}(mathrm{S})timesmathrm{f}_{2}}{|mathrm{f}_{2}|^{2}}}={frac{(8,-4,-8)times(8,8,4)}{left({sqrt{8^{2}+8^{2}+4^{2}}}right)^{2}}}=0$$
2 =
8, 4, 8 8, 8, 4
2+8
2+4
2
2 =0
$Longrightarrow$ f${}_{3}$ =row${}_{3}$(S) = (8, -4, 8) $Longrightarrow$ Orthogonal basis of S =S${}^{prime}$=$begin{bmatrix}4&8&8 -8&8&-4 8&4&-8end{bmatrix}$
=
4 8 8 
8 8 4
8 4 8
Orthonormal basis of S = f1 f1 , f2 f2 , f3
f3 =
1
3
2
3
2
3 

2
3
2
3 
1
3
2
3
1
3 
2
3
$$mathrm{P}_{varepsilontotheta}=[varepsilonquad|quadtheta]=begin{bmatrix}1&0&0 0&1&0 0&0&1end{bmatrix}begin{bmatrix}1&0&1 1&1&0 0&1&1end{bmatrix}$$
1 0 0
0 1 0
0 0 1
1 0 1
1 1 0
0 1 1
Step 3. Normalized orthogonal basis, we divide each vector in orthogonal basis with their norm. 

Method 1: By using Gauss-Jordan *elimination* method we construct an *augmented* matrices by placing the vectors of the target basis ε on the left and the vectors of the source basis θ on the right, forming ε | θ . Then, by applying elementary row *operations*, we transform the left side into the identity *matrix* I. When the left side becomes I, the resulting matrix on the right side will be the transition matrix from ε to θ and conversely or ε | θ will converted to I | Pεθ and reversely. 

a)
Step 1. Transit basis from ε to θ

$$begin{array}{c|c}{{0}}&{{1}} {{0}}&{{1}} {{1}}&{{0}}end{array}$$
$$begin{bmatrix}1&1 0&1 0&1end{bmatrix}$$
1 0 0
0 1 0
0 0 1
1 0 1
1 1 0
0 1 1
$${begin{array}{l l}{0}&{1} {1}&{0} {1}&{1}end{array}}left(mathrm{row}_{2}-mathrm{row}_{1}right)left[begin{array}{l}{1} {-1} {0}end{array}right]$$

(row2 row1) 1 0 0 
1 1 0
0 0 1
1 0 1
0 1 1
0 1 1
$$left[{begin{array}{l}{1} {-1} {0}end{array}}right.$$
1 0 0 
1 1 0
0 0 1
1 0 1
0 1 1
0 1 1
$$begin{array}{r l}{0}&{1} {1}&{-1} {1}&{1}end{array}left(begin{array}{l}{mathrm{row_{3}-row_{2}}} {longrightarrow} {end{array}right)$$
(row3 row2) 1 0 0 
$$begin{array}{c c c c}{{0}}&{{0}}&{{1}}&{{0}}&{{1}} {{1}}&{{0}}&{{0}}&{{1}}&{{-1}} {{0}}&{{1}}&{{0}}&{{1}}&{{1}}end{array}$$
$$begin{array}{l}{{left[begin{array}{l}{1} {-1} {1}end{array}right.}}end{array}$$
$$begin{array}{c}{{0}} {{1}} {{-1}}end{array}$$
$$left|begin{array}{l}{{1}} {{0}} {{0}}end{array}right.$$
1 1 0
1 1 1
1 0 1
0 1 1
0 0 2
$${begin{array}{r}{1} {-1} {2}end{array}}left[begin{array}{l}{{stackrel{mathrm{row3}}{2}}} {{longrightarrow}} {{begin{array}{l}{-1} {frac{1}{2}}end{array}}}end{array}right]$$

$$begin{array}{r l}{0}&{0} {1}&{0} {-1}&{1}end{array}$$
$$begin{array}{r r}{{1}}&{{0}} {{0}}&{{1}} {{0}}&{{0}}end{array}$$
$$left[{begin{array}{l}{1} {-1} {1}end{array}}right.$$
1 0 0 
1 1 0
1 1 1
1 0 1
0 1 1
0 0 2

row3
2
$$begin{array}{c}{{0}} {{1}} {{-frac{1}{2}}}end{array}$$
1 0 1 
1 1 0
1
2 
1
2
1
2
$$begin{array}{c c}{{0}}&{{1}} {{1}}&{{-1}} {{0}}&{{2}}end{array}$$
$$begin{array}{c|c}{{1}}&{{1}} {{0}}&{{0}} {{frac{1}{2}}}&{{0}}end{array}$$
$$begin{array}{c}{{0}} {{1}} {{0}}end{array}$$
$$begin{array}{c}{{1}} {{-1}} {{1}}end{array}$$
1 0 1
0 1 1
0 0 1
1
1
2
1
2
$$left[begin{array}{l l l}{{1}}&{{}}&{{0}}&{{}} {{-1}}&{{}}&{{1}}&{{}} {{1}}&{{}}&{{-{frac{1}{2}}}}&{{{frac{1}{2}}}}end{array}right]$$
1 0 0 
1 1 0
$$begin{array}{r r}{{1}}&{{0}} {{0}}&{{1}} {{0}}&{{0}}end{array}$$
1 0 1
0 1 1
0 0 1
$${begin{array}{r}{1} {-1} {1}end{array}}{left(mathrm{row}_{2}+mathrm{row}_{3}right)}$$
(row2 + row3)
(row1 row3)
2 
$$left[begin{array}{l l l}{{frac{1}{2}}}&{}&{{frac{1}{2}}} {-{frac{1}{2}}}&{}&{{frac{1}{2}}} {{frac{1}{2}}}&{}&{{-{frac{1}{2}}}}end{array}right]$$

1 2
$$left.begin{array}{l}{{-{frac{1}{2}}}} {{{frac{1}{2}}}} {{{frac{1}{2}}}} {{{frac{1}{2}}}}end{array}right|left.begin{array}{l l l}{{1}}&{{0}}&{{0}} {{0}}&{{1}}&{{0}} {{0}}&{{0}}&{{1}}end{array}right|$$
2 

1 2
1
2 
1 0 0 0 1 0 0 0 1

$$Longrightarrowmathrm{P}_{varepsilonrightarrowtheta}={left[begin{array}{l l l}{{frac{1}{2}}}&{}&{{frac{1}{2}}} {-{frac{1}{2}}}&{}&{{frac{1}{2}}} {{frac{1}{2}}}&{}&{{-{frac{1}{2}}}}end{array}right]}$$
 Pεθ =
1
2
1
2 

1
2
1
2
1
2 
1
2

1
2 
1
2
$$begin{array}{c}{{0]}} {{0]}} {{1]}}end{array}$$
1
2
b)
Step 1. Convert matrix ε in θ | ε to identity matrix. 

$$mathrm{P}_{thetatovarepsilon}=[thetaquad|quadvarepsilon]={begin{bmatrix}1&0&1 1&1&0 0&1&1end{bmatrix}}{begin{bmatrix}1&0 0&1 0&0end{bmatrix}}$$
Pθε = θ | ε =
1 0 1
1 1 0
0 1 1
1 0 0
0 1 0
0 0 1
Step 2. Because the matrix ε already identity matrix so we can say that. 

$$mathrm{P}_{thetatovarepsilon}=theta=begin{bmatrix}1&0 1&1 0&1end{bmatrix}$$
Pθε = θ =
1 0 1
1 1 0
0 1 1
Another proof to conclude Pεθ and Pθε is the change of basis matrix from ε to θ and reversely is the inverse matrix of Pεθ is Pθε Method 2:
By transiting each vectors in matrix basis θ to basis ε. The resulting set of vectors in basis θ after transited will be the the change of basis matrix from θ to ε and reversely. 

a)
Step 1. Find vectors in θ to ε through solution of the augmented matrix constructed by θ and each vectors in ε

θX | column1(ε) =
1 0 1
1 1 0
0 1 1
1
0
0
1 0 1 1 1 0 0 1 1 1 0 0 1 0 1 0 1 1 0 1 1 1 1 0

(row2 row1) 1 0 1
0 1 1 0 1 1

(row3 row2) 1 0 1
0 1 1
0 0 2
$$begin{array}{c}{{1}} {{-1}} {{1}}end{array}left|begin{array}{c}{{1}} {{-1}} {{0}}end{array}right|$$
1
0
$${begin{array}{l}{1} {-1} {2}end{array}}{left|begin{array}{l}{1} {-1} {1}end{array}right|}$$
$$mathbf{Sigma}^{1}$$
1
1 
 $bftextit{=}begin{cases}x+z=1 y-z=-1 2z=1end{cases}Longrightarrowbegin{cases}x=frac{1}{2} y=-frac{1}{2} z=frac{1}{2}end{cases}$ = v1 $bftextit{[}theta Xquad|quad column_2(epsilon)]$= $bfbegin{bmatrix}1&0&1 1&1&0 0&1&1end{bmatrix}$ $bftextit{begin{bmatrix}1&0&1 1&1&0 0&1&1end{bmatrix}begin{bmatrix}0 1 0end{bmatrix}$ (row$bftextit{_2}-row$1) $bftextit{begin{bmatrix}1&0 0&1 0&1end{bmatrix}}$

x =
1
2
y = 
1
2
z =
1
2 
= v1
θX | column2(ε) =
1 0 1
1 1 0
0 1 1

(row2 row1) 1 0 1
$$begin{array}{c}{{0]}} {{1]}} {{0]}}end{array}$$
0
1
0
$$begin{array}{c|c}{{1}}&{{0}} {{-1}}&{{1}} {{1}}&{{0}}end{array}$$
0 1 1
0 1 1
0
1
0
$${begin{array}{r l}{0}&{1} {1}&{-1} {1}&{1}end{array}}{left|begin{array}{l}{0} {1} {0}end{array}right|} (mathrm{row}_{3}-mathrm{row}_{2}) {left[begin{array}{l}{1} {0} {0}end{array}right.}$$
$$begin{bmatrix}1 0 0end{bmatrix}$$

(row3 row2) 1 0 1
0 1 1 0 0 2
0
$$begin{array}{c|c}{{1}}&{{0}} {{-1}}&{{1}} {{2}}&{{-1}}end{array}$$
1 
$$={left{begin{array}{l l}{mathrm{x,=,0}} {mathrm{y-z=1implies{left{begin{array}{l l}{mathrm{x=frac{1}{2}}} {mathrm{y=frac{1}{2}}} {mathrm{z=-frac{1}{2}}}end{array}right.}}end{array}right.}={mathrm{v}}_{2}$$

x =
1
2
y =
1
2
z = 
1
2 
= v2
θX | column3(ε) =
1 0 1
1 1 0
0 1 1
1 0 1 1 1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 1 1 0 0 1 = x + z = 0 y z = 0 2z = 1

(row2 row1) 1 0 1

(row3 row2) 1 0 1

x = 
1
2
y =
1
2
z =
1
2 
= v3
$$begin{array}{c|c}{{1}}&{{0}} {{0}}&{{0}} {{1}}&{{1}}end{array}$$
0
0
1
$$begin{array}{r r}{0}&{1} {1}&{-1} {1}&{1}end{array}left|begin{array}{l}{0} {0} {1}end{array}right|$$ $$begin{array}{r r}{0}&{1} {1}&{-1} {0}&{2}end{array}left|begin{array}{l}{0} {0} {1}end{array}right|$$
0 1 1
0 1 1
0
0
1
0 1 1
0 0 2
0
0
1 
$${mathrm{Step2.}}qquadmathrm{P_{varepsilonto0}=left(v_{1},,v_{2},,v_{3}right)=left[begin{array}{l l l}{{frac{1}{2}}}&{{frac{1}{2}}}&{{-frac{1}{2}}} {{-frac{1}{2}}}&{{frac{1}{2}}}&{{frac{1}{2}}} {{frac{1}{2}}}&{{-frac{1}{2}}}&{{frac{1}{2}}}end{array}right]}$$
, v3 =
1
2 
1
2 

1
2
1
2
2 
1
2
1
2
1 0 0 0 1 0 0 0 1 1 1 0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 1 0 0 0 1 1 0 1

x =1
y = 1
z = 0 
= u1

x = 0
y = 1
z = 1 
= u2

x = 1
y = 0
z = 1 
= u3
$$begin{array}{r}{|{mathrm{boldmathcolumn}}_{1}(theta)]=begin{bmatrix}1&0 0&1 0&0end{bmatrix}} {mathrm{boldmathcolumn}}_{2}(theta)]=begin{bmatrix}1&0 0&1 0&0end{bmatrix}} {mathrm{boldmathboldmathcolumn}}_{3}(theta)]=begin{bmatrix}1&0 0&1 0&0end{bmatrix}}end{array}$$
$$begin{array}{r l}{[varepsilonmathbf{X}}&{{}]} { }&{} { }&{} {[varepsilonmathbf{X}}&{{}]} { }&{} { }&{} { }&{} {[varepsilonmathbf{X}}&{{}]}end{array}$$
εX | column1(θ) =
εX | column2(θ) =
εX | column3(θ) =
$${mathrm{Step2.}}qquadquadmathrm{P}_{thetatovarepsilon}=left(mathrm{u}_{1}, mathrm{u}_{2}, mathrm{u}_{3}right)=begin{bmatrix}1 1 0end{bmatrix}$$
, u3 =
$begin{array}{c c}&1 cdot&0 cdot&1end{array}$ . 
1 0 1
1 1 0
0 1 1
b)
Step 1. Find vectors in ε to θ through solution of the augmented matrix constructed by ε and each vectors in θ. 

## Chapter 2. Result 2.1 Question 1.

# Given the matrix $mathbf{A}=begin{bmatrix}1 2 1end{bmatrix}$ . 
Given the matrix A =

# . Find all values of $a,$ for which? 
$$bigstarbigstarbigstarbigstar$$
1 2 1
2 2 1
1 2 a
. Find all values of a for which
$$operatorname*{det}(mathrm{A}){=}0.$$
det( A)=0. 
$${mathcal{I}}=-1$$
a = 1

## 2.2 Question 2.

Solve the following system of linear equations by using Gaussian Elimination method. 

a)

$$left{begin{array}{l l}{x+3-z=3} {x-2y+2z=4} {2x+y+z=7}end{array}right.$$

b)

$$left{begin{array}{l l}{X+5y-2z=4} {3x-mathrm{y}+z=3} {5x+y-2z=4}end{array}right.$$
x + 5y 2z = 4
3x y + z = 3
5x + y 2z = 4
x + 3 z = 3
x 2y + 2z = 4
2x + y + z = 7
a) The system of linear equations has only one solution x = 1, y = 1, z = 1

$$begin{array}{r l}{{mathrm{b),,The}}}&{{}{mathrm{system}}}&{{mathrm{of}}}&{{mathrm{linear}}} {mathrm{x={frac{-18-4T}{5}},mathrm{y={frac{-1+3T}{5}},z=T}}}end{array}$$
, y =
1+3T
5
, z = T
b) The system of linear equations has infinite solution

## 2.3 Question 3.

Let v1= 1;1;1 , v2= 2;5;1 , v3= 3;0;5 . Show that the set B = v1
, v2 v3 is a basis of R
3
.

Matrix B is linear independent rank(B) = dim(R
3)=3 B is a basis of R
3

## 2.4 Question 4.

# Find a matrix P that diagonalize $A=frac{1}{2}$. 
Find a matrix P that diagonalize A =
$$begin{bmatrix}1&2&2 2&1&1 0&0&1end{bmatrix}$$
1 2 2
2 1 1
0 0 1
$$mathbf{P}={left[begin{array}{l}{-1} {1} {0}end{array}right]}$$
$$begin{array}{r l}{1}&{{}}&{1} {1}&{{}}&{2} {0}&{{}}&{-2}end{array}$$
P =
1 1 1
1 1 2
0 0 2

## 2.5 Question 5.

Let S = v1= 2;4;3 , v2 = 2;4;2 , v3= 6;4;2 . Find the coordinate vector of v = 54, 12, 9 relative to S. 

v s =
3
$$[mathbf{v}]_{mathrm{{s}}}={left[begin{array}{l}{3} {6} {-6}end{array}right]}$$

## 6 2.6 Question 6.

Use the Gram-Schmidt orthonormalization process to transform the basis S = v1 = 4;-8;8 , v2 = 8;8;4 , v3 = 8;-4;8 for R
3 into an orthonormal basis. 

orthonormal basis of S =
1
3
2
3
2
3 

2
3
2
3 
1
3
2
3
1
3 
2
$${mathrm{orthonormalbasisofS=}}left[{frac{1}{3}}quad{frac{2}{3}}quad{frac{2}{3}}right]$$ $${mathrm{orthonormalbasisofS=}}left[{begin{array}{l l l}{{frac{1}{3}}}&{{frac{2}{3}}}&{{frac{2}{3}}} {{-{frac{2}{3}}}}&{{{frac{2}{3}}}}&{{-{frac{1}{3}}}} {{{frac{2}{3}}}}&{{{frac{1}{3}}}}&{{-{frac{2}{3}}}}end{array}}right]$$

![37_image_0.png](37_image_0.png)

Consider the vector space R3 with two bases:
ε = ε1
, ε2
, ε3 in wihich ε1 = 1, 0, 0 , ε2 = 0, 1, 0 , ε3 = 0, 0, 1 θ = θ1
, θ2
, θ3 in wihich θ1 = 1, 1, 0 , θ2 = 0, 1, 1 , θ3 = 1, 0, 1 a) Find the transition matrix from the basis ε to the basis θ. 

b) Find the transition matrix from the basis θ to the basis ε . 

$$mathbf{a})$$ $$mathbf{p}_{varepsilontotheta}=left[begin{array}{l l l}{{frac{1}{2}}}&{{}}&{{frac{1}{2}}}&{{-frac{1}{2}}} {{-frac{1}{2}}}&{{}}&{{frac{1}{2}}}&{{}} {{frac{1}{2}}}&{{-frac{1}{2}}}&{{}}&{{frac{1}{2}}}end{array}right]$$
Pεθ =
1

1
2
1
2
2 
1
2
1
2
$mathbf{b}$). 
4) $mathbf{P}_{thetatovarepsilon}mathbf{=}begin{bmatrix}1&0&1 1&1&0 0&1&1end{bmatrix}$. 

b)
Pθε =
1 0 1
1 1 0
0 1 1
2 
1
2 
# Refferences

Clay, A. (2015). Introduction to linear algebra. University of Manitoba:
https://adamjclay.github.io/linear_notes.pdf Kin, E. (2025). Introduction to linear algebra: https://elijahkin.github.io/teaching
/math240.pdf Ricardo, H. (2009). A modern introduction to linear *algebra*. Press CRC. 

Boby, M. a. K. (n.d.). The Sarrus Rule.docx. Scribd: https://fr.scribd.com/d ocument/629312168/The-Sarrus-Rule-docx Wikipedia contributors. (2025, December 2). Laplace *expansion*. Wikipedia. 

https://en.wikipedia.org/wiki/Laplace_expansion Wikipedia contributors. (2025, December 14). Cramers *rule*. Wikipedia. 

https://en.wikipedia.org/wiki/Cramer%27s_rule Khrushchev, S. (2024). Gauss-Jordan Elimination. In Classroom *companion:*
economics (pp. 183). https://doi.org/10.1007/978-3-031-68682-5_1 33