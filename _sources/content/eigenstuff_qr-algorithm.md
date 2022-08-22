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

# The QR algorithm for finding eigenvalues and eigenvectors

In the previous sections, we discussed finding the eigenvalues and eigenvectors of a matrix $\boldsymbol{A}$ largely abstractly, without much interest in how we would actually do this in practice. As we saw, we can find the eigenvalues (in theory) by finding the zeros of the degree-$n$ polynomial $p(\lambda) = \det(\boldsymbol{A} - \lambda \boldsymbol{I})$. If we had these eigenvalues, say $\lambda_1,\dots, \lambda_n$, then we could find the eigenvectors fairly easily by solving the linear system of equations

$$
(\boldsymbol{A} - \lambda_i \boldsymbol{I})\boldsymbol{v} = 0,
$$

e.g. by using the QR decomposition and backsubstitution. The latter component would be a feasible way to find the eigenvectors in practice if we knew what the eigenvalues were. Unfortunately, finding the zeros of $p(\lambda)$ this is not a particularly practical approach, beyond the 2- or 3-dimensional case. Instead, we require other algorithms to find the eigenvalues. We saw one method on the homework for doing this called the _power method_. Here we briefly introduce another popular algorithm which uses the QR decomposition called the QR algorithm, which we outline below.

$$
\begin{align}
&\underline{\textbf{QR algorithm}: \text{find the eigenvalues of an $n\times n$ matrix $\boldsymbol{A}$}} \\
&\textbf{input}:\text{$n\times n$ matrix }\boldsymbol{A}\in \mathbb{R}^{n\times n} \\
&\hspace{0mm} \text{while $\boldsymbol{A}$ is not approximately upper triangular:}\\
&\hspace{10mm} \boldsymbol{Q}, \boldsymbol{R} = \texttt{qr_decomposition}(\boldsymbol{A})\\
&\hspace{10mm} \text{update }\boldsymbol{A} = \boldsymbol{R}\boldsymbol{Q}\\
&\hspace{0mm} \text{return } \text{diag}(\boldsymbol{A})\\
\end{align}
$$

This algorithm works due to the following two properties. First, note that for a single interation we have

$$
\boldsymbol{A}' = \boldsymbol{RQ} = \boldsymbol{Q^\top Q R Q} = \boldsymbol{Q}^\top \boldsymbol{AQ}
$$

where $\boldsymbol{Q}$ is an orthogonal matrix. Because the matrices $\boldsymbol{A}$ and $\boldsymbol{A}'$ differ only by an orthogonal transformation on either side, they are what we call _similar_ matrices. It turns out that similar matrices always have the same eigenvalues. To see this, let $(\lambda, \boldsymbol{v})$ be an eigenvalue/eigenvector pair for $\boldsymbol{A}'$, and let $\boldsymbol{A} = \boldsymbol{Q^\top\boldsymbol{A}'\boldsymbol{Q}}$ be defined as above. Then

$$
\lambda\boldsymbol{v} = \boldsymbol{A}'\boldsymbol{v} = \boldsymbol{QA Q^\top v} \iff \lambda \boldsymbol{Q^\top v} = \boldsymbol{A Q^\top v}.
$$

This means that $(\lambda, \boldsymbol{Q^\top v})$ is an eigenvalue/eigenvector pair for the matrix $\boldsymbol{A}$, and so $\boldsymbol{A}$ and $\boldsymbol{A}'$ have the same eigenvalues, and eigenvectors which differ by a factor of $\boldsymbol{Q}^\top$. Thus at each iteration in the QR algorithm, the matrices $\boldsymbol{A}$ have the same eigenvalues.

The next step we do not prove, but will show numerically. It turns out that for "nice" matrices (in particular, matrices that have distinct eigenvalues), the QR algorithm converges to an upper triangular matrix. Therefore, as we saw in the previous section, we can read off the eigenvalues of this matrix by checking its diagonal entries. Let's see a simple example that illustrates this.

```{code-cell}
import numpy as np

A = np.random.normal(size= (3,3))
A = np.dot(A.T, A)

for i in range(10):
    Q,R = np.linalg.qr(A)
    A = np.dot(R,Q)
    print('A at iteration i = %s is' % i)
    print(A)
```

As we can see, the lower triangular portion of $\boldsymbol{A}$ is becoming closer and closer to zero after more iterations. Hence, since the eigenvalues are unchanged at each iteration, we can read of the eigenvalues of $\boldsymbol{A}$ from the eigenvalues of the (approximately) triangular matrix that we get after several iterations. Let's now implement our own `eigenvalue_decomposition_qr` function which uses the QR algorthm to find the eigenvalues of a matrix $\boldsymbol{A}$.

```{code-cell}
def eigenvalue_decomposition_qr(A):
    '''
    find the eigenvalues of a matrix using the QR decomposition
    '''
    A0 = A

    # first implement the QR algorithm
    while not np.allclose(A0, np.triu(A0)):
        Q,R = np.linalg.qr(A0)
        A0 = np.dot(R, Q)

    values = np.diag(A0)
    return values
```

Now let's test our implementation against the usual numpy `eig` function.

```{code-cell}
A = np.random.normal(size=(5,5))
A = np.dot(A.T, A)

values_qr = eigenvalue_decomposition_qr(A)
print(values_qr)

values, vectors = np.linalg.eig(A)
print(values)
```

Indeed, the two algorithms give the same output (though potentially not ordered in the same way).
