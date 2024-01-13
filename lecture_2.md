# Lecture 2: Optimization and Gradient Descent

## Optimization

- In ML, we want to minimize a loss function
  - typically a sum of losses over the training set
- Can think of ML as a 3 step process:
  1. Choose **model**: controls space of possible functions that map X to y
  2. Choose **loss function**: measures how well the model fits the data
  3. Choose **optimization** algorithm: finds the best model

### Optimization Terminology

- **Optimization**: process to min/max a function
- **Objective Function**: function to be optimized
- **Domain**: set to search for optimal value
- **Minimizer**: value that minimizes the objective function

### Loss Function

Common loss function is MSE (mean squared error):

$$L(w) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2$$

Using a simple linear regression model $y = w_0 + w_1x$, we can rewrite the loss function as:

$$L(w) = \frac{1}{n} \sum_{i=1}^n ((w_0 + w_1x_i) - y_i)^2$$

So optimization is finding the values of $w_0$ and $w_1$ that minimize the loss function, $L(w)$.

### Notation

$$
\mathbf{y}=
\left[
\begin{array}{c} y_1 \\
\vdots \\
y_i \\
\vdots\\
y_n
\end{array}
\right]_{n \times 1}, \quad
\mathbf{X}=
\left[
\begin{array}{c} \mathbf{x}_1 \\
\vdots \\
\mathbf{x}_i \\
\vdots\\
\mathbf{x}_n
\end{array}
\right]_{n \times d}
= \left[\begin{array}{cccc}
x_{11} & x_{12} & \cdots & x_{1 d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{i 1} & x_{i 2} & \cdots & x_{i d}\\
\vdots & \vdots & \ddots & \vdots \\
x_{n 1} & x_{n 2} & \cdots & x_{n d}
\end{array}\right]_{n \times d},
\quad
\mathbf{w}=
\left[
\begin{array}{c} w_1 \\
\vdots\\
w_d
\end{array}
\right]_{d \times 1}
$$

- $n$: number of examples
- $d$: number of input features/dimensions

The goal is to find the weights $\mathbf{w}$ that minimize the loss function.

## Gradient Descent

- One of the most important optimization algorithms in ML
- Iterative optimization algorithm
- Cost: $O(ndt)$ for t iterations, better than brute force search $O(nd^2 + d^3)$

$$w_{t+1} = w_t - \alpha \nabla= L(w_t)$$

- $w_t$: current value of the weights
- $\alpha$: learning rate
- $\nabla L(w_t)$: gradient of the loss function at $w_t$

### GD with a Single Parameter

### GD with Multiple Parameters

### Other Optimization Algorithms

$$
$$
