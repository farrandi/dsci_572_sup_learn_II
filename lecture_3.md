# Lecture 3: Stochastic Gradient Descent and Neural Networks

## Stochastic Gradient Descent

- Instead of updating our parameters based on a gradient calculated using all training data, we simply use **one of our data points** (the $i$-th one)

**Gradient Descent**

Loss function:

$$\text{MSE} = \mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum^{n}_{i=1} (\mathbf{x}_i \mathbf{w} - y_i)^2$$

Update procedure:

$$\mathbf{w}^{j+1} = \mathbf{w}^{j} - \alpha \nabla_\mathbf{w} \mathcal{L}(\mathbf{w}^{j})$$

**Stochastic Gradient Descent**

Loss function:

$$\text{MSE}_i = \mathcal{L}_i(\mathbf{w}) = (\mathbf{x}_i \mathbf{w} - y_i)^2$$

Update procedure:
$$\mathbf{w}^{j+1} = \mathbf{w}^{j} - \alpha \nabla_\mathbf{w} \mathcal{L}_i(\mathbf{w}^{j})$$

### Mini-batch Gradient Descent

| Gradient Descent    | Stochastic Gradient Descent |
| ------------------- | --------------------------- |
| Use all data points | Use one data point          |
| Slow                | Fast                        |
| Accurate            | Less Accurate               |

- **Mini-batch Gradient Descent** is a (in-between) compromise between the two
- Instead of using a single data point, we use a small batch of data points d

#### Mini-batch Creation

1. Shuffle and divide all data into $k$ batches, every example is used once
   - **Default in PyTorch**
   - An example will only show up in one batch
2. Choose some examples for each batch **without replacement**
   - An example may show up in multiple batches
   - The same example cannot show up in the same batch more than once
3. Choose some examples for each batch **with replacement**
   - An example may show up in multiple batches
   - The same example may show up in the same batch more than once

### Terminology

Assume we have a dataset of $n$ observations (also known as _rows, samples, examples, data points, or points_)

- **Iteration**: each time you update model weights

- **Batch**: a subset of data used in an iteration

- **Epoch**: One full pass through the dataset to look at all $n$ observations

In other words,

- In **GD**, each iteration involves computing the gradient over all examples, so

$$1 \: \text{iteration} = 1 \: \text{epoch}$$

- In **SGD**, each iteration involves one data point, so

$$n \text{ iterations} = 1 \: \text{epoch}$$

- In **MGD**, each iteration involves a batch of data, so

$$
\begin{align}
\frac{n}{\text{batch size}} \text{iterations} &= 1 \text{ epoch}\\
\end{align}
$$

**\*Note**: nobody really says "minibatch SGD", we just say SGD: in SGD you can specify a batch size of anything between 1 and $n$

## Neural Networks

-

<img src="images/3_nn.png" width="500">

- **Input layer**: the features of the data
- **Hidden layer**: the layer(s) between the input and output layers
- **Output layer**: the prediction(s) of the model
