# Lecture 3 and 4: Stochastic Gradient Descent and Neural Networks + PyTorch

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

- Models that does a good job of approximating complex non-linear functions
- It is a sequence of layers, each of which is a linear transformation followed by a non-linear transformation

### Components

- **Node (or neuron)**: a single unit in a layer
- **Input layer**: the features of the data
- **Hidden layer**: the layer(s) between the input and output layers
- **Output layer**: the prediction(s) of the model
- **Weights**: the parameters of the model
- **Activation function**: the non-linear transformation (e.g. ReLU, Sigmoid, Tanh, etc.)

<img src="images/3_nn.png" width="600">

_X : (n x d), W : (h x d), b : (n x h), where h is the number of hidden nodes_
_b is actually 1 x hs, but we can think of it as n x hs because it is broadcasted_

$$\mathbf{H}^{(1)} = \phi^{(1)} (\mathbf{X}\mathbf{W}^{(1)\text{T}} + \mathbf{b}^{(1)})$$

$$\mathbf{H}^{(2)} = \phi^{(2)} (\mathbf{H}^{(1)}\mathbf{W}^{(2)\text{T}} + \mathbf{b}^{(2)})$$

$$\mathbf{Y} = (\mathbf{H}^{(2)}\mathbf{W}^{(3)\text{T}} + \mathbf{b}^{(3)})$$

- In a layer,
  $$\text{ num of weights} = \text{num of nodes in previous layer} \times \text{num of nodes in current layer}$$

$$\text{num of biases} = \text{num of nodes in current layer}$$

$$\text{num of parameters} = \text{num of weights} + \text{num of biases}$$

#### Activation Functions

<img src="images/4_act_funcs.png" width="600">

#### Finding gradient of loss in a neural network

- **Backpropagation**: a method to calculate the gradient of the loss function with respect to the weights
- **Chain rule**: a method to calculate the gradient of a function composed of multiple functions
- It is pretty complicated, but PyTorch does it for us

### Deep Learning

- Neural networks with > 1 hidden layer
  - NN with 1 hidden layer: shallow neural network

## PyTorch for Neural Networks

- PyTorch is a popular open-source machine learning library by Facebook based on Torch
- It is a Python package that provides two high-level features:
  - Tensor computation (like NumPy) with strong GPU acceleration
  - Gradient computation through automatic differentiation

### Tensors

- Similar to `ndarray` in NumPy

```python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5]) # int
x = torch.tensor([1, 2, 3, 4, 5.]) # float
x = torch.tensor([[1, 2], [3, 4], [5, 6]])

y = torch.zeros(3, 2)
y = torch.ones(3, 2)
y = torch.rand(3, 2)

# Check the shape, dimensions, and data type
x.shape
x.ndim
x.dtype

# Operations
a = torch.rand(1, 3)
b = torch.rand(3, 1)

a + b # broadcasting
a * b # element-wise multiplication
a @ b # matrix multiplication
a.mean()
a.sum()

# Indexing
a[0,:] # first row
a[0] # first row
a[:,0] # first column

# Convert to NumPy
x.numpy()
```

### GPU with PyTorch

```python
# Check if GPU is available
torch.backends.mps.is_available() # mac M chips
torch.cuda.is_available() # Nvidia GPU

# To activate GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x.to('cpu') # move tensor to cpu
```

#### Gradient Computation

- use `backward()` to compute the gradient, backpropagation

```python
X = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
w = torch.tensor([1.0], requires_grad=True)  # Random initial weight
y = torch.tensor([2.0, 4.0, 6.0], requires_grad=False)  # Target values
mse = ((X * w - y)**2).mean()
mse.backward()
w.grad
```

### Linear Regression with PyTorch

- Every NN model has to inherit from `torch.nn.Module`

```python
from torch import nn

class linearRegression(nn.Module):  # inherit from nn.Module

    def __init__(self, input_size, output_size):
        super().__init__()  # call the constructor of the parent class

        self.linear = nn.Linear(input_size, output_size,)  # wX + b

    def forward(self, x):
        out = self.linear(x)
        return out

# Create a model
model = linearRegression(1, 1) # input size, output size

# View model
summary(model)

## Train the model
LEARNING_RATE = 0.02
criterion = nn.MSELoss()  # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # optimization algorithm is SGD

# DataLoader for mini-batch
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 50
dataset = TensorDataset(X_t, y_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training
def trainer(model, criterion, optimizer, dataloader, epochs=5, verbose=True):
    """Simple training wrapper for PyTorch network."""

    for epoch in range(epochs):
        losses = 0

        for X, y in dataloader:

            optimizer.zero_grad()       # Clear gradients w.r.t. parameters
            y_hat = model(X).flatten()  # Forward pass to get output
            loss = criterion(y_hat, y)  # Calculate loss
            loss.backward()             # Getting gradients w.r.t. parameters
            optimizer.step()            # Update parameters
            losses += loss.item()       # Add loss for this batch to running total

        if verbose: print(f"epoch: {epoch + 1}, loss: {losses / len(dataloader):.4f}")

trainer(model, criterion, optimizer, dataloader, epochs=30, verbose=True)
```

### Non-linear Regression with PyTorch

- use `torch.nn.Sequential` to create a model

```python
class nonlinRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.main = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),  # input -> hidden layer
            nn.Sigmoid(),                        # sigmoid activation function in hidden layer
            nn.Linear(hidden_size, output_size)  # hidden -> output layer
        )

    def forward(self, x):
        x = self.main(x)
        return x
```

### Common Criteria and Optimizers for PyTorch

| Task                       | Criterion (Loss) | Optimizer |
| -------------------------- | ---------------- | --------- |
| Regression                 | MSELoss          | SGD       |
| Binary Classification      | BCELoss          | Adam      |
| Multi-class Classification | CrossEntropyLoss | Adam      |

- Input of CrossEntropyLoss doesn't need to be normalized (i.e. no need to sum to 1/ no need to use `nn.Softmax`)

```python
# criterions
from torch import nn
reg_criterion = torch.nn.MSELoss()
bc_criterion = torch.nn.BCEWithLogitsLoss()
mse_criterion = torch.nn.CrossEntropyLoss()

# optimizers
from torch import optim
reg_optim = torch.optim.SGD(model.parameters(), lr=0.2)
class_optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```
