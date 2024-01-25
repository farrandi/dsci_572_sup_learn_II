# Lecture 5: Training Neural Networks

## Backpropagation

### Basic concept

- It is to calculate the gradient of the loss function with respect to the weights
- It is a special case of the chain rule of calculus

- **Process:**

  1. Do "forward pass" to calculate the output of the network (prediction and loss)

  <img src="images/5_back1.png" width="600">

  2. Do "backward pass" to calculate the gradients of the loss function with respect to the weights

  <img src="images/5_back2.png" width="450">

### Torch: Autograd

- `torch.autograd` is PyTorch's automatic differentiation engine that powers neural network training

```python
import torch

# Create model
class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.layer1 = torch.nn.Linear(1, 6)
        self.dropout = torch.nn.Dropout(0.2) # dropout layer
        ...

    def forward(self, x):
        x = self.layer1(x)
        ...
        return x

model = network()
criterion = torch.nn.MSELoss()

# Forward pass
loss = criterion(model(x), y)
# Backward pass
loss.backward()

# Access gradients
print(model.layer1.weight.grad) # or model.layer1.weight.bias.grad

# Update weights
model.state_dict() # get the current weights
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.step() # update weights
```

### Vanishing and Exploding Gradients

- Backpropagation can suffer from two problems because of multiple chain rule applications:
  - **Vanishing gradients**: the gradients of the loss function with respect to the weights become very small
    - 0 gradients because of underflow
  - **Exploding gradients**: the gradients of the loss function with respect to the weights become very large
- Possible solutions:
  - Use **ReLU** activation function: but it can also suffer from the dying ReLU problem (gradients are 0)
  - **Weight initialization**: initialize the weights with small values
  - **Batch normalization**: normalize the input layer by adjusting and scaling the activations
  - **Skip connections**: add connections that skip one or more layers
  - **Gradient clipping**: clip the gradients during backpropagation

### Training Neural Networks in PyTorch

#### Preventing Overfitting

- Add validation loss to the training loop
- Early stopping: if we see the validation loss is increasing, we stop training
  - Define a patience parameter: if the validation loss increases for `patience` epochs, we stop training
- Regularization: add a penalty term to the loss function to prevent overfitting
  - See [573 notes](https://mds.farrandi.com/block_3/573_model_sel/573_model_sel#regularization) for more details
  - `weight_decay` parameter in the optimizer
- Dropout: randomly set some neurons to 0 during training
  - It prevents overfitting by reducing the complexity of the model
  - `torch.nn.Dropout(0.2)`

### PyTorch Trainer Code

```python
import torch
import torch.nn as nn

def trainer(model, criterion, optimizer, trainloader, validloader, epochs=5, patience=5):
    """Simple training wrapper for PyTorch network."""

    train_loss = []
    valid_loss = []

    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        valid_batch_loss = 0

        # Training
        for X, y in trainloader:

            optimizer.zero_grad()       # Zero all the gradients w.r.t. parameters

            y_hat = model(X).flatten()  # Forward pass to get output
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()             # Calculate gradients w.r.t. parameters
            optimizer.step()            # Update parameters

            train_batch_loss += loss.item()  # Add loss for this batch to running total

        train_loss.append(train_batch_loss / len(trainloader))

        # Validation
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time

            for X_valid, y_valid in validloader:

                y_hat = model(X_valid).flatten()  # Forward pass to get output
                loss = criterion(y_hat, y_valid)  # Calculate loss based on output

                valid_batch_loss += loss.item()

        valid_loss.append(valid_batch_loss / len(validloader))

        # Early stopping
        if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
            consec_increases += 1
        else:
            consec_increases = 0
        if consec_increases == patience:
            print(f"Stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!")
            break

    return train_loss, valid_loss
```

- Using the `trainer` function:

```python
import torch
import torch.nn
import torch.optim

torch.manual_seed(1)

model = network(1, 6, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05) # weight_decay=0.01 for L2 regularization
train_loss, valid_loss = trainer(model, criterion, optimizer, trainloader, validloader, epochs=201, patience=3)

plot_loss(train_loss, valid_loss)
```

### Universal Approximation Theorem

- Any continuous function can be approximated arbitrarily well by a neural network with a single hidden layer
  - In other words, NN are universal function approximators
