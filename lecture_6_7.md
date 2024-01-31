# Lecture 6: CNN

## Convolutional Neural Networks (CNN)

<img src="images/6_cnn_ex.png" width="500">

- Drastically reduces the number of params (compared to NN):
  - have activations depend on small number of inputs
  - same parameters (convolutional filter) are used for different parts of the image

### Convolution

- Idea: use a small filter/kernel to extract features from the image
  - Filter: a small matrix of weights

<img src="images/6_conv.gif" width="250">

- Notice that the filter results in a smaller output image
  - This is because we are not padding the image
  - We can add padding to the image to keep the same size
    - Padding: add zeros around the image
  - Can also add stride to move the filter more than 1 pixel at a time

### CNN Structure

<img src="images/6_cnn_struct.png" width="500">

_[img src](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)_

### CNN in PyTorch

#### 1. Convolutional Layer

```python
conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3,3))
```

- Arguments:
  - `in_channels`: number of input channels (gray scale image has 1 channel, RGB has 3)
  - `out_channels`: number of output channels (similar to hidden nodes in NN)
  - `kernel_size`: size of the filter
  - stride: how many pixels to move the filter each time
  - padding: how many pixels to add around the image

<img src="images/6_conv_layer.png" width="350">

- Size of input image (e.g. 256x256) doesn't matter, what matters is: `in_channels`, `out_channels`, `kernel_size`

$$\text{total params} = (\text{out channels} \times \text{in channels} \times \text{kernel size}^2) + \text{out channels}$$

$$\text{output size} = \frac{\text{input size} - \text{kernel size} + 2 \times \text{padding}}{\text{stride}} + 1$$

##### Dimensions of images and kernel tensors in PyTorch

\<Insert diagram here\>

- Images: `[batch_size, channels, height, width]`
- Kernel: `[out_channels, in_channels, kernel_height, kernel_width]`

Note: before passing the image to the convolutional layer, we need to reshape it to the correct dimensions. Also if you want to `plt.imshow()` the image, you need to reshape it back to `[height, width, channels]`.

#### 2. Flattening

- See the diagram above, its to go from `feature learning` -> `classification`
- At the end need to either do regression or classification

#### 3. Pooling

- IDea: reduce the size of the image
  - less params
  - less overfitting
- Common types:
  - **Max pooling**: take the max value in each region
    - Works well since it takes the sharpest features
  - **Average pooling**: take the average value in each region

#### Putting it all together

```python
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=1,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(in_channels=3,
                out_channels=2,
                kernel_size=(3, 3),
                padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Flatten(),
            torch.nn.Linear(1250, 1)
        )

    def forward(self, x):
        out = self.main(x)
        return out
```

```python
# Trainer code
def trainer(
    model, criterion, optimizer, trainloader, validloader, epochs=5, verbose=True
):
    train_loss, train_accuracy, valid_loss, valid_accuracy = [], [], [], []
    for epoch in range(epochs):  # for each epoch
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0

        # Training
        for X, y in trainloader:
            if device.type in ['cuda', 'mps']:
                    X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # Zero all the gradients w.r.t. parameters
            y_hat = model(X)  # Forward pass to get output
            idx = torch.softmax(y_hat, dim=1).argmax(dim=1) # Multiclass classification
            loss = criterion(y_hat, y)
            loss.backward()  # Calculate gradients w.r.t. parameters
            optimizer.step()  # Update parameters
            train_batch_loss += loss.item()  # Add loss for this batch to running total
            train_batch_acc += (
                    (idx.squeeze() == y).type(
                        torch.float32).mean().item()
                )
        train_loss.append(train_batch_loss / len(trainloader))
        train_accuracy.append(train_batch_acc / len(trainloader))

        # Validation
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for X, y in validloader:
                if device.type in ['cuda', 'mps']:
                    X, y = X.to(device), y.to(device)
                y_hat = model(X)
                idx = torch.softmax(y_hat, dim=1).argmax(dim=1)
                loss = criterion(y_hat, y)
                valid_batch_loss += loss.item()
                valid_batch_acc += (
                    (idx.squeeze() == y).type(
                        torch.float32).mean().item()
                )
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(valid_batch_acc / len(validloader))  # accuracy

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1}:",
                f"Train Loss: {train_loss[-1]:.3f}.",
                f"Valid Loss: {valid_loss[-1]:.3f}.",
                f"Train Accuracy: {train_accuracy[-1]:.2f}.",
                f"Valid Accuracy: {valid_accuracy[-1]:.2f}.",
            )

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }
    return results
```

### Preparing Data

#### Turning images to tensors

- Normally there are 2 steps:
  1. create a `dataset` object: the raw data
  2. create a `dataloader` object: batches the data, shuffles, etc.
- Use `torchvision` to load the data
  - `torchvision.datasets.ImageFolder`: loads images from folders
  - Assumes structure: `root/class_1/xxx.png`, `root/class_2/xxx.png`, ...

```python
import torch
from torchvision import datasets, transforms

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# create transform object
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# create dataset object
train_dataset = datasets.ImageFolder(root='path/to/data', transform=data_transforms)

# check out the data
train_dataset.classes # list of classes
train_dataset.targets # list of labels
train_dataset.samples # list of (path, label) tuples

# create dataloader object
train_loader = torch.utils.data.DataLoader(
    train_dataset,          # our raw data
    batch_size=BATCH_SIZE,  # the size of batches we want the dataloader to return
    shuffle=True,           # shuffle our data before batching
    drop_last=False         # don't drop the last batch even if it's smaller than batch_size
)

# get a batch of data
images, labels = next(iter(train_loader))
```

#### Saving and loading PyTorch models

- [PyTorch documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- Convention: `.pt` or `.pth` file extension

```python
PATH = "models/eva_cnn.pt"

# load model
model = bitmoji_CNN() # must have defined the model class
model.load_state_dict(torch.load(PATH))
model.eval() # set model to evaluation mode (not training mode)

# save model
torch.save(model.state_dict(), PATH)
```

#### Data augmentation

- To make CNN more robust to different images + increase the size of the dataset
- Common augmentations:

  - Crop
  - Rotate
  - Flip
  - Color jitter

```python
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomVerticalFlip(p=0.5), # p=0.5 means 50% chance of applying this augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
```

### Hyperparameter Tuning

- NN has a lot of hyperparameters

  - Grid search will take a long time
  - Need a smarter approach: **Optimization Algorithms**

- Examples: Ax (we will use this), Raytune, Neptune, skorch.

### Transfer Learning

- Idea: use a pre-trained model and fine-tune it to our specific task
- Install from `torchvision.models`
  - All models have been trained on ImageNet dataset (224x224 images)
- See [here for code](https://pages.github.ubc.ca/MDS-2023-24/DSCI_572_sup-learn-2_students/lectures/07_cnns-pt2.html#using-pre-trained-models-out-of-the-box)

#### Approach 1: Adding layers to pre-trained model

```python
densenet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')

for param in densenet.parameters():  # Freeze parameters so we don't update them
    param.requires_grad = False
# can fine-tune to freeze only some layers

list(densenet.named_children())[-1] # check the last layer

# update the last layer
new_layers = nn.Sequential(
    nn.Linear(1024, 500),
    nn.ReLU(),
    nn.Linear(500, 1)
)
densenet.classifier = new_layers
```

Then train the model as usual.

```python
densenet.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=2e-3)
results = trainer(densenet, criterion, optimizer, train_loader, valid_loader, device, epochs=10)
```

#### Approach 2: Use Extracted Features in a New Model

- Idea:
  1. Take output from pre-trained model
  2. Feed output to a new model

```python
def get_features(model, train_loader, valid_loader):
    """
    Extract features from both training and validation datasets using the provided model.

    This function passes data through a given neural network model to extract features. It's designed
    to work with datasets loaded using PyTorch's DataLoader. The function operates under the assumption
    that gradients are not required, optimizing memory and computation for inference tasks.
    """

    # Disable gradient computation for efficiency during inference
    with torch.no_grad():
        # Initialize empty tensors for training features and labels
        Z_train = torch.empty((0, 1024))  # Assuming each feature vector has 1024 elements
        y_train = torch.empty((0))

        # Initialize empty tensors for validation features and labels
        Z_valid = torch.empty((0, 1024))
        y_valid = torch.empty((0))

        # Process training data
        for X, y in train_loader:
            # Extract features and concatenate them to the corresponding tensors
            Z_train = torch.cat((Z_train, model(X)), dim=0)
            y_train = torch.cat((y_train, y))

        # Process validation data
        for X, y in valid_loader:
            # Extract features and concatenate them to the corresponding tensors
            Z_valid = torch.cat((Z_valid, model(X)), dim=0)
            y_valid = torch.cat((y_valid, y))

    # Return the feature and label tensors
    return Z_train, y_train, Z_valid, y_valid
```

Now we can use the extracted features to train a new model.

```python
# Extract features from the pre-trained model
densenet = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
densenet.classifier = nn.Identity()  # remove that last "classification" layer
Z_train, y_train, Z_valid, y_valid = get_features(densenet, train_loader, valid_loader)

# Train a new model using the extracted features
# Let's scale our data
scaler = StandardScaler()
Z_train = scaler.fit_transform(Z_train)
Z_valid = scaler.transform(Z_valid)

# Fit a model
model = LogisticRegression(max_iter=1000)
model.fit(Z_train, y_train)
```
