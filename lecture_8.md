# Lecture 8: Advanced CNN

## Advanced CNN

### Generative vs Discriminative Models

| Generative Models                                                         | Discriminative Models                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Directly model the joint probability distribution of the input and output | Model the conditional probability of the output given the input |
| Directly model $P(y\|x)$                                                  | Estimate $P(x\|y)$ to then deduce $P(y\|x)$                     |
| Build model for each class                                                | Make boundary between classes                                   |
| "Generate or draw a cat"                                                  | "Distinquish between cats and dogs"                             |
| Examples: Naibe bayes, ChatGPT                                            | Examples: Logistic Regression, SVM, Tree based models, CNN      |

### Autoencoders

<img src="images/8_autoencoder.png" width="300">

- Designed to reconstruct the input
- Encoder and a decoder
- Why do we need autoencoders?
  - Dimensionality reduction
  - Denoising

#### Dimensionality Reduction

- Maybe the z axis is unimportant in the input space for classification

```python
from torch import nn

class autoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```python
# Set up the training
BATCH_SIZE = 100
torch.manual_seed(1)
X_tensor = torch.tensor(X, dtype=torch.float32)
dataloader = DataLoader(X_tensor,
                        batch_size=BATCH_SIZE)
model = autoencoder(3, 2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
EPOCHS = 5

for epoch in range(EPOCHS):
    for batch in dataloader:
        optimizer.zero_grad()           # Clear gradients w.r.t. parameters
        y_hat = model(batch)            # Forward pass to get output
        loss = criterion(y_hat, batch)  # Calculate loss
        loss.backward()                 # Getting gradients w.r.t. parameters
        optimizer.step()                # Update parameters

# Use encoder
model.eval()
X_encoded = model.encoder(X_tensor)
```

#### Denoising

- Remove noise from the input
- Use _Transposed Convolution Layers_ to upsample the input
  - Normal convolution: downsample (output is smaller than input)
  - Transposed convolution: upsample (output is larger than input)

```python
def conv_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)  # reduce x-y dims by two; window and stride of 2
    )

def deconv_block(input_channels, output_channels, kernel_size):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=2),
        nn.ReLU()
    )

class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 16),
            conv_block(16, 8)
        )
        self.decoder = nn.Sequential(
            deconv_block(8, 8, 3),
            deconv_block(8, 16, 2),
            deconv_block(16, 32, 2),
            nn.Conv2d(32, 1, 3, padding=1)  # final conv layer to decrease channel back to 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)  # get pixels between 0 and 1
        return x
```

````python
# Set up the training
EPOCHS = 20
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
img_list = []

for epoch in range(EPOCHS):
    losses = 0
    for batch, _ in trainloader:
        noisy_batch = batch + noise * torch.randn(*batch.shape)
        noisy_batch = torch.clip(noisy_batch, 0.0, 1.0)
        optimizer.zero_grad()
        y_hat = model(noisy_batch)
        loss = criterion(y_hat, batch)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(f"epoch: {epoch + 1}, loss: {losses / len(trainloader):.4f}")
    # Save example results each epoch so we can see what's going on
    with torch.no_grad():
        noisy_8 = noisy_batch[:1, :1, :, :]
        model_8 = model(input_8)
        real_8 = batch[:1, :1, :, :]
    img_list.append(utils.make_grid([noisy_8[0], model_8[0], real_8[0]], padding=1))```
````

## Generative Adversarial Networks (GANs)

- Model used to generate new data (indistinguishable from real data)
- No need for labels (unsupervised learning)
- See [here](https://developers.google.com/machine-learning/gan/gan_structure)

<img src="images/8_gan_struct.png" width="500">

- Two networks:
  - Generator: creates new data
  - Discriminator: tries to distinguish between real and fake data
- Both are battling each other:
  - Generator tries to create data that the discriminator can't distinguish from real data
  - Discriminator tries to distinguish between real and fake data

### Training GANs

1. Train the discriminator (simple binary classification)
   - Train the discriminator on real data
   - Train the discriminator on fake data (generated by the generator)
2. Train the generator
   - Generate fake images with the generator and label them as real
   - Pass to discriminator and ask it to classify them (real or fake)
   - Pass judgement to a loss function (see how far it is from the ideal output)
     - ideal output: all fake images are classified as real
   - Do backpropagation and update the generator
3. Repeat

### Pytorch Implementation

1. Creating the data loader

   ```python
   DATA_DIR = "../input/face-recognition-dataset/Extracted Faces"

   BATCH_SIZE = 64
   IMAGE_SIZE = (128, 128)

   data_transforms = transforms.Compose([
       transforms.Resize(IMAGE_SIZE), # uses CPU (bottleneck)
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
   data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
   ```

2. Creating the generator

   ```python

   class Generator(nn.Module):

    def __init__(self, LATENT_SIZE):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),

            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
   ```

3. Creating the discriminator

   ```python

   class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
   ```

4. Instantiating the models

   ```python
   device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

   LATENT_SIZE = 100
   generator = Generator(LATENT_SIZE).to(device)
   discriminator = Discriminator().to(device)

   criterion = nn.BCELoss()

   optimizerG = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
   optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    generator.apply(weights_init)
    discriminator.apply(weights_init);
   ```

5. Training the GAN

   ```python
    img_list = []
   fixed_noise = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1).to(device)

    NUM_EPOCHS = 50
   from statistics import mean
   print('Training started:\n')

    D_real_epoch, D_fake_epoch, loss_dis_epoch, loss_gen_epoch = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        D_real_iter, D_fake_iter, loss_dis_iter, loss_gen_iter = [], [], [], []

        for real_batch, _ in data_loader:

            # STEP 1: train discriminator
            # ==================================
            optimizerD.zero_grad()

            real_batch = real_batch.to(device)
            real_labels = torch.ones((real_batch.shape[0],), dtype=torch.float).to(device)

            output = discriminator(real_batch).view(-1)
            loss_real = criterion(output, real_labels)

            # Iteration book-keeping
            D_real_iter.append(output.mean().item())

            # Train with fake data
            noise = torch.randn(real_batch.shape[0], LATENT_SIZE, 1, 1).to(device)

            fake_batch = generator(noise)
            fake_labels = torch.zeros_like(real_labels)

            output = discriminator(fake_batch.detach()).view(-1)
            loss_fake = criterion(output, fake_labels)

            # Update discriminator weights
            loss_dis = loss_real + loss_fake
            loss_dis.backward()
            optimizerD.step()

            # Iteration book-keeping
            loss_dis_iter.append(loss_dis.mean().item())
            D_fake_iter.append(output.mean().item())

            # STEP 2: train generator
            # ==================================
            optimizerG.zero_grad()

            # Calculate the output with the updated weights of the discriminator
            output = discriminator(fake_batch).view(-1)
            loss_gen = criterion(output, real_labels)
            loss_gen.backward()

            # Book-keeping
            loss_gen_iter.append(loss_gen.mean().item())

            # Update generator weights and store loss
            optimizerG.step()

        print(f"Epoch ({epoch + 1}/{NUM_EPOCHS})\t",
            f"Loss_G: {mean(loss_gen_iter):.4f}",
            f"Loss_D: {mean(loss_dis_iter):.4f}\t",
            f"D_real: {mean(D_real_iter):.4f}",
            f"D_fake: {mean(D_fake_iter):.4f}")

        # Epoch book-keeping
        loss_gen_epoch.append(mean(loss_gen_iter))
        loss_dis_epoch.append(mean(loss_dis_iter))
        D_real_epoch.append(mean(D_real_iter))
        D_fake_epoch.append(mean(D_fake_iter))

        # Keeping track of the evolution of a fixed noise latent vector
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            #img_list.append(utils.make_grid(fake_images, normalize=True, nrows=10))

    print("\nTraining ended.")
   ```

6. Visualize training process
   ```python
    plt.plot(np.array(loss_gen_epoch), label='loss_gen')
    plt.plot(np.array(loss_dis_epoch), label='loss_dis')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend();
   ```
   ```python
    plt.plot(np.array(D_real_epoch), label='D_real')
    plt.plot(np.array(D_fake_epoch), label='D_fake')
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.legend();
   ```

### Multi-Input Networks

```python
class multiModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, image, data):
        x_cnn = self.cnn(image) # 1st model: CNN
        x_fc = self.fc(data) # 2nd model: Fully connected
        return torch.cat((x_cnn, x_fc), dim=1) # concatenate the two outputs
```
