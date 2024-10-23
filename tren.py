import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F  

# Transform image to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download dataset
full_trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

num_train = 50000
indices = list(range(num_train)) 
trainset = torch.utils.data.Subset(full_trainset, indices)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=False)

# Define a Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*3*3, 128)  
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)  
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []
val_accuracies = []

# Training the network
epochs = 10  
time0 = time()

for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
    
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(trainloader))
    
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in valloader:
            output = model(images)
            val_loss += criterion(output, labels).item()
            
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
    
    val_losses.append(val_loss / len(valloader))
    val_accuracies.append(correct / len(valset))
    
    print(f"Epoch {e+1}/{epochs} - "
          f"Training loss: {train_losses[-1]} - "
          f"Validation loss: {val_losses[-1]} - "
          f"Validation Accuracy: {val_accuracies[-1]*100:.2f}%")
    
print(f"\nTraining Time (in minutes) = {(time()-time0)/60}")

torch.save(model.state_dict(), 'cnn_model.pth')

# Plot training and validation losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
