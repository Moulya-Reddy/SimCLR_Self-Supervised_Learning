import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import SimCLR, NTXentLoss
import torchvision.transforms as transforms

# Data Loader for CIFAR-10
def get_cifar10_loader(batch_size=256):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader

# Hyperparameters
epochs = 10
batch_size = 256
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader = get_cifar10_loader(batch_size=batch_size)

# Model & Optimizer
model = SimCLR().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = NTXentLoss()

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0
    print(f"Epoch {epoch+1}/{epochs} started...")

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()
        z_i = model(images)
        z_j = model(images)  # Augmented views

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{epochs}] completed. Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "simclr_model.pth")
