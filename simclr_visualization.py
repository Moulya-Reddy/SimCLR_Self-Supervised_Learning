import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from simclr import SimCLR

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLR().to(device)
model.load_state_dict(torch.load("simclr_model.pth"))
model.eval()

# Load Data for Visualization
def get_cifar10_loader(batch_size=1000):
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader

train_loader = get_cifar10_loader(batch_size=1000)
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Move images to the same device as the model
images = images.to(device)

# Get embeddings
with torch.no_grad():
    embeddings = model(images).cpu().numpy()

# t-SNE Visualization
tsne = TSNE(n_components=2)
z_tsne = tsne.fit_transform(embeddings)

plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap='jet')
plt.colorbar()
plt.title("t-SNE Visualization of Image Representations")
plt.show()

