import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)  # Fixed the pretrained argument
        resnet.fc = nn.Identity()  # Remove classification layer
        self.encoder = resnet
        self.projection = ProjectionHead()

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return F.normalize(projections, dim=1)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        labels = torch.arange(batch_size).to(z_i.device)

        similarity_matrix = torch.mm(z_i, z_j.T) / self.temperature
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

        return loss

