# SimCLR - Self-Supervised Learning on CIFAR-10  

This project implements **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)** using **PyTorch** and **ResNet-18**.  
The model is trained on the **CIFAR-10 dataset** and visualized using **t-SNE**.  

## Features:-

*Self-Supervised Learning** – No labels needed! The model learns representations from augmented images.  
*Contrastive Learning* – Uses **NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)**.  
*ResNet-18 Backbone* – Feature extraction using a deep convolutional network.  
*Projection Head* – A two-layer MLP to map features into contrastive space.  
*t-SNE Visualization* – Plots learned representations in 2D space.  

## Project Structure:-

```
/simclr-project
│── simclr.py                # Full SimCLR model implementation
│── simclr_training.py        # Training loop for SimCLR
│── simclr_visualization.py   # t-SNE visualization
│── requirements.txt          # Dependencies
```

## Installation:- 

##1.**Clone the Repository**  
```sh
git clone https://github.com/yourusername/simclr-cifar10.git
cd simclr-cifar10
```

##2.**Set Up Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

##3.**Install Dependencies**  
```sh
pip install -r requirements.txt
```

## How It Works:- 

## ** Data Augmentation (Stronger Views)**
We use **random resizing, flipping, color jitter, and grayscale conversion** to generate two different versions of each image.  
```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])
```

---

## ** Model Architecture**  

SimCLR consists of **two main components**:  
1. **ResNet-18 Encoder** – Extracts image features.  
2. **Projection Head** – A simple MLP that maps features into contrastive space.  

```python
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.fc = nn.Identity()  # Remove classification layer
        self.encoder = resnet
        self.projection = ProjectionHead()

    def forward(self, x):
        features = self.encoder(x)
        return self.projection(features)
```

---

## ** Contrastive Loss (NT-Xent Loss)**  
Instead of classifying images, SimCLR **compares** them to different augmentations of the same image.  

```python
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
```

## Training the Model:-  

## ** Train the SimCLR Model**
```sh
python simclr_training.py
```

- **Training on CIFAR-10**
  - 50 epochs
  - Batch size: **256**
  - Learning rate: **0.001**
  - Optimizer: **Adam**

Example log:
```
Epoch 1/50 started...
Batch 10/196 - Loss: 5.8124
Batch 20/196 - Loss: 4.9231
Epoch [1/50] completed. Loss: 4.8569
```

 After training, the model is saved as: **`simclr_model.pth`**  

---

## Visualizing Learned Representations:-

### **5️⃣ Run t-SNE Visualization**
```sh
python simclr_visualization.py
```

This step:  
✅ Loads the trained SimCLR model.  
✅ Extracts **feature embeddings** from CIFAR-10 images.  
✅ Uses **t-SNE** to visualize high-dimensional features in **2D space**.  

Example output:
```plaintext
Generating t-SNE plot...
```

## ** Sample Visualization Output**  
The trained model clusters **similar images closer together** in the learned space.  

![t-SNE Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/T-SNE_Example.png/800px-T-SNE_Example.png)  

## Summary:-

SimCLR is a powerful **self-supervised learning** framework that **learns meaningful representations** without labeled data. This project **trains SimCLR on CIFAR-10** and **visualizes embeddings with t-SNE**.  
