import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import math
import seaborn as sns


from kan_convolutional.KANLinear import KANLinear
from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional import convolution 

root = "C:\\Users\\Anurag Dutta\\Desktop\\project\\data"


class UrineCultureData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.class_names = []

        # Iterate through each subfolder (class)
        for label, subfolder in enumerate(os.listdir(root_dir)):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                self.class_names.append(subfolder)  # Store class names
                for img_file in os.listdir(subfolder_path):
                    if img_file.endswith('.jpg'):
                        self.image_files.append(os.path.join(subfolder_path, img_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def get_class_names(self):
        return self.class_names

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = UrineCultureData(root_dir=root, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class KANC_MLP(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super(KANC_MLP, self).__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device
        )
        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            device=device
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(625, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x

model = KANC_MLP()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')


conf_matrix = confusion_matrix(all_labels, all_preds)

formatted_classes = [rf'\texttt{{{cls}}}' for cls in dataset.get_class_names()]

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 30,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=formatted_classes)
disp.plot(cmap=plt.cm.BuPu, ax=ax)
plt.title(r'\textbf{KAN-Convolution}', fontsize=40, pad=20)
plt.xlabel(r'\textbf{Predicted Label}', fontsize=28, labelpad=30)
plt.ylabel(r'\textbf{True Label}', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()

# Save the confusion matrix as a PDF
plt.savefig('kanc_mlp_cm.pdf', format='pdf')

# Show the plot
plt.show()
