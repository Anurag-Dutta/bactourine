import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import os
import time


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = r"C:\Users\Anurag Dutta\Desktop\project"

dataset = datasets.ImageFolder(root=data_path, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_block(64, 128, 2, start_with_relu=False),
            self._make_block(128, 256, 2, start_with_relu=True),
            self._make_block(256, 728, 2, start_with_relu=True)
        )
        
        self.middle_flow = nn.Sequential(
            *[self._make_block(728, 728, 1, start_with_relu=True) for _ in range(8)]
        )
        
        self.exit_flow = nn.Sequential(
            self._make_block(728, 1024, 2, start_with_relu=True),
            SeparableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_block(self, in_channels, out_channels, stride, start_with_relu=True):
        block = []
        if start_with_relu:
            block.append(nn.ReLU())
        block.append(SeparableConv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(SeparableConv2d(out_channels, out_channels, 3, padding=1))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.MaxPool2d(3, stride=2, padding=1))
        return nn.Sequential(*block)
    
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = Xception(num_classes=3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in val_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        all_preds.extend(predicted.numpy())
        all_targets.extend(targets.numpy())

    print(f"Validation Accuracy: {correct / total * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'xception_classifier.pth')