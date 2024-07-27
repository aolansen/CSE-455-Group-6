import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
import time
from tqdm import tqdm
from torchinfo import summary

import preprocessor

class CustomHead(nn.Module):
    def __init__(self, featr):
        super(CustomHead, self).__init__()
        self.fc1 = nn.Linear(featr, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x




# Load a pre-trained ResNet model
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
featr = model.fc.in_features
model.fc = CustomHead(featr)


summary(model, input_size=(preprocessor.batch_size, preprocessor.channels, preprocessor.img_width, preprocessor.img_height))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001)

# Training loop
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load('emotion_detector_epoch_2l.pth'))
for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    model.train()
    running_loss = 0.0

    start_time = time.time()

    # Use tqdm for progress bar
    with tqdm(total=len(preprocessor.dataloaders['train']), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_idx, (inputs, labels) in enumerate(preprocessor.dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    epoch_loss = running_loss / len(preprocessor.dataloaders['train'].dataset)
    elapsed_time = time.time() - start_time
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s')

    # Save checkpoint every epoch
    torch.save(model.state_dict(), f'emotion_detector_epoch_{epoch + 1}.pth')