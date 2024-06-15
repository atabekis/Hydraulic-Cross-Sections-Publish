import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from functions import to_torch, X, y
from augment import X_aug, augmented_targets
#
#
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Convert data to PyTorch datasets
# train_dataset, test_dataset = to_torch(X_aug, augmented_targets)
#
# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Define the model
# class HydraulicCNN(nn.Module):
#     def __init__(self):
#         super(HydraulicCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
#         self.fc1 = nn.Linear(64 * 5 * 5, 128, bias=False)
#         self.fc2 = nn.Linear(128, 1, bias=False)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 5 * 5)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # Initialize the model
# model = HydraulicCNN().to(device)
#
# # Define the RMSPE loss function
# def rmspe(y_pred, y_true):
#     return torch.sqrt(((y_true - y_pred) / y_true).pow(2).mean())
#
# # Define the optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training function
# def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         if epoch % 10 == 0:
#             print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Train the model
# train_model(model, train_loader, rmspe, optimizer, num_epochs=10)

# Define augmentation function
def augment(image):
    rotations = [np.rot90(image, k) for k in range(4)]  # id, r90, r180, r270
    flips = [np.fliplr(image), np.flipud(image)]
    flip_rotations = [np.flipud(np.rot90(image, k)) for k in [1, 3]]
    return rotations + flips + flip_rotations

# Define prediction function with augmentation
def predict_with_augmentation(model, images):
    def pred(model, image):
        model.eval()
        augmented_images = augment(image)
        augmented_tensors = [torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) for img in augmented_images]
        predictions = [model(tensor) for tensor in augmented_tensors]
        avg_prediction = torch.mean(torch.stack(predictions))
        return avg_prediction.item()

    predictions = []
    for image in images:
        predi = pred(model, image)
        predictions.append(predi)
    return predictions

# Example usage


# preds = predict_with_augmentation(model, X)
# print(preds)