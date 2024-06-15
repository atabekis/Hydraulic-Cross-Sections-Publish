import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define SpongeData class
class SpongeData(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        self.targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Define to_torch function
def to_torch(X, y, split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    train = SpongeData(X_train, y_train)
    test = SpongeData(X_test, y_test)
    return train, test


# Load augmented data
augmented_images = np.load('data/pub_input.npy') # Subsample of 10 images
augmented_labels = np.load('data/pub_out.npy')  # Subsample of 10 labels

# Convert data to PyTorch datasets
train_dataset, test_dataset = to_torch(augmented_images, augmented_labels)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the model
class VGGNetNoBias(nn.Module):
    def __init__(self, bias=False):
        super(VGGNetNoBias, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 4096, bias=bias),  # Adjust input size based on final conv layer's output
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1, bias=bias)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


# Initialize the model
model = VGGNetNoBias().to(device)


# Define the RMSPE loss function
def rmspe(y_pred, y_true):
    return torch.sqrt(((y_true - y_pred) / y_true).pow(2).mean())


# Define the optimizer with L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


# Train the model
train_model(model, train_loader, rmspe, optimizer, num_epochs=10)


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

X = np.load('data/pub_input.npy')
preds = predict_with_augmentation(model, X)
print(preds)