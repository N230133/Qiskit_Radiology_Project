"""
Step 2: Train classical CNN baseline and extract features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

# Load preprocessed data
print("Loading data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Convert to tensors (grayscale - take first channel)
X_train_tensor = torch.tensor(X_train[:, :, :, 0], dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test[:, :, :, 0], dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define CNN backbone (feature extractor)
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return x

# Define full classifier
class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train classical model
print("Training classical CNN baseline...")
model = ClassicalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save classical model
torch.save(model.state_dict(), 'classical_cnn.pth')
print("✓ Classical CNN trained and saved!")

# Extract features for quantum model
print("Extracting features for quantum model...")
model.eval()
with torch.no_grad():
    # Use backbone only
    backbone = model.backbone
    train_features = backbone(X_train_tensor).numpy()
    test_features = backbone(X_test_tensor).numpy()

# Reduce to 4 features using PCA
print("Reducing features to 4 dimensions (PCA)...")
pca = PCA(n_components=4)
train_features_reduced = pca.fit_transform(train_features)
test_features_reduced = pca.transform(test_features)

# Save features
np.save('train_features.npy', train_features_reduced)
np.save('test_features.npy', test_features_reduced)
np.save('train_labels.npy', y_train)
np.save('test_labels.npy', y_test)

# Save PCA for later use
import pickle
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

print(f"✓ Features extracted: Train {train_features_reduced.shape}, Test {test_features_reduced.shape}")