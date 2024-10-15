# %%
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os

from torch.optim import Adam
import torch
import torch.nn as nn
import pandas as pd

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 64
MODEL_NAME = "main"

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"  # for those with MacBooks
elif torch.cuda.is_available():
    DEVICE = "cuda"
print(f"Using {DEVICE} DEVICE")

# Dataset and DataLoader (using MNIST dataset)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),  # MNIST normalization values
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

img_shape = train_dataset[0][0].shape
IMG_SIZE = img_shape[1]
IN_CHANNELS = img_shape[0]
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}, with {IN_CHANNELS} channels")


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE, NUM_CLASSES)

    def forward(self, input):
        input = input.view(-1, IMG_SIZE * IMG_SIZE)
        output = self.fc1(input)

        return output


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
    return total_loss / len(train_loader), correct / len(train_loader.dataset) * 100


def test_model_accuracy(model, criterion):
    model.eval()

    accuracy = 0.0
    running_loss = 0.0
    number_of_samples = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            running_loss += loss.item()
            number_of_samples += 1
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels.to(DEVICE)).sum().item()

    test_loss = running_loss / number_of_samples
    accuracy = 100 * accuracy / number_of_samples / BATCH_SIZE

    return accuracy, test_loss


def main():
    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"experiments/{timestamp}_{MODEL_NAME}"
    os.makedirs(experiment_folder, exist_ok=True)

    model = Network().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"Total number of parameters: {count_parameters(model)}")

    all_data = []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2f}")

        test_accuracy, test_loss = test_model_accuracy(model, criterion)
        print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2f}")

        data = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        }
        all_data.append(data)

    # Save results to experiment folder
    df = pd.DataFrame(all_data)
    df.to_csv(f"{experiment_folder}/results.csv", index=False)

    # Save the trained model to experiment folder
    torch.save(model.state_dict(), f"{experiment_folder}/model.pth")
    print(f"Model saved to {experiment_folder}/model.pth")

    print("Finished Training")


if __name__ == "__main__":
    main()
