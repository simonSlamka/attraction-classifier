import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from dataset import ImageFolderDataset
from model import BinaryImageClassifier

# Check if GPU is available and if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = ImageFolderDataset("/home/simtoon/smtn_girls_likeOrNot/.faces")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2)

# Create data loaders for training and validation sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize the model
model = BinaryImageClassifier()
model = model.to(device)  # Move the model to GPU if available

# Define the loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU if available

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Run validation after each epoch
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move images and labels to GPU if available
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        print('Validation loss after epoch %d: %.3f' % (epoch + 1, val_loss/len(val_loader)))
        print('Validation accuracy after epoch %d: %d %%' % (epoch + 1, 100 * correct / total))

print("Finished Training")
# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Saved the model to model.pth")