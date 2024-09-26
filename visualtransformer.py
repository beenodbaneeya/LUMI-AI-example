import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from hdf5_dataset import HDF5Dataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}%')


#Load datasets
with HDF5Dataset('train_images.hdf5', transform=transform) as train_dataset, \
     HDF5Dataset('val_images.hdf5', transform=transform) as val_dataset:

    #subset for quick testing
    fraction = 0.1
    train_indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * fraction)]
    val_indices = torch.randperm(len(val_dataset))[:int(len(val_dataset) * fraction)]


    reduced_train_dataset = Subset(train_dataset, train_indices)
    reduced_val_dataset = Subset(val_dataset, val_indices)

    #DataLoaders
    #set pin_memory for faster data loading
    train_loader = DataLoader(reduced_train_dataset, batch_size=32, shuffle=True, num_workers=7, pin_memory=True)
    val_loader = DataLoader(reduced_val_dataset, batch_size=32, shuffle=False, num_workers=7, pin_memory=True)

    #Model SetUp
    model = vit_b_16(weights='DEFAULT')

    #these two lines could be useful if we need to use our own dataset and based on that if we need to modify the final layer of the model.
    #num_classes = 5  # This should be adjusted based on the dataset i.e. number of categories in our dataset. Its purpose is to define the number of output classes for the classification task.
    #model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)  #its purpose is to replace the pre-trained model's final classification layer by creating a new fully connected layer(where the number of output features should match the dataset1s classes)


     # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, criterion, optimizer, train_loader, val_loader)
    
    # Save model
    torch.save(model.state_dict(), 'vit_b_16_imagenet.pth')




    """
    fraction = 0.1
    indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * fraction)]
    reduced_train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(reduced_train_dataset, batch_size=32, shuffle=True, num_workers=7)

    indices = torch.randperm(len(val_dataset))[:int(len(val_dataset) * fraction)]
    reduced_val_dataset = Subset(val_dataset, indices)
    val_loader = DataLoader(reduced_val_dataset, batch_size=32, shuffle=True, num_workers=7)

    train_model(model, criterion, optimizer, train_loader, val_loader)
    torch.save(model.state_dict(), 'vit_b_16_imagenet.pth')
    """


