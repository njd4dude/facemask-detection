import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet34_Weights
import wandb
import time
import os

# 3/23: left off learning how to use wandbai and implementation.
# 3/24: left off bout to test 1 run and see how it logs on wandb

# Load the pre-trained ResNet34 model
def loadModel():
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    return model

def customizePretrainedModel(model):
    # Freeze all layers
    print(model)
    for param in model.parameters():
        param.requires_grad = False

    # Customize output layer for 3-class classification
    n_inputs = model.fc.in_features
    model.fc = nn.Linear(n_inputs, 3)
    print(f"Reinitialized model with output features as 3: {model.fc}")

def printModelArchitecture(model):
    # Display model architecture
    features_resnet34 = list(model.children())
    print(features_resnet34)

def getDevice():
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    return device

def defineModelHyperParams(learning_rate_value=0.001, momentum_value=0.9):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate_value, momentum=momentum_value)
    return criterion, optimizer


# Training function
def train(train_dataloader, validation_dataloader, criterion, optimizer, device, n_epochs=4, dataset_name="default"):
    print(f"\nStarting training with device: {device}\n",)
    
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch} - Training...")
        
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_predictions = 0
        
        model.train()  # Set model to training mode
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_predictions += labels.size(0)
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i+1}, Training Loss: {running_loss / 20:.4f}")
                running_loss = 0.0
        
        train_accuracy = 100 * correct_train_predictions / total_train_predictions
        print(f"Epoch {epoch}, Training Accuracy: {train_accuracy:.2f}%")

        
        # Validation
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        val_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch}, Validation Loss: {val_loss / len(validation_dataloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # log training and validation metrics to wandb
        run.log({
            "epoch": epoch,
            "training_loss": running_loss / len(train_dataloader),
            "training_accuracy": train_accuracy,
            "validation_loss": val_loss / len(validation_dataloader),
            "validation_accuracy": val_accuracy
        })

    
    print("\nFinished Training")

    output_weights_dir = f"Weights/{dataset_name}"
    os.makedirs(output_weights_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    torch.save(model.state_dict(), os.path.join(output_weights_dir, "trained_model_weights.pth"))
    run.finish()

    print(f"Saved trained model weights to {output_weights_dir} and logged metrics to wandb.")

# taks 3/3: left off going to validation this trained og model on the my opencv camera
if __name__ == "__main__":

    numEpochs = 8
    learningRate = 0.0001
    datasetFolderName ="dataset2-trashCameraQuality"

    # Start a new wandb run to track this script.
    run = wandb.init(
        project="facemask_detection",    # Specify your project
        config={                         # Track hyperparameters and metadata
            "learning_rate": learningRate,
            "epochs": numEpochs,
            "pretrained": False,
            "dataset": datasetFolderName,
        },
    )

    model = loadModel()
    customizePretrainedModel(model)
    printModelArchitecture(model)
    device = getDevice()
    model = model.to(device)
    criterion, optimizer = defineModelHyperParams(learning_rate_value=learningRate)

    train_dataloader = torch.load(f"Dataloaders/{datasetFolderName}/train_dataloader.pth", weights_only=False)
    validation_dataloader =  torch.load(f"DataLoaders/{datasetFolderName}/validation_dataloader.pth", weights_only=False)

    print("\n\n CHECKING DATALOADERS----------------")
    print("\ntrain_dataloader")
    print(train_dataloader)
    print("\nvalidation_dataloader")
    print(validation_dataloader)

    start_time = time.time()  # or time.perf_counter()
    train(train_dataloader, validation_dataloader, criterion, optimizer, device, numEpochs, datasetFolderName) 
    end_time = time.time()  # or time.perf_counter()]
    elapsed_time = end_time - start_time
    # Write the elapsed time to a text file
    with open("elapsed_time.txt", "a") as file:
        file.write(f"Elapsed Time: {elapsed_time} seconds\n")

    print(f"Elapsed Time: {elapsed_time} seconds")