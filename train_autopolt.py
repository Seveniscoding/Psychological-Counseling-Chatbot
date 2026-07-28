import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
from PIL import Image
import timm
from dataset import CustomImageDataset
import matplotlib.pyplot as plt #for trainning loss and accurcy showing
import pandas as pd
import yaml
from datetime import datetime
import shutil
import numpy as np

#manage
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Trainer:
    def __init__(self, config, save_path):
        """
        Initialize the Trainer with data transformations, datasets, model, and optimization components.
        
        Args:
        save_path (str): Directory to save model checkpoints.
        """
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors (scales values to [0, 1])
            transforms.Normalize((0.5, 0.5, 0.5))  # Normalize with mean=0.5 and std=0.5 (scales to [-1, 1])

        ])
        
        self.save_path = save_path
        
        # Create datasets and data loaders
        self.train_dataset = CustomImageDataset(root_dir=config['train_data_path'], transform=self.transform)
        self.test_dataset = CustomImageDataset(root_dir=config['test_data_path'], transform=test_transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config['batch_size'], shuffle=False)

        
        # Initialize the model
        self.model = timm.create_model('timm/convnextv2_pico.fcmae_ft_in1k', pretrained=True)
        print(self.model)
        self.model.head.fc = nn.Linear(512, config['num_classes'])  # Modify the final layer for 8 classes
        
        # Define loss function
        # CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        # Adam optimizer with a learning rate of 0.00001
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Move model to the selected device

        #adding for result showing
        self.train_losses = []
        self.val_accuracies = []

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.
        
        Args:
        num_epochs (int): Number of epochs to train for.
        """
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)
            running_loss = 0.0
            
            # Training loop
            for images, labels in tqdm(self.train_loader):
                # Move data to the selected device
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()  # Zero the parameter gradients
                
                outputs = self.model(images)  # Forward pass: compute predicted outputs by passing inputs to the model
                
                # Calculate the loss
                # CrossEntropyLoss expects raw outputs and class indices, not one-hot encoded
                # It internally applies softmax to the output
                loss = self.criterion(outputs, labels)
                
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step()  # Perform a single optimization step (parameter update)
                
                running_loss += loss.item()  # Accumulate the loss
            
            # Print average loss for the epoch
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")

            #store average loss for ploting
            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            


            
            # Evaluation loop
            self.model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm, etc.)
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
                for images, labels in tqdm(self.test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    predicted = torch.argmax(outputs.data, 1)  # Get the index of the max log-probability
                    predicted_labels = torch.argmax(labels.data, 1)  # Convert one-hot to class indices
                    total += labels.size(0)
                    correct += (predicted == predicted_labels).sum().item()

            
            # Print accuracy
            print(f"Accuracy on test set: {(correct / total) * 100}%")
            accuracy = (correct / total) * 100
            self.val_accuracies.append(accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model, os.path.join(self.save_path, 'best.pt'))
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
            # Save model checkpoint
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model, os.path.join(self.save_path, f"{epoch}.pt"))
            
            #Plot after each epoch
            self.plot_metrics()
            
    # for polting and image saving 
    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        epochs = list(range(1, len(self.train_losses) + 1))
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss and Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.save_path, 'training_progress.png')
        plt.savefig(plot_path)
        plt.close()
    
        best_epoch = int(np.argmax(self.val_accuracies))  #find the best accuracy 
        best_accuracy = self.val_accuracies[best_epoch]
        best_loss = self.train_losses[best_epoch]
    
        df = pd.DataFrame({
            'Epoch': epochs,
            'Training Loss': self.train_losses,
            'Validation Accuracy': self.val_accuracies
        })
        summary = pd.DataFrame({
            'Best Epoch': [best_epoch + 1],
            'Best Accuracy': [best_accuracy],
            'Best Loss': [best_loss]
        })
    
        with pd.ExcelWriter(os.path.join(self.save_path, 'training_metrics.xlsx')) as writer:
            df.to_excel(writer, sheet_name='All Metrics', index=False)
            summary.to_excel(writer, sheet_name='Best Performance', index=False)


            
# Usage updated for management
config = load_config('configs/config_batchsize16_lr1e-4_epochs_20.yaml')   #changed batchesize from 10 to 16

# Auto-generate folder name with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
save_path = f"checkpoints/run_{timestamp}"
os.makedirs(save_path, exist_ok=True)

# Save a copy of the config file into the checkpoint folder
shutil.copy('configs/config_batchsize16_lr1e-4_epochs_20.yaml', os.path.join(save_path, 'config.yaml'))

trainer = Trainer(config, save_path)
trainer.train(config['epochs'])


