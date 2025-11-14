#!/usr/bin/env python3
"""
Week 1 Starter Code: Single-GPU Training Baseline

YOUR TASK: Complete all the TODO sections and fill in the ______ blanks.

This script trains a CNN on MNIST. You'll implement:
- Data loading
- Model definition
- Training loop
- Validation
- Checkpointing

TARGET: >95% validation accuracy in <10 minutes

HINTS:
- Read the TODO comments carefully
- Check the PyTorch docs if stuck
- Start simple, then add features
- Test frequently!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os


# ============================================================================
# SECTION 1: MODEL DEFINITION
# ============================================================================

class SimpleCNN(nn.Module):
    """
    A simple CNN for MNIST classification

    TODO: Complete the model architecture
    - Conv layer 1: 1 input channel, 32 output channels, 3x3 kernel
    - Conv layer 2: 32 input channels, 64 output channels, 3x3 kernel
    - FC layer 1: 9216 inputs (64*12*12 after pooling), 128 outputs
    - FC layer 2: 128 inputs, 10 outputs (10 digits)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # TODO: Define convolutional layers
        self.conv1 = nn.Conv2d(______, ______, kernel_size=______)  # Fill in: in_channels, out_channels, kernel_size
        self.conv2 = ______(______, ______, kernel_size=______)  # Fill in: Conv2d layer

        # TODO: Define fully connected layers
        self.fc1 = ______(______, ______)  # Fill in: Linear layer (9216 → 128)
        self.fc2 = ______(______, ______)  # Fill in: Linear layer (128 → 10)

        # Pooling and dropout (provided)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Forward pass

        TODO: Implement the forward pass
        - Apply conv1 → ReLU → pool
        - Apply conv2 → ReLU → pool
        - Flatten
        - Apply fc1 → ReLU → dropout
        - Apply fc2

        Args:
            x: Input tensor [batch_size, 1, 28, 28]

        Returns:
            Output tensor [batch_size, 10]
        """
        # TODO: First conv block
        x = self.conv1(x)
        x = F._______(x)  # Fill in: activation function
        x = self.______(x)  # Fill in: pooling

        # TODO: Second conv block
        x = ______(x)  # Fill in: conv2
        x = ______(x)  # Fill in: activation
        x = ______(x)  # Fill in: pooling

        # TODO: Flatten for FC layers
        # Hint: x.view(x.size(0), -1) flattens while keeping batch dimension
        x = ______.______(______.size(0), -1)

        # TODO: First FC block
        x = self.fc1(x)
        x = ______(x)  # Fill in: activation
        x = self.dropout(x)

        # TODO: Output layer
        x = ______(x)  # Fill in: fc2

        return x


# ============================================================================
# SECTION 2: DATA LOADING
# ============================================================================

def get_data_loaders(batch_size=64, data_dir='./data'):
    """
    Create train and validation data loaders for MNIST

    TODO: Implement data loading with proper transforms

    Args:
        batch_size: Batch size for training
        data_dir: Directory to store/load data

    Returns:
        train_loader, val_loader
    """
    # TODO: Define transforms
    # Hint: MNIST needs normalization with mean=0.1307, std=0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((______,), (______,))  # Fill in: mean and std
    ])

    # TODO: Load training data
    # Hint: Use datasets.MNIST with train=True
    train_dataset = datasets.______(
        root=data_dir,
        train=______,  # Fill in: True or False?
        download=True,
        transform=transform
    )

    # TODO: Load validation data
    val_dataset = datasets.MNIST(
        root=data_dir,
        train=______,  # Fill in: True or False for validation?
        download=True,
        transform=______  # Fill in: transform
    )

    # TODO: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=______,  # Fill in: batch_size
        shuffle=______,  # Fill in: True or False for training?
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=______,  # Fill in: True or False for validation?
        num_workers=2
    )

    return train_loader, val_loader


# ============================================================================
# SECTION 3: TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    TODO: Implement the training loop

    Args:
        model: Neural network
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.____()  # TODO: Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (data, target) in enumerate(pbar):
        # TODO: Move data to device
        data, target = data.to(______), target.to(______)  # Fill in: device

        # TODO: Zero gradients
        ______.zero_grad()  # Fill in: optimizer

        # TODO: Forward pass
        output = ______(data)  # Fill in: model

        # TODO: Compute loss
        loss = ______(output, target)  # Fill in: criterion

        # TODO: Backward pass
        loss.______()  # Fill in: backward

        # TODO: Update weights
        optimizer.______()  # Fill in: step

        # Compute accuracy
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================================
# SECTION 4: VALIDATION FUNCTION
# ============================================================================

def validate(model, val_loader, criterion, device):
    """
    Validate the model

    TODO: Implement validation loop (no gradient computation needed!)

    Args:
        model: Neural network
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.____()  # TODO: Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # TODO: Disable gradient computation for validation
    with torch.______():  # Fill in: no_grad
        for data, target in val_loader:
            # TODO: Move data to device
            data, target = ______.to(device), ______.to(device)

            # TODO: Forward pass
            output = model(______)  # Fill in: data

            # TODO: Compute loss
            loss = criterion(______, ______)  # Fill in: output, target

            # Compute accuracy
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================================
# SECTION 5: CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, epoch, loss, filepath='checkpoint.pt'):
    """
    Save model checkpoint

    TODO: Save all necessary components for resuming training

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        filepath: Where to save
    """
    # TODO: Create checkpoint dictionary
    checkpoint = {
        'epoch': ______,  # Fill in: epoch
        'model_state_dict': model.______(),  # Fill in: state_dict
        'optimizer_state_dict': ______.state_dict(),  # Fill in: optimizer
        'loss': ______,  # Fill in: loss
    }

    # TODO: Save to file
    torch.______(checkpoint, filepath)  # Fill in: save
    print(f'✓ Checkpoint saved to {filepath}')


def load_checkpoint(model, optimizer, filepath='checkpoint.pt'):
    """
    Load model checkpoint

    TODO: Load and restore model and optimizer states

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filepath: Checkpoint file path

    Returns:
        epoch: Epoch to resume from
        loss: Loss at checkpoint
    """
    if not os.path.exists(filepath):
        print(f'No checkpoint found at {filepath}')
        return 0, float('inf')

    # TODO: Load checkpoint
    checkpoint = torch.______(filepath)  # Fill in: load

    # TODO: Restore states
    model.______(checkpoint['model_state_dict'])  # Fill in: load_state_dict
    optimizer.load_state_dict(checkpoint['______'])  # Fill in: key name

    epoch = checkpoint['______']  # Fill in: key name
    loss = checkpoint['______']  # Fill in: key name

    print(f'✓ Checkpoint loaded from {filepath} (epoch {epoch})')
    return epoch, loss


# ============================================================================
# SECTION 6: MAIN TRAINING LOOP
# ============================================================================

def main():
    """
    Main training function

    TODO: Put it all together!
    """
    # Hyperparameters
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001

    # TODO: Set device
    device = torch.device('cuda' if torch.cuda.______ else 'cpu')  # Fill in: is_available
    print(f'Using device: {device}')

    # TODO: Load data
    train_loader, val_loader = get_data_loaders(batch_size)

    # TODO: Create model
    model = ______().to(device)  # Fill in: SimpleCNN
    print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

    # TODO: Create loss function and optimizer
    criterion = nn.______()  # Fill in: CrossEntropyLoss
    optimizer = torch.optim.______(model.parameters(), lr=learning_rate)  # Fill in: Adam

    # Training loop
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # TODO: Train for one epoch
        train_loss, train_acc = ______(
            model, train_loader, criterion, optimizer, device, epoch
        )  # Fill in: train_epoch

        # TODO: Validate
        val_loss, val_acc = ______(
            model, val_loader, criterion, device
        )  # Fill in: validate

        # Print results
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%')

        # TODO: Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ______(model, optimizer, epoch, val_loss, 'best_model.pt')  # Fill in: save_checkpoint
            print(f'  ✓ New best model! Validation accuracy: {val_acc:.2f}%')

    print(f'\n{"="*60}')
    print(f'Training complete! Best validation accuracy: {best_val_acc:.2f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()


# ============================================================================
# BONUS CHALLENGES (Optional - Try after completing the basics!)
# ============================================================================

"""
Once you have the basic version working, try these enhancements:

1. Add learning rate scheduling
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

2. Add early stopping
   If validation loss doesn't improve for 3 epochs, stop training

3. Add data augmentation
   transforms.RandomRotation(10),
   transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))

4. Try different optimizers
   SGD, RMSprop, AdamW

5. Experiment with model architectures
   Add more conv layers, change filter sizes, add batch normalization

6. Track metrics
   Use tensorboard or wandb to visualize training

7. Add gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

8. Implement mixed precision training
   Use torch.cuda.amp for faster training
"""
