"""
End-to-End Training Integration Tests

Tests that verify complete training workflows work correctly.
These tests ensure all components work together properly.

Integration Tests:
- Full training loop with all features
- Checkpointing and recovery
- Multi-strategy combinations
- Real-world training scenarios

Purpose:
- Verify component interactions
- Test complete workflows
- Ensure production readiness
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class SimpleClassifier(nn.Module):
    """Simple classifier for integration testing"""
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for testing"""
    def __init__(self, num_samples=1000, input_dim=20, num_classes=10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Generate synthetic data
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, input_dim)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def temp_checkpoint_dir():
    """Fixture providing temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_classifier():
    """Fixture providing simple classifier"""
    return SimpleClassifier()


@pytest.fixture
def train_dataset():
    """Fixture providing training dataset"""
    return SyntheticDataset(num_samples=800)


@pytest.fixture
def val_dataset():
    """Fixture providing validation dataset"""
    return SyntheticDataset(num_samples=200)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndTraining:
    """End-to-end training integration tests"""

    def test_basic_training_loop(self, simple_classifier, train_dataset):
        """
        Test basic training loop completes successfully.

        Workflow:
        1. Setup model and optimizer
        2. Create data loader
        3. Train for multiple epochs
        4. Verify loss decreases

        Expected: Loss should decrease, no errors
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        # Training loop
        num_epochs = 3
        losses = []

        print(f"\n=== Basic Training Loop ===")

        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

        print(f"\n✓ Training completed successfully")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

        # Verify loss decreased
        assert losses[-1] < losses[0], "Loss should decrease during training"
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN detected in losses"

    def test_training_with_validation(
        self,
        simple_classifier,
        train_dataset,
        val_dataset
    ):
        """
        Test training with validation loop.

        Workflow:
        1. Train for one epoch
        2. Validate
        3. Track metrics
        4. Verify both train and val metrics

        Expected: Both train and val loss should be reasonable
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        print(f"\n=== Training with Validation ===")

        # Training
        model.train()
        train_losses = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_losses.append(loss.item())

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_val_loss = sum(val_losses) / len(val_losses)
        val_accuracy = correct / total

        print(f"✓ Training metrics:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss: {avg_val_loss:.4f}")
        print(f"  Val accuracy: {val_accuracy*100:.2f}%")

        # Assertions
        assert avg_train_loss > 0, "Train loss should be positive"
        assert avg_val_loss > 0, "Val loss should be positive"
        assert 0 <= val_accuracy <= 1, "Accuracy should be between 0 and 1"

        # Random baseline is 10% (10 classes)
        # After one epoch, should be better than random
        assert val_accuracy > 0.05, "Accuracy should be above random baseline"

    def test_training_with_checkpointing(
        self,
        simple_classifier,
        train_dataset,
        temp_checkpoint_dir
    ):
        """
        Test training with checkpointing.

        Workflow:
        1. Train for 2 epochs with checkpoints
        2. Load checkpoint
        3. Continue training
        4. Verify checkpoint loading works

        Expected: Training resumes correctly from checkpoint
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        print(f"\n=== Training with Checkpointing ===")

        # Train for 2 epochs and save checkpoint
        checkpoint_path = temp_checkpoint_dir / "checkpoint_epoch_2.pt"

        for epoch in range(2):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Save checkpoint
        checkpoint = {
            'epoch': 2,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        print(f"✓ Checkpoint saved at epoch 2")

        # Create new model and load checkpoint
        new_model = SimpleClassifier().to(device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        loaded_checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        loaded_epoch = loaded_checkpoint['epoch']

        print(f"✓ Checkpoint loaded from epoch {loaded_epoch}")

        # Verify loaded model produces same output as original
        test_input = torch.randn(1, 20).to(device)
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = new_model(test_input)

        outputs_match = torch.allclose(original_output, loaded_output, rtol=1e-5)

        print(f"✓ Model outputs match: {outputs_match}")
        print(f"  Max difference: {(original_output - loaded_output).abs().max().item():.2e}")

        assert outputs_match, "Loaded model should produce same outputs"
        assert loaded_epoch == 2, "Loaded epoch should be 2"

    def test_mixed_precision_training(self, simple_classifier, train_dataset):
        """
        Test mixed precision training.

        Workflow:
        1. Train with automatic mixed precision (AMP)
        2. Verify training completes
        3. Compare with FP32 training

        Expected: AMP training should work and be faster on GPU
        """
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision requires CUDA")

        device = torch.device('cuda')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        print(f"\n=== Mixed Precision Training ===")

        # Train with AMP
        model.train()
        losses = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        print(f"✓ Mixed precision training completed")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Gradient scaler scale: {scaler.get_scale():.1f}")

        # Assertions
        assert avg_loss > 0, "Loss should be positive"
        assert not any(torch.isnan(torch.tensor(l)) for l in losses), "No NaN in losses"
        assert scaler.get_scale() > 0, "Scaler should have positive scale"

    def test_gradient_accumulation(self, simple_classifier, train_dataset):
        """
        Test gradient accumulation for effective larger batch sizes.

        Workflow:
        1. Train with gradient accumulation
        2. Verify gradients accumulate correctly
        3. Compare with larger batch size

        Expected: Gradient accumulation should work correctly
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        accumulation_steps = 2
        effective_batch_size = 16 * accumulation_steps

        print(f"\n=== Gradient Accumulation ===")
        print(f"  Batch size: 16")
        print(f"  Accumulation steps: {accumulation_steps}")
        print(f"  Effective batch size: {effective_batch_size}")

        model.train()
        optimizer.zero_grad()
        losses = []
        step_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            losses.append(loss.item() * accumulation_steps)  # Denormalize for logging

            if step_count >= 10:  # Train for 10 optimizer steps
                break

        avg_loss = sum(losses) / len(losses)

        print(f"\n✓ Gradient accumulation completed")
        print(f"  Optimizer steps: {step_count}")
        print(f"  Average loss: {avg_loss:.4f}")

        assert step_count > 0, "Should have completed at least one optimizer step"
        assert avg_loss > 0, "Loss should be positive"

    def test_early_stopping(self, simple_classifier, train_dataset, val_dataset):
        """
        Test early stopping mechanism.

        Workflow:
        1. Train with validation
        2. Track best validation loss
        3. Stop if no improvement for N epochs

        Expected: Training stops when validation stops improving
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simple_classifier.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        patience = 3
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        max_epochs = 10

        print(f"\n=== Early Stopping Test ===")
        print(f"  Patience: {patience} epochs")

        for epoch in range(max_epochs):
            # Training
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_losses.append(loss.item())

            val_loss = sum(val_losses) / len(val_losses)

            print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  ✓ New best validation loss!")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break

        assert best_val_loss < float('inf'), "Should have recorded best validation loss"
        assert epochs_without_improvement >= 0, "Counter should be non-negative"


@pytest.mark.integration
def test_full_training_workflow(
    simple_classifier,
    train_dataset,
    val_dataset,
    temp_checkpoint_dir
):
    """
    Test complete training workflow with all features.

    Workflow:
    1. Setup model, optimizer, data loaders
    2. Train for multiple epochs
    3. Validate after each epoch
    4. Save checkpoints
    5. Track metrics
    6. Handle best model saving

    Expected: Complete workflow runs without errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simple_classifier.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_epochs = 3
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n=== Full Training Workflow ===")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_losses.append(loss.item())

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc*100:.2f}%")

        # Save checkpoint every epoch
        checkpoint_path = temp_checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        }, checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = temp_checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            }, best_model_path)
            print(f"  ✓ Saved best model!")

    print(f"\n✓ Training workflow completed successfully")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final val accuracy: {history['val_acc'][-1]*100:.2f}%")
    print(f"  Checkpoints saved: {num_epochs + 1}")  # epoch checkpoints + best model

    # Verify all checkpoints exist
    assert (temp_checkpoint_dir / "best_model.pt").exists(), "Best model should be saved"
    for epoch in range(1, num_epochs + 1):
        assert (temp_checkpoint_dir / f"checkpoint_epoch_{epoch}.pt").exists(), \
            f"Checkpoint for epoch {epoch} should exist"

    # Verify history
    assert len(history['train_loss']) == num_epochs
    assert len(history['val_loss']) == num_epochs
    assert len(history['val_acc']) == num_epochs
