"""
Checkpoint Compatibility Regression Tests

Tests that ensure checkpoint format remains compatible across versions.
These tests prevent breaking changes in checkpoint save/load functionality.

Regression Tests:
- Checkpoint format compatibility
- Model loading from old checkpoints
- Optimizer state compatibility
- Metadata preservation

Purpose:
- Ensure checkpoint backward compatibility
- Detect breaking changes in save/load
- Verify checkpoint migration paths
"""

import pytest
import torch
import torch.nn as nn
import os
import tempfile
from pathlib import Path
import json


class TestModel(nn.Module):
    """Test model for checkpoint compatibility"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CheckpointVersionManager:
    """Manages checkpoint versions for regression testing"""

    def __init__(self, version_dir='test_results/checkpoint_versions'):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def save_versioned_checkpoint(
        self,
        version: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict
    ):
        """Save a versioned checkpoint"""
        checkpoint_path = self.version_dir / f"checkpoint_v{version}.pt"

        checkpoint = {
            'version': version,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_versioned_checkpoint(self, version: str):
        """Load a versioned checkpoint"""
        checkpoint_path = self.version_dir / f"checkpoint_v{version}.pt"

        if not checkpoint_path.exists():
            return None

        return torch.load(checkpoint_path)

    def get_checkpoint_info(self, version: str) -> dict:
        """Get information about a checkpoint version"""
        checkpoint = self.load_versioned_checkpoint(version)

        if checkpoint is None:
            return None

        return {
            'version': checkpoint.get('version'),
            'keys': list(checkpoint.keys()),
            'model_keys': list(checkpoint['model_state_dict'].keys()),
            'metadata': checkpoint.get('metadata', {})
        }


@pytest.fixture
def checkpoint_manager():
    """Fixture providing checkpoint version manager"""
    return CheckpointVersionManager()


@pytest.fixture
def test_model():
    """Fixture providing test model"""
    torch.manual_seed(42)
    return TestModel()


@pytest.mark.regression
class TestCheckpointCompatibility:
    """Checkpoint compatibility regression tests"""

    def test_checkpoint_format_v1(self, checkpoint_manager, test_model):
        """
        Test checkpoint format version 1.

        Purpose: Establish baseline checkpoint format
        """
        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)

        # Train for a few steps
        for _ in range(5):
            x = torch.randn(4, 10)
            target = torch.randint(0, 5, (4,))
            loss = nn.functional.cross_entropy(test_model(x), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save checkpoint
        metadata = {
            'epoch': 1,
            'loss': 1.5,
            'framework_version': '1.0.0'
        }

        checkpoint_path = checkpoint_manager.save_versioned_checkpoint(
            '1.0',
            test_model,
            optimizer,
            metadata
        )

        # Verify checkpoint was saved
        assert checkpoint_path.exists(), "Checkpoint not saved"

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path)

        assert 'version' in checkpoint, "Version not in checkpoint"
        assert 'model_state_dict' in checkpoint, "Model state not in checkpoint"
        assert 'optimizer_state_dict' in checkpoint, "Optimizer state not in checkpoint"
        assert 'metadata' in checkpoint, "Metadata not in checkpoint"

        print(f"\n✓ Checkpoint v1.0 format:")
        print(f"  Keys: {list(checkpoint.keys())}")
        print(f"  Model params: {len(checkpoint['model_state_dict'])}")
        print(f"  Optimizer groups: {len(checkpoint['optimizer_state_dict']['param_groups'])}")
        print(f"  Metadata: {checkpoint['metadata']}")

    def test_load_old_checkpoint(self, checkpoint_manager, test_model):
        """
        Test loading checkpoint from previous version.

        Purpose: Ensure backward compatibility
        """
        # Try to load v1.0 checkpoint
        old_checkpoint = checkpoint_manager.load_versioned_checkpoint('1.0')

        if old_checkpoint is None:
            # Create it first
            optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
            metadata = {'epoch': 1, 'loss': 1.5, 'framework_version': '1.0.0'}
            checkpoint_manager.save_versioned_checkpoint('1.0', test_model, optimizer, metadata)
            old_checkpoint = checkpoint_manager.load_versioned_checkpoint('1.0')

        # Create new model and load old checkpoint
        new_model = TestModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load state
        try:
            new_model.load_state_dict(old_checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(old_checkpoint['optimizer_state_dict'])
            load_successful = True
        except Exception as e:
            load_successful = False
            error_msg = str(e)

        print(f"\n✓ Load Old Checkpoint: {'PASS' if load_successful else 'FAIL'}")
        if load_successful:
            print(f"  Loaded version: {old_checkpoint['version']}")
            print(f"  Metadata: {old_checkpoint['metadata']}")

            # Verify model works after loading
            x = torch.randn(4, 10)
            with torch.no_grad():
                output = new_model(x)

            print(f"  Model functional: ✓")
            print(f"  Output shape: {output.shape}")
        else:
            print(f"  Error: {error_msg}")

        assert load_successful, "Failed to load old checkpoint"

    def test_checkpoint_metadata_preservation(self, checkpoint_manager, test_model):
        """
        Test that metadata is preserved correctly.

        Purpose: Ensure training metadata survives save/load
        """
        optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)

        # Rich metadata
        metadata = {
            'epoch': 42,
            'global_step': 10000,
            'best_loss': 0.123,
            'training_config': {
                'batch_size': 32,
                'learning_rate': 0.01,
                'optimizer': 'sgd'
            },
            'metrics': {
                'train_loss': 0.5,
                'val_loss': 0.6,
                'train_acc': 0.85,
                'val_acc': 0.82
            }
        }

        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_versioned_checkpoint(
            '2.0',
            test_model,
            optimizer,
            metadata
        )

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        loaded_metadata = loaded_checkpoint['metadata']

        # Verify all metadata preserved
        def compare_dicts(d1, d2, path=""):
            for key in d1:
                current_path = f"{path}.{key}" if path else key
                assert key in d2, f"Key {current_path} missing in loaded metadata"

                if isinstance(d1[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                else:
                    assert d1[key] == d2[key], \
                        f"Value mismatch at {current_path}: {d1[key]} vs {d2[key]}"

        compare_dicts(metadata, loaded_metadata)

        print(f"\n✓ Metadata Preservation: PASS")
        print(f"  All {len(metadata)} top-level keys preserved")
        print(f"  Training config: {loaded_metadata['training_config']}")
        print(f"  Metrics: {loaded_metadata['metrics']}")

    def test_optimizer_state_compatibility(self, checkpoint_manager):
        """
        Test optimizer state compatibility across different optimizers.

        Purpose: Ensure optimizer state can be loaded correctly
        """
        model1 = TestModel()
        torch.manual_seed(42)

        # Train with Adam
        optimizer_adam = torch.optim.Adam(model1.parameters(), lr=0.001)

        for _ in range(10):
            x = torch.randn(8, 10)
            target = torch.randint(0, 5, (8,))
            loss = nn.functional.cross_entropy(model1(x), target)
            loss.backward()
            optimizer_adam.step()
            optimizer_adam.zero_grad()

        # Save checkpoint
        checkpoint_manager.save_versioned_checkpoint(
            '3.0_adam',
            model1,
            optimizer_adam,
            {'optimizer': 'adam'}
        )

        # Load into new model with same optimizer type
        model2 = TestModel()
        optimizer_adam2 = torch.optim.Adam(model2.parameters(), lr=0.001)

        checkpoint = checkpoint_manager.load_versioned_checkpoint('3.0_adam')
        model2.load_state_dict(checkpoint['model_state_dict'])

        try:
            optimizer_adam2.load_state_dict(checkpoint['optimizer_state_dict'])
            load_successful = True
        except Exception as e:
            load_successful = False
            error_msg = str(e)

        print(f"\n✓ Optimizer State Compatibility: {'PASS' if load_successful else 'FAIL'}")

        if load_successful:
            # Verify optimizer state was actually loaded
            state = optimizer_adam2.state_dict()
            assert len(state['state']) > 0, "Optimizer state is empty"
            print(f"  Loaded optimizer state for {len(state['state'])} param groups")
            print(f"  Optimizer type: {checkpoint['metadata']['optimizer']}")
        else:
            print(f"  Error: {error_msg}")

        assert load_successful, "Failed to load optimizer state"

    def test_checkpoint_size_regression(self, checkpoint_manager, test_model):
        """
        Test that checkpoint sizes remain reasonable.

        Purpose: Detect checkpoint bloat
        """
        optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)

        # Train briefly
        for _ in range(3):
            x = torch.randn(4, 10)
            target = torch.randint(0, 5, (4,))
            loss = nn.functional.cross_entropy(test_model(x), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_versioned_checkpoint(
            '4.0',
            test_model,
            optimizer,
            {'test': 'size_check'}
        )

        # Get checkpoint size
        checkpoint_size_bytes = checkpoint_path.stat().st_size
        checkpoint_size_kb = checkpoint_size_bytes / 1024

        # Calculate expected size (rough estimate)
        num_params = sum(p.numel() for p in test_model.parameters())
        expected_size_kb = (num_params * 4) / 1024  # 4 bytes per float32 param

        # Adam has 2 extra states per param (momentum and variance)
        expected_with_optimizer_kb = expected_size_kb * 3  # model + 2 states

        size_ratio = checkpoint_size_kb / expected_with_optimizer_kb

        print(f"\n✓ Checkpoint Size:")
        print(f"  Actual size: {checkpoint_size_kb:.2f} KB")
        print(f"  Expected size: {expected_with_optimizer_kb:.2f} KB")
        print(f"  Size ratio: {size_ratio:.2f}x")
        print(f"  Number of parameters: {num_params}")

        # Checkpoint should be within 2x of expected size
        assert size_ratio < 2.0, \
            f"Checkpoint size bloated: {size_ratio:.2f}x expected size"

    def test_partial_checkpoint_loading(self, checkpoint_manager):
        """
        Test loading checkpoint with missing or extra keys.

        Purpose: Ensure graceful handling of model architecture changes
        """
        # Create and save model with specific architecture
        class OldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        old_model = OldModel()
        optimizer = torch.optim.SGD(old_model.parameters(), lr=0.01)

        checkpoint_manager.save_versioned_checkpoint(
            '5.0_old',
            old_model,
            optimizer,
            {'model': 'old_architecture'}
        )

        # Try to load into model with extra layer
        class NewModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 10)  # Changed dimension
                self.fc3 = nn.Linear(10, 5)   # New layer

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        new_model = NewModel()
        checkpoint = checkpoint_manager.load_versioned_checkpoint('5.0_old')

        # Load with strict=False to allow partial loading
        try:
            new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            partial_load_successful = True
            print(f"\n✓ Partial Checkpoint Loading: PASS")
            print(f"  Loaded checkpoint into modified architecture")
        except Exception as e:
            partial_load_successful = False
            print(f"\n✗ Partial Checkpoint Loading: FAIL")
            print(f"  Error: {str(e)}")

        # This should succeed with strict=False
        assert partial_load_successful, "Partial loading failed"


@pytest.mark.regression
def test_checkpoint_format_documentation(checkpoint_manager):
    """
    Document the current checkpoint format.

    Purpose: Maintain documentation of checkpoint structure
    """
    print("\n=== Checkpoint Format Documentation ===\n")
    print("Standard checkpoint format:")
    print("{")
    print("  'version': str,              # Checkpoint format version")
    print("  'model_state_dict': dict,    # Model parameters")
    print("  'optimizer_state_dict': dict,# Optimizer state")
    print("  'metadata': {                # Training metadata")
    print("    'epoch': int,              # Current epoch")
    print("    'global_step': int,        # Global training step")
    print("    'best_loss': float,        # Best validation loss")
    print("    'training_config': dict,   # Training hyperparameters")
    print("    'metrics': dict            # Current metrics")
    print("  }")
    print("}")
    print("\nRequired keys: version, model_state_dict, optimizer_state_dict")
    print("Optional keys: metadata (but recommended)")

    # Get info about existing checkpoints
    for version in ['1.0', '2.0', '3.0_adam', '4.0', '5.0_old']:
        info = checkpoint_manager.get_checkpoint_info(version)
        if info:
            print(f"\n✓ Checkpoint v{version}:")
            print(f"  Keys: {info['keys']}")
            print(f"  Model params: {len(info['model_keys'])}")
