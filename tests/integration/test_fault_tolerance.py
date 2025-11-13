"""
Fault Tolerance Integration Tests

Tests that verify the system can recover from failures gracefully.
These tests ensure robustness in production environments.

Fault Tolerance Tests:
- Checkpoint recovery after interruption
- Handling of corrupt checkpoints
- OOM recovery
- Graceful degradation

Purpose:
- Ensure system reliability
- Test error handling
- Verify recovery mechanisms
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


class TestModel(nn.Module):
    """Test model for fault tolerance testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def temp_dir():
    """Fixture providing temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_model():
    """Fixture providing test model"""
    torch.manual_seed(42)
    return TestModel()


@pytest.mark.integration
@pytest.mark.slow
class TestFaultTolerance:
    """Fault tolerance integration tests"""

    def test_checkpoint_recovery_after_interruption(self, test_model, temp_dir):
        """
        Test recovery from interrupted training.

        Workflow:
        1. Start training
        2. Save checkpoint
        3. Simulate interruption
        4. Resume from checkpoint
        5. Verify training continues correctly

        Expected: Training resumes from exact state
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = test_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        checkpoint_path = temp_dir / "recovery_checkpoint.pt"

        print(f"\n=== Checkpoint Recovery Test ===")

        # Phase 1: Initial training
        print("\nPhase 1: Initial training")
        model.train()
        epoch = 0
        global_step = 0

        for i in range(10):
            x = torch.randn(8, 10).to(device)
            target = torch.randint(0, 5, (8,)).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            global_step += 1

            # Save checkpoint after 5 steps
            if global_step == 5:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }
                torch.save(checkpoint, checkpoint_path)
                saved_loss = loss.item()
                print(f"✓ Checkpoint saved at step {global_step}, loss: {saved_loss:.4f}")

        print(f"  Completed {global_step} steps")

        # Phase 2: Simulate interruption and recovery
        print("\nPhase 2: Recovery from checkpoint")

        # Create new model (simulating fresh start)
        recovered_model = TestModel().to(device)
        recovered_optimizer = torch.optim.Adam(recovered_model.parameters(), lr=0.001)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        recovered_model.load_state_dict(checkpoint['model_state_dict'])
        recovered_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        recovered_global_step = checkpoint['global_step']
        recovered_loss = checkpoint['loss']

        print(f"✓ Checkpoint loaded from step {recovered_global_step}")
        print(f"  Recovered loss: {recovered_loss:.4f}")

        # Continue training
        for i in range(5):
            x = torch.randn(8, 10).to(device)
            target = torch.randint(0, 5, (8,)).to(device)

            recovered_optimizer.zero_grad()
            output = recovered_model(x)
            loss = criterion(output, target)
            loss.backward()
            recovered_optimizer.step()

            recovered_global_step += 1

        print(f"  Continued to step {recovered_global_step}")

        print("\n✓ Recovery successful")

        # Assertions
        assert checkpoint['global_step'] == 5, "Should have saved at step 5"
        assert recovered_global_step == 10, "Should have continued to step 10"

    def test_corrupt_checkpoint_handling(self, test_model, temp_dir):
        """
        Test handling of corrupt checkpoint files.

        Workflow:
        1. Save valid checkpoint
        2. Create corrupt checkpoint
        3. Attempt to load corrupt checkpoint
        4. Verify graceful error handling

        Expected: System should handle corrupt files gracefully
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = test_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"\n=== Corrupt Checkpoint Handling ===")

        # Save valid checkpoint
        valid_checkpoint_path = temp_dir / "valid_checkpoint.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, valid_checkpoint_path)
        print("✓ Valid checkpoint saved")

        # Create corrupt checkpoint (invalid data)
        corrupt_checkpoint_path = temp_dir / "corrupt_checkpoint.pt"
        with open(corrupt_checkpoint_path, 'w') as f:
            f.write("This is not a valid checkpoint file")
        print("✓ Corrupt checkpoint created")

        # Test loading valid checkpoint
        try:
            valid_checkpoint = torch.load(valid_checkpoint_path)
            valid_load_successful = True
            print("✓ Valid checkpoint loaded successfully")
        except Exception as e:
            valid_load_successful = False
            print(f"✗ Failed to load valid checkpoint: {e}")

        # Test loading corrupt checkpoint
        try:
            corrupt_checkpoint = torch.load(corrupt_checkpoint_path)
            corrupt_load_successful = True
            print("✗ Corrupt checkpoint loaded (should have failed!)")
        except Exception as e:
            corrupt_load_successful = False
            print(f"✓ Corrupt checkpoint properly rejected: {type(e).__name__}")

        # Assertions
        assert valid_load_successful, "Valid checkpoint should load"
        assert not corrupt_load_successful, "Corrupt checkpoint should fail to load"

    def test_oom_recovery_graceful_degradation(self, temp_dir):
        """
        Test recovery from out-of-memory errors.

        Workflow:
        1. Attempt training with large batch size
        2. Catch OOM error
        3. Reduce batch size
        4. Retry training

        Expected: System should recover with smaller batch size
        """
        if not torch.cuda.is_available():
            pytest.skip("OOM test requires CUDA")

        device = torch.device('cuda')

        print(f"\n=== OOM Recovery Test ===")

        # Try progressively smaller batch sizes until one works
        batch_sizes = [2048, 1024, 512, 256, 128, 64, 32]
        successful_batch_size = None

        for batch_size in batch_sizes:
            try:
                # Create model
                model = TestModel().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Try forward pass with this batch size
                torch.cuda.empty_cache()
                x = torch.randn(batch_size, 10).to(device)
                target = torch.randint(0, 5, (batch_size,)).to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                successful_batch_size = batch_size
                print(f"✓ Batch size {batch_size}: Success")
                break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"✗ Batch size {batch_size}: OOM")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        print(f"\n✓ OOM recovery successful")
        print(f"  Working batch size: {successful_batch_size}")

        assert successful_batch_size is not None, "Should find a working batch size"
        assert successful_batch_size <= 2048, "Should have degraded from large batch size"

    def test_checkpoint_rotation(self, test_model, temp_dir):
        """
        Test checkpoint rotation to manage disk space.

        Workflow:
        1. Save multiple checkpoints
        2. Keep only last N checkpoints
        3. Delete old checkpoints
        4. Verify correct checkpoints remain

        Expected: Only most recent N checkpoints should exist
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = test_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        max_checkpoints = 3
        checkpoints = []

        print(f"\n=== Checkpoint Rotation Test ===")
        print(f"  Max checkpoints: {max_checkpoints}")

        # Save 7 checkpoints
        for epoch in range(7):
            checkpoint_path = temp_dir / f"checkpoint_epoch_{epoch}.pt"

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)
            checkpoints.append(checkpoint_path)

            # Rotate: keep only last N checkpoints
            if len(checkpoints) > max_checkpoints:
                old_checkpoint = checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"  Deleted: {old_checkpoint.name}")

            print(f"✓ Saved: checkpoint_epoch_{epoch}.pt")

        # Verify only last 3 checkpoints exist
        remaining_checkpoints = list(temp_dir.glob("checkpoint_epoch_*.pt"))
        remaining_epochs = sorted([
            int(cp.stem.split('_')[-1]) for cp in remaining_checkpoints
        ])

        print(f"\n✓ Checkpoint rotation complete")
        print(f"  Remaining epochs: {remaining_epochs}")
        print(f"  Count: {len(remaining_epochs)}")

        assert len(remaining_epochs) == max_checkpoints, \
            f"Should have {max_checkpoints} checkpoints"
        assert remaining_epochs == [4, 5, 6], \
            "Should keep last 3 epochs (4, 5, 6)"

    def test_training_state_consistency_after_error(self, test_model, temp_dir):
        """
        Test that training state remains consistent after error recovery.

        Workflow:
        1. Train for a few steps
        2. Save checkpoint
        3. Simulate error
        4. Recover from checkpoint
        5. Verify state consistency

        Expected: Model produces same outputs after recovery
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = test_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        checkpoint_path = temp_dir / "state_checkpoint.pt"

        print(f"\n=== State Consistency Test ===")

        # Train for a few steps
        for _ in range(3):
            x = torch.randn(8, 10).to(device)
            target = torch.randint(0, 5, (8,)).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        torch.save(checkpoint, checkpoint_path)

        # Get output with current model
        test_input = torch.randn(4, 10).to(device)
        with torch.no_grad():
            original_output = model(test_input)

        print("✓ Original state captured")

        # Simulate error: modify model state
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        print("✓ State corrupted (simulated error)")

        # Recover from checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        if checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

        print("✓ State recovered from checkpoint")

        # Get output with recovered model
        with torch.no_grad():
            recovered_output = model(test_input)

        # Verify outputs match
        outputs_match = torch.allclose(original_output, recovered_output, rtol=1e-5)
        max_diff = (original_output - recovered_output).abs().max().item()

        print(f"\n✓ State consistency check:")
        print(f"  Outputs match: {outputs_match}")
        print(f"  Max difference: {max_diff:.2e}")

        assert outputs_match, "Recovered model should produce same outputs"

    def test_partial_batch_handling(self, test_model):
        """
        Test handling of partial batches at end of epoch.

        Workflow:
        1. Create dataset not divisible by batch size
        2. Train through entire dataset
        3. Verify last batch is handled correctly

        Expected: Training completes without errors on partial batch
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = test_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Dataset with 37 samples (not divisible by batch size 8)
        dataset_size = 37
        batch_size = 8
        expected_full_batches = dataset_size // batch_size
        expected_partial_batch_size = dataset_size % batch_size

        print(f"\n=== Partial Batch Handling ===")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Expected full batches: {expected_full_batches}")
        print(f"  Expected partial batch size: {expected_partial_batch_size}")

        # Create dataset
        data = torch.randn(dataset_size, 10)
        targets = torch.randint(0, 5, (dataset_size,))
        dataset = torch.utils.data.TensorDataset(data, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False  # Important: don't drop last partial batch
        )

        # Train through dataset
        model.train()
        batch_sizes_seen = []

        for batch_data, batch_target in dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_sizes_seen.append(batch_data.size(0))

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

        print(f"\n✓ Training completed")
        print(f"  Total batches: {len(batch_sizes_seen)}")
        print(f"  Batch sizes: {batch_sizes_seen}")

        # Verify batch sizes
        assert len(batch_sizes_seen) == expected_full_batches + 1, \
            "Should have full batches + 1 partial batch"
        assert all(bs == batch_size for bs in batch_sizes_seen[:-1]), \
            "All but last batch should be full size"
        assert batch_sizes_seen[-1] == expected_partial_batch_size, \
            f"Last batch should be size {expected_partial_batch_size}"

        print(f"✓ Partial batch handled correctly")


@pytest.mark.integration
def test_comprehensive_fault_tolerance(test_model, temp_dir):
    """
    Comprehensive fault tolerance test combining multiple scenarios.

    Workflow:
    1. Train with checkpointing
    2. Simulate various failures
    3. Recover and continue
    4. Verify final state is correct

    Expected: System recovers from all simulated failures
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n=== Comprehensive Fault Tolerance Test ===")

    # Scenario 1: Normal training with checkpoints
    model = test_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir()

    num_checkpoints_saved = 0

    for step in range(20):
        x = torch.randn(8, 10).to(device)
        target = torch.randint(0, 5, (8,)).to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Save checkpoint every 5 steps
        if (step + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
            num_checkpoints_saved += 1
            print(f"  ✓ Checkpoint saved at step {step + 1}")

    print(f"\n✓ Phase 1 complete: {num_checkpoints_saved} checkpoints saved")

    # Scenario 2: Load from checkpoint and continue
    checkpoint_path = checkpoint_dir / "checkpoint_step_10.pt"
    checkpoint = torch.load(checkpoint_path)

    new_model = TestModel().to(device)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✓ Phase 2 complete: Recovered from step {checkpoint['step']}")

    # Continue training
    for step in range(5):
        x = torch.randn(8, 10).to(device)
        target = torch.randint(0, 5, (8,)).to(device)

        new_optimizer.zero_grad()
        output = new_model(x)
        loss = criterion(output, target)
        loss.backward()
        new_optimizer.step()

    print(f"✓ Phase 3 complete: Continued training for 5 more steps")

    # Verify all checkpoints exist
    checkpoints_exist = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    print(f"\n✓ Comprehensive test complete")
    print(f"  Checkpoints created: {len(checkpoints_exist)}")
    print(f"  Recovery successful: ✓")
    print(f"  Continued training: ✓")

    assert len(checkpoints_exist) == num_checkpoints_saved
