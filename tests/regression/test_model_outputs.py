"""
Model Output Regression Tests

Tests that ensure model outputs remain consistent across code changes.
These tests detect unintended changes in model behavior.

Regression Tests:
- Forward pass output consistency
- Gradient computation consistency
- Checkpoint compatibility
- Deterministic behavior verification

Purpose:
- Catch breaking changes early
- Ensure reproducibility
- Maintain backward compatibility
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for regression testing"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RegressionTester:
    """Helper class for regression testing"""

    def __init__(self, baseline_dir='test_results/baselines'):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, name: str, data: dict):
        """Save baseline data for future regression tests"""
        filepath = self.baseline_dir / f"{name}.json"

        # Convert numpy/torch arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                serializable_data[key] = value.detach().cpu().numpy().tolist()
            else:
                serializable_data[key] = value

        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def load_baseline(self, name: str) -> dict:
        """Load baseline data"""
        filepath = self.baseline_dir / f"{name}.json"
        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def compare_outputs(
        self,
        current: torch.Tensor,
        baseline: list,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """Compare current output with baseline"""
        baseline_tensor = torch.tensor(baseline)
        return torch.allclose(current.cpu(), baseline_tensor, rtol=rtol, atol=atol)


@pytest.fixture
def regression_tester():
    """Fixture providing regression tester"""
    return RegressionTester()


@pytest.fixture
def simple_model():
    """Fixture providing simple model with fixed seed"""
    torch.manual_seed(42)
    return SimpleModel()


@pytest.mark.regression
class TestModelOutputs:
    """Model output regression tests"""

    def test_forward_pass_consistency(self, regression_tester, simple_model):
        """
        Test that forward pass produces consistent outputs.

        Purpose: Detect unintended changes in forward pass logic
        """
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Fixed input
        x = torch.randn(4, 10)

        # Get current output
        simple_model.eval()
        with torch.no_grad():
            output = simple_model(x)

        baseline_name = 'forward_pass_simple_model'
        baseline = regression_tester.load_baseline(baseline_name)

        if baseline is None:
            # First run - save baseline
            regression_tester.save_baseline(baseline_name, {
                'output': output,
                'input_shape': list(x.shape),
                'output_shape': list(output.shape)
            })
            print(f"\n✓ Baseline saved: {baseline_name}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output mean: {output.mean().item():.6f}")
        else:
            # Compare with baseline
            is_consistent = regression_tester.compare_outputs(
                output,
                baseline['output']
            )

            print(f"\n✓ Forward Pass Consistency: {'PASS' if is_consistent else 'FAIL'}")
            print(f"  Expected shape: {baseline['output_shape']}")
            print(f"  Actual shape: {list(output.shape)}")

            if not is_consistent:
                current_mean = output.mean().item()
                baseline_mean = torch.tensor(baseline['output']).mean().item()
                print(f"  ⚠ Output mismatch!")
                print(f"    Current mean: {current_mean:.6f}")
                print(f"    Baseline mean: {baseline_mean:.6f}")
                print(f"    Difference: {abs(current_mean - baseline_mean):.6e}")

            assert is_consistent, "Forward pass output changed from baseline"

    def test_gradient_consistency(self, regression_tester, simple_model):
        """
        Test that gradients remain consistent.

        Purpose: Ensure backward pass hasn't changed
        """
        torch.manual_seed(42)

        # Fixed input and target
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 5, (4,))

        # Forward and backward
        output = simple_model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        # Get gradients
        fc1_weight_grad = simple_model.fc1.weight.grad.clone()
        fc1_bias_grad = simple_model.fc1.bias.grad.clone()

        baseline_name = 'gradient_simple_model'
        baseline = regression_tester.load_baseline(baseline_name)

        if baseline is None:
            # Save baseline
            regression_tester.save_baseline(baseline_name, {
                'fc1_weight_grad': fc1_weight_grad,
                'fc1_bias_grad': fc1_bias_grad,
                'loss': loss.item()
            })
            print(f"\n✓ Gradient baseline saved")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  FC1 weight grad norm: {fc1_weight_grad.norm().item():.6f}")
        else:
            # Compare gradients
            weight_consistent = regression_tester.compare_outputs(
                fc1_weight_grad,
                baseline['fc1_weight_grad']
            )
            bias_consistent = regression_tester.compare_outputs(
                fc1_bias_grad,
                baseline['fc1_bias_grad']
            )

            is_consistent = weight_consistent and bias_consistent

            print(f"\n✓ Gradient Consistency: {'PASS' if is_consistent else 'FAIL'}")
            print(f"  Weight gradients: {'✓' if weight_consistent else '✗'}")
            print(f"  Bias gradients: {'✓' if bias_consistent else '✗'}")
            print(f"  Current loss: {loss.item():.6f}")
            print(f"  Baseline loss: {baseline['loss']:.6f}")

            assert is_consistent, "Gradients changed from baseline"

    def test_deterministic_behavior(self, regression_tester):
        """
        Test that model behavior is deterministic with same seed.

        Purpose: Ensure reproducibility
        """
        def run_training_step(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            x = torch.randn(8, 10)
            target = torch.randint(0, 5, (8,))

            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            return {
                'loss': loss.item(),
                'fc1_weight': model.fc1.weight.data.clone(),
                'fc1_bias': model.fc1.bias.data.clone()
            }

        # Run twice with same seed
        seed = 123
        result1 = run_training_step(seed)
        result2 = run_training_step(seed)

        # Compare results
        loss_match = abs(result1['loss'] - result2['loss']) < 1e-6
        weight_match = torch.allclose(result1['fc1_weight'], result2['fc1_weight'])
        bias_match = torch.allclose(result1['fc1_bias'], result2['fc1_bias'])

        is_deterministic = loss_match and weight_match and bias_match

        print(f"\n✓ Deterministic Behavior: {'PASS' if is_deterministic else 'FAIL'}")
        print(f"  Loss match: {'✓' if loss_match else '✗'}")
        print(f"  Weight match: {'✓' if weight_match else '✗'}")
        print(f"  Bias match: {'✓' if bias_match else '✗'}")

        if not is_deterministic:
            print(f"  ⚠ Non-deterministic behavior detected!")
            print(f"    Run 1 loss: {result1['loss']:.8f}")
            print(f"    Run 2 loss: {result2['loss']:.8f}")

        assert is_deterministic, "Behavior is not deterministic with same seed"

    def test_model_output_range(self, simple_model):
        """
        Test that model outputs stay within expected ranges.

        Purpose: Catch numerical instabilities
        """
        torch.manual_seed(42)

        # Test with various inputs
        test_cases = [
            ('normal', torch.randn(10, 10)),
            ('zeros', torch.zeros(10, 10)),
            ('ones', torch.ones(10, 10)),
            ('large', torch.randn(10, 10) * 10),
            ('small', torch.randn(10, 10) * 0.1)
        ]

        results = {}

        for name, x in test_cases:
            simple_model.eval()
            with torch.no_grad():
                output = simple_model(x)

            # Check for NaN or Inf
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_range = (output.min().item(), output.max().item())
            output_mean = output.mean().item()
            output_std = output.std().item()

            results[name] = {
                'has_nan': has_nan,
                'has_inf': has_inf,
                'range': output_range,
                'mean': output_mean,
                'std': output_std
            }

            print(f"\n✓ Input: {name}")
            print(f"  NaN: {'✗ FAIL' if has_nan else '✓ OK'}")
            print(f"  Inf: {'✗ FAIL' if has_inf else '✓ OK'}")
            print(f"  Range: [{output_range[0]:.2f}, {output_range[1]:.2f}]")
            print(f"  Mean: {output_mean:.4f}, Std: {output_std:.4f}")

            # Assertions
            assert not has_nan, f"NaN detected in output for {name} input"
            assert not has_inf, f"Inf detected in output for {name} input"

            # Check reasonable output range (adjusted for logits)
            assert -100 < output_range[0] < 100, f"Output range unreasonable for {name}"
            assert -100 < output_range[1] < 100, f"Output range unreasonable for {name}"

    def test_batch_size_independence(self, simple_model):
        """
        Test that per-sample outputs are independent of batch size.

        Purpose: Ensure batch processing doesn't affect individual samples
        """
        torch.manual_seed(42)

        # Create test samples
        x1 = torch.randn(1, 10)
        x2 = torch.randn(1, 10)
        x_batch = torch.cat([x1, x2], dim=0)

        simple_model.eval()
        with torch.no_grad():
            # Process individually
            out1 = simple_model(x1)
            out2 = simple_model(x2)

            # Process as batch
            out_batch = simple_model(x_batch)

        # Compare
        match1 = torch.allclose(out1, out_batch[0:1], rtol=1e-5, atol=1e-7)
        match2 = torch.allclose(out2, out_batch[1:2], rtol=1e-5, atol=1e-7)

        print(f"\n✓ Batch Size Independence: {'PASS' if match1 and match2 else 'FAIL'}")
        print(f"  Sample 1 match: {'✓' if match1 else '✗'}")
        print(f"  Sample 2 match: {'✓' if match2 else '✗'}")

        if not match1:
            print(f"  Sample 1 max diff: {(out1 - out_batch[0:1]).abs().max().item():.2e}")
        if not match2:
            print(f"  Sample 2 max diff: {(out2 - out_batch[1:2]).abs().max().item():.2e}")

        assert match1 and match2, "Batch processing affects individual samples"


@pytest.mark.regression
def test_create_initial_baselines(regression_tester, simple_model):
    """
    Helper test to create initial baselines.

    Run this once to establish baselines, then run other tests to verify.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Create baselines for all regression tests
    print("\n=== Creating Regression Test Baselines ===")

    # Forward pass baseline
    x = torch.randn(4, 10)
    simple_model.eval()
    with torch.no_grad():
        output = simple_model(x)

    regression_tester.save_baseline('forward_pass_simple_model', {
        'output': output,
        'input_shape': list(x.shape),
        'output_shape': list(output.shape)
    })
    print("✓ Forward pass baseline created")

    # Gradient baseline
    torch.manual_seed(42)
    x = torch.randn(4, 10, requires_grad=True)
    target = torch.randint(0, 5, (4,))
    simple_model_new = SimpleModel()  # Fresh model
    torch.manual_seed(42)  # Reset for consistent weights
    simple_model_new = SimpleModel()

    output = simple_model_new(x)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()

    regression_tester.save_baseline('gradient_simple_model', {
        'fc1_weight_grad': simple_model_new.fc1.weight.grad,
        'fc1_bias_grad': simple_model_new.fc1.bias.grad,
        'loss': loss.item()
    })
    print("✓ Gradient baseline created")

    print("\n✓ All baselines created successfully")
