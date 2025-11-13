"""
Tests for torch.compile Integration

This module tests the torch.compile integration with distributed training,
including compilation correctness, performance measurement, and compatibility
with different distributed strategies.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from distributed_training.advanced.compile_trainer import (
    compile_model,
    CompileConfig,
    CompiledTrainerMixin,
    CompilationProfiler,
    get_compile_preset,
    COMPILE_PRESETS
)


class SimpleModel(nn.Module):
    """Simple model for compilation testing"""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    """Fixture providing simple model"""
    return SimpleModel()


@pytest.fixture
def compile_config():
    """Fixture providing default compile config"""
    return CompileConfig()


@pytest.mark.unit
class TestCompileConfig:
    """Tests for CompileConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CompileConfig()

        assert config.backend == "inductor"
        assert config.mode == "default"
        assert config.fullgraph == False
        assert config.dynamic is None
        assert config.options is None
        assert config.disable == False

    def test_custom_config(self):
        """Test custom configuration"""
        config = CompileConfig(
            backend="cudagraphs",
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False
        )

        assert config.backend == "cudagraphs"
        assert config.mode == "reduce-overhead"
        assert config.fullgraph == True
        assert config.dynamic == False

    def test_disabled_config(self):
        """Test disabled compilation"""
        config = CompileConfig(disable=True)

        assert config.disable == True


@pytest.mark.unit
class TestCompileModel:
    """Tests for compile_model function"""

    def test_basic_compilation(self, simple_model):
        """Test basic model compilation"""
        compiled_model = compile_model(simple_model)

        # Check model is returned (may or may not be compiled depending on PyTorch version)
        assert compiled_model is not None
        assert isinstance(compiled_model, nn.Module)

    def test_compilation_with_config(self, simple_model):
        """Test compilation with custom config"""
        config = CompileConfig(mode="default")
        compiled_model = compile_model(simple_model, config)

        assert compiled_model is not None

    def test_compilation_disabled(self, simple_model):
        """Test that disabled compilation returns original model"""
        config = CompileConfig(disable=True)
        compiled_model = compile_model(simple_model, config)

        # Should return original model unchanged
        assert compiled_model is simple_model

    def test_compiled_model_forward(self, simple_model):
        """Test that compiled model can perform forward pass"""
        compiled_model = compile_model(simple_model)

        x = torch.randn(4, 10)
        output = compiled_model(x)

        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_compiled_model_backward(self, simple_model):
        """Test that compiled model can perform backward pass"""
        compiled_model = compile_model(simple_model)

        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randint(0, 5, (4,))

        output = compiled_model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in compiled_model.parameters():
            assert param.grad is not None

    def test_compilation_with_kwargs(self, simple_model):
        """Test compilation with additional kwargs"""
        compiled_model = compile_model(
            simple_model,
            backend="inductor",
            mode="default"
        )

        assert compiled_model is not None


@pytest.mark.unit
class TestCompilePresets:
    """Tests for compilation presets"""

    def test_preset_development(self):
        """Test development preset"""
        preset = get_compile_preset('development')

        assert preset.backend == 'inductor'
        assert preset.mode == 'default'
        assert preset.dynamic == True

    def test_preset_production(self):
        """Test production preset"""
        preset = get_compile_preset('production')

        assert preset.backend == 'inductor'
        assert preset.mode == 'max-autotune'
        assert preset.dynamic == False

    def test_preset_low_latency(self):
        """Test low latency preset"""
        preset = get_compile_preset('low_latency')

        assert preset.backend == 'cudagraphs'
        assert preset.mode == 'reduce-overhead'
        assert preset.fullgraph == True

    def test_preset_debug(self):
        """Test debug preset"""
        preset = get_compile_preset('debug')

        assert preset.backend == 'aot_eager'
        assert preset.dynamic == True

    def test_preset_disabled(self):
        """Test disabled preset"""
        preset = get_compile_preset('disabled')

        assert preset.disable == True

    def test_invalid_preset(self):
        """Test that invalid preset raises error"""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_compile_preset('invalid_preset')

    def test_all_presets_defined(self):
        """Test that all expected presets are defined"""
        expected_presets = [
            'development',
            'production',
            'low_latency',
            'debug',
            'disabled'
        ]

        for preset_name in expected_presets:
            preset = get_compile_preset(preset_name)
            assert isinstance(preset, CompileConfig)


@pytest.mark.unit
class TestCompiledTrainerMixin:
    """Tests for CompiledTrainerMixin"""

    def test_setup_compilation_enabled(self, simple_model):
        """Test compilation setup when enabled"""
        mixin = CompiledTrainerMixin()
        config = CompileConfig()

        compiled_model = mixin.setup_compilation(simple_model, config)

        assert compiled_model is not None

    def test_setup_compilation_disabled(self, simple_model):
        """Test compilation setup when disabled"""
        mixin = CompiledTrainerMixin()
        config = CompileConfig(disable=True)

        compiled_model = mixin.setup_compilation(simple_model, config)

        # Should return original model
        assert compiled_model is simple_model

    def test_setup_compilation_before_wrap(self, simple_model):
        """Test compilation before distributed wrapping"""
        mixin = CompiledTrainerMixin()
        config = CompileConfig()

        compiled_model = mixin.setup_compilation(
            simple_model,
            config,
            compile_before_wrap=True
        )

        assert compiled_model is not None

    def test_compile_after_wrap(self, simple_model):
        """Test compilation after distributed wrapping"""
        mixin = CompiledTrainerMixin()
        config = CompileConfig()

        # Simulate wrapped model (just use original for test)
        wrapped_model = simple_model

        compiled_model = mixin.compile_after_wrap(wrapped_model, config)

        assert compiled_model is not None


@pytest.mark.performance
class TestCompilationProfiler:
    """Tests for CompilationProfiler"""

    def test_profiler_basic(self, simple_model):
        """Test basic profiler functionality"""
        profiler = CompilationProfiler()

        x = torch.randn(32, 10)

        # Profile eager mode
        with profiler.profile('eager'):
            output = simple_model(x)

        # Check timing recorded
        assert 'eager' in profiler.times
        assert profiler.times['eager'] > 0

    def test_profiler_comparison(self, simple_model):
        """Test profiler comparison between eager and compiled"""
        profiler = CompilationProfiler()

        x = torch.randn(32, 10)

        # Profile eager
        with profiler.profile('eager'):
            for _ in range(10):
                output = simple_model(x)

        # Compile and profile
        compiled_model = compile_model(simple_model)

        with profiler.profile('compiled'):
            for _ in range(10):
                output = compiled_model(x)

        # Get results
        results = profiler.get_results()

        assert 'times' in results
        assert 'eager' in results['times']
        assert 'compiled' in results['times']

    def test_profiler_results(self, simple_model):
        """Test profiler results calculation"""
        profiler = CompilationProfiler()

        x = torch.randn(32, 10)

        # Profile both modes
        with profiler.profile('eager'):
            output = simple_model(x)

        compiled_model = compile_model(simple_model)

        with profiler.profile('compiled'):
            output = compiled_model(x)

        results = profiler.get_results()

        # Check speedup calculated
        if 'speedup' in results:
            assert results['speedup'] > 0
            assert 'time_saved_ms' in results


@pytest.mark.integration
class TestCompilationIntegration:
    """Integration tests for torch.compile"""

    def test_compilation_with_training_loop(self, simple_model):
        """Test compilation in a training loop"""
        compiled_model = compile_model(simple_model)
        optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        compiled_model.train()
        for _ in range(5):
            x = torch.randn(8, 10)
            target = torch.randint(0, 5, (8,))

            optimizer.zero_grad()
            output = compiled_model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Should complete without errors
        assert True

    def test_compilation_model_save_load(self, simple_model, tmp_path):
        """Test saving and loading compiled model"""
        compiled_model = compile_model(simple_model)

        # Forward pass
        x = torch.randn(4, 10)
        output_before = compiled_model(x)

        # Save state dict
        save_path = tmp_path / "compiled_model.pt"
        torch.save(compiled_model.state_dict(), save_path)

        # Load into new model
        new_model = SimpleModel()
        new_model.load_state_dict(torch.load(save_path))

        # Compare outputs
        new_model.eval()
        compiled_model.eval()
        with torch.no_grad():
            output_after = new_model(x)

        # Outputs should match
        assert torch.allclose(output_before, output_after, rtol=1e-4)

    def test_compilation_deterministic(self, simple_model):
        """Test that compiled model is deterministic"""
        torch.manual_seed(42)
        compiled_model = compile_model(simple_model)

        x = torch.randn(4, 10)

        # Multiple forward passes with same input
        outputs = []
        for _ in range(3):
            compiled_model.eval()
            with torch.no_grad():
                output = compiled_model(x)
                outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])


@pytest.mark.unit
def test_compilation_documentation():
    """Test that all functions have documentation"""
    from distributed_training.advanced import compile_trainer

    # Check module docstring
    assert compile_trainer.__doc__ is not None

    # Check class docstrings
    assert CompileConfig.__doc__ is not None
    assert CompiledTrainerMixin.__doc__ is not None
    assert CompilationProfiler.__doc__ is not None

    # Check function docstrings
    assert compile_model.__doc__ is not None
    assert get_compile_preset.__doc__ is not None

    print("\n✓ All compile_trainer components are documented")


@pytest.mark.unit
def test_compile_example_usage():
    """Test example usage from documentation"""
    # This tests that the documented examples actually work

    # Example 1: Basic compilation
    model = SimpleModel()
    compiled_model = compile_model(model)
    assert compiled_model is not None

    # Example 2: With custom config
    config = CompileConfig(mode='default')
    compiled_model = compile_model(model, config)
    assert compiled_model is not None

    # Example 3: Using preset
    preset = get_compile_preset('development')
    compiled_model = compile_model(model, preset)
    assert compiled_model is not None

    print("\n✓ All documented examples work correctly")
