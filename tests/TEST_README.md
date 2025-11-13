# Comprehensive Test Suite Documentation

This document provides detailed information about the test suite for the Distributed Training Framework.

## Table of Contents

- [Overview](#overview)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Performance Tests](#performance-tests)
- [Regression Tests](#regression-tests)
- [Integration Tests](#integration-tests)
- [Test Markers](#test-markers)
- [Expected Outcomes](#expected-outcomes)
- [Troubleshooting](#troubleshooting)

## Overview

The test suite is designed to ensure the reliability, performance, and correctness of the distributed training framework. It includes:

- **160+ tests** covering all major components
- **Performance benchmarks** for throughput, memory, and scalability
- **Regression tests** to prevent breaking changes
- **Integration tests** for end-to-end workflows
- **Fault tolerance tests** for production reliability

### Test Coverage

```
Total Tests: 160+
├── Unit Tests: 60+ (existing)
├── Performance Tests: 30+
├── Regression Tests: 20+
└── Integration Tests: 50+

Code Coverage Target: >85%
```

## Test Categories

### 1. Unit Tests (tests/)
**Purpose**: Test individual components in isolation

**Location**: `tests/test_*.py`

**Coverage**:
- Data parallel training (DDP)
- Model parallel layers
- Checkpoint management
- Data loading and prefetching
- Performance profiling

**Run**:
```bash
pytest tests/test_*.py -m unit -v
```

### 2. Performance Tests (tests/performance/)
**Purpose**: Measure and benchmark system performance

**Location**: `tests/performance/`

**Tests**:
- `test_throughput.py` - Training throughput measurements
- `test_memory.py` - Memory usage profiling
- `test_scalability.py` - Multi-GPU scaling tests

**Key Metrics**:
- Samples per second
- Memory usage (MB)
- Scaling efficiency
- GPU utilization

**Run**:
```bash
pytest tests/performance/ -v
```

### 3. Regression Tests (tests/regression/)
**Purpose**: Ensure backward compatibility and prevent regressions

**Location**: `tests/regression/`

**Tests**:
- `test_model_outputs.py` - Model output consistency
- `test_checkpoint_compatibility.py` - Checkpoint format compatibility

**Features**:
- Baseline comparison
- Deterministic behavior verification
- Output range validation
- Checkpoint version management

**Run**:
```bash
pytest tests/regression/ -v
```

### 4. Integration Tests (tests/integration/)
**Purpose**: Test complete workflows and component interactions

**Location**: `tests/integration/`

**Tests**:
- `test_end_to_end_training.py` - Complete training workflows
- `test_fault_tolerance.py` - Failure recovery and robustness

**Scenarios**:
- Full training with validation
- Mixed precision training
- Gradient accumulation
- Early stopping
- Checkpoint recovery
- OOM handling

**Run**:
```bash
pytest tests/integration/ -v
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/distributed_training --cov-report=html

# Run specific test category
pytest tests/performance/ -v
pytest tests/regression/ -v
pytest tests/integration/ -v
```

### Running by Marker

```bash
# Unit tests only
pytest -m unit

# Performance tests (slow)
pytest -m performance

# Regression tests
pytest -m regression

# Integration tests
pytest -m integration

# Distributed tests (requires multiple GPUs)
pytest -m distributed

# Skip slow tests
pytest -m "not slow"
```

### Running Specific Tests

```bash
# Run specific test file
pytest tests/performance/test_throughput.py -v

# Run specific test class
pytest tests/performance/test_throughput.py::TestThroughput -v

# Run specific test method
pytest tests/performance/test_throughput.py::TestThroughput::test_baseline_throughput -v
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4

# Run with auto-detection of CPU count
pytest -n auto
```

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── TEST_README.md              # This file
├── conftest.py                 # Shared fixtures (optional)
│
├── test_*.py                   # Unit tests
├── test_data_parallel.py
├── test_model_parallel.py
├── test_checkpointing.py
├── test_profiling.py
├── test_data_loading.py
│
├── performance/                # Performance tests
│   ├── __init__.py
│   ├── test_throughput.py
│   ├── test_memory.py
│   └── test_scalability.py
│
├── regression/                 # Regression tests
│   ├── __init__.py
│   ├── test_model_outputs.py
│   └── test_checkpoint_compatibility.py
│
└── integration/                # Integration tests
    ├── __init__.py
    ├── test_end_to_end_training.py
    └── test_fault_tolerance.py
```

### Test Results

Test results are saved to:
```
test_results/
├── throughput_results.json     # Performance metrics
├── memory_results.json         # Memory profiles
├── scalability_results.json    # Scaling data
├── baselines/                  # Regression baselines
│   ├── forward_pass_simple_model.json
│   └── gradient_simple_model.json
└── checkpoint_versions/        # Checkpoint compatibility
    ├── checkpoint_v1.0.pt
    ├── checkpoint_v2.0.pt
    └── ...
```

## Performance Tests

### test_throughput.py

**Tests training throughput (samples/sec) for different configurations.**

#### Tests:

1. **test_baseline_throughput**
   - Measures baseline training speed
   - **Expected**: >100 samples/sec (CPU), >1000 samples/sec (GPU)
   - **Output**: Samples/sec, time per batch, peak memory

2. **test_batch_size_scaling**
   - Tests how throughput scales with batch size
   - **Expected**: Throughput increases with batch size
   - **Output**: Throughput for each batch size

3. **test_mixed_precision_speedup**
   - Measures FP16 vs FP32 performance
   - **Expected**: 1.5-3x speedup on modern GPUs
   - **Output**: FP32 speed, FP16 speed, speedup ratio

4. **test_gradient_accumulation_throughput**
   - Tests throughput with gradient accumulation
   - **Expected**: Similar samples/sec with larger effective batch
   - **Output**: Effective batch size, optimizer steps/sec

#### Example Output:

```
✓ Baseline Throughput: 1247.3 samples/sec
  Time per batch: 25.67 ms
  Peak memory: 892.1 MB

✓ FP32: 1247.3 samples/sec
✓ FP16: 2891.5 samples/sec
✓ Speedup: 2.32x
```

#### Usage:

```bash
# Run throughput tests
pytest tests/performance/test_throughput.py -v

# Save results
pytest tests/performance/test_throughput.py --save-results

# View results
cat test_results/throughput_results.json
```

### test_memory.py

**Tests memory usage and efficiency.**

#### Tests:

1. **test_baseline_memory_usage**
   - Measures base memory requirements
   - **Expected**: Memory scales with model size
   - **Output**: Parameter memory, activation memory, total peak

2. **test_batch_size_memory_scaling**
   - Tests memory scaling with batch size
   - **Expected**: Linear increase in activation memory
   - **Output**: Memory for each batch size

3. **test_gradient_checkpointing_memory_savings**
   - Measures memory reduction from checkpointing
   - **Expected**: 50-80% reduction in activation memory
   - **Output**: Memory with/without checkpointing, savings %

4. **test_optimizer_state_memory**
   - Compares optimizer memory usage
   - **Expected**: Adam uses ~3x more than SGD
   - **Output**: Memory for each optimizer type

5. **test_mixed_precision_memory_savings**
   - Measures FP16 memory savings
   - **Expected**: ~2x reduction in parameter memory
   - **Output**: FP32 vs FP16 memory, savings %

#### Example Output:

```
✓ Baseline Memory Usage:
  Parameters: 246.8 MB
  Forward pass: 128.4 MB
  Backward pass: 89.2 MB
  Total peak: 464.4 MB

✓ Memory Usage:
  Without checkpointing: 892.1 MB
  With checkpointing: 234.6 MB
  Savings: 73.7%
```

### test_scalability.py

**Tests scaling properties across multiple GPUs.**

#### Tests:

1. **test_data_parallel_weak_scaling**
   - Tests constant time with proportional data/GPUs
   - **Expected**: <20% time variance
   - **Output**: Time per epoch for each GPU count

2. **test_data_parallel_strong_scaling**
   - Tests speedup with fixed problem size
   - **Expected**: >70% efficiency at 4 GPUs
   - **Output**: Throughput, scaling efficiency, speedup

3. **test_pipeline_parallel_efficiency**
   - Tests pipeline efficiency with micro-batches
   - **Expected**: >80% efficiency with 16 micro-batches
   - **Output**: Efficiency, bubble time

4. **test_communication_overhead_estimation**
   - Estimates communication costs
   - **Expected**: Overhead increases with model size and GPUs
   - **Output**: Communication time, compute time, overhead %

5. **test_batch_size_scaling_across_gpus**
   - Tests effective batch size scaling
   - **Expected**: Linear scaling
   - **Output**: Effective batch size per GPU count

#### Example Output:

```
✓ 1 GPU(s): Throughput: 100.0 samples/sec
✓ 2 GPU(s): Throughput: 190.5 samples/sec, Efficiency: 95.2%
✓ 4 GPU(s): Throughput: 362.0 samples/sec, Efficiency: 90.5%

✓ 16 micro-batches:
  Theoretical efficiency: 84.2%
  Bubble time: 15.8%
```

## Regression Tests

### test_model_outputs.py

**Ensures model outputs remain consistent across changes.**

#### Tests:

1. **test_forward_pass_consistency**
   - Compares outputs with saved baseline
   - **Purpose**: Detect unintended logic changes
   - **First run**: Saves baseline
   - **Subsequent runs**: Compares with baseline

2. **test_gradient_consistency**
   - Verifies gradient computation consistency
   - **Purpose**: Ensure backward pass unchanged
   - **Output**: Gradient match status

3. **test_deterministic_behavior**
   - Tests reproducibility with same seed
   - **Purpose**: Ensure determinism
   - **Output**: Loss match, weight match, bias match

4. **test_model_output_range**
   - Checks outputs for NaN/Inf
   - **Purpose**: Catch numerical instabilities
   - **Output**: NaN/Inf status, output range

5. **test_batch_size_independence**
   - Verifies batch processing doesn't affect individual samples
   - **Purpose**: Ensure correctness
   - **Output**: Per-sample match status

#### Example Output:

```
✓ Forward Pass Consistency: PASS
  Expected shape: [4, 5]
  Actual shape: [4, 5]

✓ Deterministic Behavior: PASS
  Loss match: ✓
  Weight match: ✓
  Bias match: ✓

✓ Input: normal
  NaN: ✓ OK
  Inf: ✓ OK
  Range: [-2.34, 3.45]
  Mean: 0.0012, Std: 0.9876
```

#### Creating Baselines:

```bash
# First time: create baselines
pytest tests/regression/test_model_outputs.py::test_create_initial_baselines -v

# Subsequent runs: verify against baselines
pytest tests/regression/test_model_outputs.py -v
```

### test_checkpoint_compatibility.py

**Ensures checkpoint format compatibility across versions.**

#### Tests:

1. **test_checkpoint_format_v1**
   - Establishes baseline checkpoint format
   - **Output**: Checkpoint structure

2. **test_load_old_checkpoint**
   - Tests loading older checkpoint versions
   - **Purpose**: Ensure backward compatibility
   - **Output**: Load success, version info

3. **test_checkpoint_metadata_preservation**
   - Verifies metadata survives save/load
   - **Purpose**: Ensure training info preserved
   - **Output**: Metadata comparison

4. **test_optimizer_state_compatibility**
   - Tests optimizer state save/load
   - **Purpose**: Ensure training can resume
   - **Output**: Optimizer state verification

5. **test_checkpoint_size_regression**
   - Monitors checkpoint file size
   - **Purpose**: Detect checkpoint bloat
   - **Output**: Size comparison

6. **test_partial_checkpoint_loading**
   - Tests loading with architecture changes
   - **Purpose**: Handle model evolution
   - **Output**: Partial load success

#### Example Output:

```
✓ Checkpoint v1.0 format:
  Keys: ['version', 'model_state_dict', 'optimizer_state_dict', 'metadata']
  Model params: 4
  Optimizer groups: 1
  Metadata: {'epoch': 1, 'loss': 1.5}

✓ Load Old Checkpoint: PASS
  Loaded version: 1.0
  Model functional: ✓

✓ Checkpoint Size:
  Actual size: 24.56 KB
  Expected size: 23.12 KB
  Size ratio: 1.06x
```

## Integration Tests

### test_end_to_end_training.py

**Tests complete training workflows.**

#### Tests:

1. **test_basic_training_loop**
   - Full training for multiple epochs
   - **Verifies**: Loss decreases, no errors
   - **Duration**: ~30 seconds

2. **test_training_with_validation**
   - Training + validation loop
   - **Verifies**: Both train and val metrics
   - **Output**: Train loss, val loss, val accuracy

3. **test_training_with_checkpointing**
   - Training with checkpoint save/load
   - **Verifies**: Checkpoint recovery works
   - **Output**: Checkpoint load success

4. **test_mixed_precision_training**
   - AMP training workflow
   - **Verifies**: Mixed precision works correctly
   - **Requires**: CUDA

5. **test_gradient_accumulation**
   - Training with gradient accumulation
   - **Verifies**: Effective batch size correct
   - **Output**: Optimizer steps count

6. **test_early_stopping**
   - Early stopping mechanism
   - **Verifies**: Stops when val loss plateaus
   - **Output**: Stopping epoch, best loss

7. **test_full_training_workflow**
   - Complete workflow with all features
   - **Verifies**: All components work together
   - **Duration**: ~1-2 minutes

#### Example Output:

```
=== Basic Training Loop ===
Epoch 1/3: Loss = 2.1234
Epoch 2/3: Loss = 1.5678
Epoch 3/3: Loss = 1.2345

✓ Training completed successfully
  Initial loss: 2.1234
  Final loss: 1.2345
  Loss reduction: 41.9%

=== Training with Validation ===
✓ Training metrics:
  Train loss: 1.2345
  Val loss: 1.3456
  Val accuracy: 67.50%

=== Mixed Precision Training ===
✓ Mixed precision training completed
  Average loss: 1.2345
  Gradient scaler scale: 65536.0
```

### test_fault_tolerance.py

**Tests system robustness and recovery.**

#### Tests:

1. **test_checkpoint_recovery_after_interruption**
   - Simulates training interruption and recovery
   - **Verifies**: Training resumes from exact state
   - **Output**: Recovery success

2. **test_corrupt_checkpoint_handling**
   - Tests handling of corrupt files
   - **Verifies**: Graceful error handling
   - **Output**: Valid load success, corrupt rejection

3. **test_oom_recovery_graceful_degradation**
   - Tests OOM recovery strategy
   - **Verifies**: Finds working batch size
   - **Output**: Successful batch size
   - **Requires**: CUDA

4. **test_checkpoint_rotation**
   - Tests checkpoint cleanup
   - **Verifies**: Only N most recent kept
   - **Output**: Remaining checkpoints

5. **test_training_state_consistency_after_error**
   - Tests state consistency after recovery
   - **Verifies**: Model produces same outputs
   - **Output**: Output match verification

6. **test_partial_batch_handling**
   - Tests handling of last partial batch
   - **Verifies**: No errors on partial batch
   - **Output**: Batch sizes processed

7. **test_comprehensive_fault_tolerance**
   - Combined failure scenarios
   - **Verifies**: System handles all failures
   - **Output**: Recovery success for all scenarios

#### Example Output:

```
=== Checkpoint Recovery Test ===
Phase 1: Initial training
✓ Checkpoint saved at step 5, loss: 1.8234

Phase 2: Recovery from checkpoint
✓ Checkpoint loaded from step 5
  Recovered loss: 1.8234
  Continued to step 10

✓ Recovery successful

=== OOM Recovery Test ===
✗ Batch size 2048: OOM
✗ Batch size 1024: OOM
✗ Batch size 512: OOM
✓ Batch size 256: Success

✓ OOM recovery successful
  Working batch size: 256

=== Checkpoint Rotation Test ===
  Max checkpoints: 3
  Deleted: checkpoint_epoch_0.pt
  Deleted: checkpoint_epoch_1.pt
  Deleted: checkpoint_epoch_2.pt
  Deleted: checkpoint_epoch_3.pt

✓ Checkpoint rotation complete
  Remaining epochs: [4, 5, 6]
  Count: 3
```

## Test Markers

Tests are organized using pytest markers for selective execution.

### Available Markers:

```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.regression    # Regression tests
@pytest.mark.performance   # Performance benchmarks
@pytest.mark.distributed   # Requires multiple GPUs
@pytest.mark.slow          # Long-running tests (>1 minute)
```

### Usage:

```bash
# Run only fast tests
pytest -m "not slow"

# Run unit and integration tests
pytest -m "unit or integration"

# Run everything except distributed tests
pytest -m "not distributed"

# Combine markers
pytest -m "performance and not slow"
```

## Expected Outcomes

### Performance Baselines

**Single GPU (V100):**
- DDP throughput: ~2000 samples/sec
- FP16 speedup: 2-3x over FP32
- Gradient checkpointing: 50-80% memory savings
- Activation memory: 30-40% of total

**Multi-GPU (4x V100):**
- DDP scaling efficiency: >90%
- Pipeline efficiency (16 micro-batches): >80%
- Communication overhead: <15%

### Regression Criteria

**Pass Criteria:**
- Forward pass outputs match baseline (rtol=1e-5)
- Gradients match baseline (rtol=1e-5)
- Deterministic with same seed
- No NaN or Inf in outputs
- Checkpoint format compatible
- Checkpoint size <2x expected

### Integration Success

**Pass Criteria:**
- Training completes without errors
- Loss decreases during training
- Validation accuracy > random baseline
- Checkpoints save and load correctly
- Mixed precision works on GPU
- Partial batches handled correctly
- System recovers from failures

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'distributed_training'`

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size in tests
# Or run on CPU
pytest --device=cpu

# Skip GPU-only tests
pytest -m "not distributed"
```

#### 3. Baseline Mismatch

**Problem**: `AssertionError: Forward pass output changed from baseline`

**Solution**:
```bash
# Regenerate baselines (only if intentional change)
pytest tests/regression/test_model_outputs.py::test_create_initial_baselines -v

# Or investigate the difference
# Check test output for details on mismatch
```

#### 4. Slow Tests

**Problem**: Tests take too long

**Solution**:
```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel
pytest -n auto

# Run specific fast tests
pytest tests/ -m unit -v
```

#### 5. Random Failures

**Problem**: Tests pass sometimes, fail other times

**Solution**:
- Check for proper seed setting
- Verify test isolation
- Check for race conditions
- Use `--verbose` for more details

### Getting Help

If tests fail:

1. **Check test output** - Look for specific error messages
2. **Run with -vv** - Get maximum verbosity
3. **Run single test** - Isolate the failing test
4. **Check logs** - Review test_results/ directory
5. **Verify environment** - Check CUDA, PyTorch versions

### Debugging Tests

```bash
# Maximum verbosity
pytest -vv

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show full traceback
pytest --tb=long

# Run specific test with debugging
pytest tests/performance/test_throughput.py::TestThroughput::test_baseline_throughput -vv -s
```

## CI/CD Integration

### GitHub Actions

Tests are automatically run on pull requests:

```yaml
# .github/workflows/tests.yml
- Unit tests on Python 3.8, 3.9, 3.10, 3.11
- Integration tests (CPU only)
- Performance tests (baseline only)
- Code coverage report
```

### Local Pre-commit

Run tests before committing:

```bash
# Run fast tests
pytest -m "not slow" --tb=short

# Run with coverage
pytest --cov=src/distributed_training --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src/distributed_training --cov-report=html
open htmlcov/index.html
```

## Continuous Improvement

### Adding New Tests

1. **Choose appropriate category** (unit/performance/regression/integration)
2. **Use existing fixtures** when possible
3. **Add clear docstrings** explaining purpose
4. **Include expected outcomes** in docstring
5. **Use appropriate markers** (@pytest.mark.*)
6. **Update this README** with new test info

### Performance Regression Tracking

Track performance over time:

```bash
# Run performance tests and save results
pytest tests/performance/ --save-results

# Compare with previous results
python scripts/compare_performance.py \
    test_results/throughput_results.json \
    previous_results/throughput_results.json
```

### Coverage Goals

```bash
# Generate coverage report
pytest --cov=src/distributed_training --cov-report=html

# View coverage
open htmlcov/index.html

# Target: >85% coverage
```

## Summary

This comprehensive test suite ensures:

✅ **Correctness** - All components work as intended
✅ **Performance** - System meets speed and memory targets
✅ **Reliability** - System recovers from failures
✅ **Compatibility** - Changes don't break existing functionality
✅ **Production-Ready** - System ready for real-world use

### Test Statistics

- **Total Tests**: 160+
- **Total Coverage**: >85%
- **Execution Time**: ~10 minutes (all tests)
- **Fast Tests**: ~2 minutes (unit tests only)

### Quick Reference

```bash
# Run everything
pytest

# Fast tests only
pytest -m "not slow"

# Specific category
pytest tests/performance/ -v

# With coverage
pytest --cov=src/distributed_training --cov-report=html

# Parallel execution
pytest -n auto
```

For more information, see:
- [Main README](../README.md)
- [Master Guide](../MASTER_GUIDE.md)
- [Project Summary](../PROJECT_SUMMARY.md)
