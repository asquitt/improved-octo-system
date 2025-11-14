# Common Errors and Solutions

## CUDA Errors

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision
- Clear cache: torch.cuda.empty_cache()

### Device-side assert
```
RuntimeError: CUDA error: device-side assert triggered
```
**Solutions:**
- Run on CPU to get better error message
- Check label ranges match model output
- Verify data types

## Distributed Errors

### Timeout
```
RuntimeError: ProcessGroupNCCL timeout
```
**Solutions:**
- Increase timeout in init_process_group
- Check network connectivity
- Verify all ranks reach synchronization points

### Rank mismatch
```
RuntimeError: rank not in the process group
```
**Solutions:**
- Ensure consistent world_size
- Check environment variables
- Verify process group initialization

### Hangs at barrier
**Solutions:**
- Check all ranks execute same code
- Remove conditional barriers
- Verify data loading doesn't hang

## Data Errors

### Shape mismatch
```
RuntimeError: size mismatch
```
**Solutions:**
- Print shapes at each step
- Verify transforms
- Check batch dimensions

### DataLoader hang
**Solutions:**
- Reduce num_workers
- Check data path exists
- Simplify transforms first

## Installation Errors

### Import errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solutions:**
- pip install torch torchvision
- Check virtual environment
- Verify Python version

### CUDA not available
**Solutions:**
- Install CUDA-enabled PyTorch
- Check GPU drivers: nvidia-smi
- Verify CUDA version matches PyTorch
