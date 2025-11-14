# Distributed Training Best Practices

## 1. Start Simple
- Get single-GPU working first
- Add distributed features incrementally
- Test at each step

## 2. Data Loading
- Use num_workers >= 4
- Pin memory for faster CPU-GPU transfer
- Use DistributedSampler for DDP
- Prefetch data

## 3. Checkpointing
- Save from rank 0 only
- Include optimizer state
- Save every N epochs
- Keep best and latest

## 4. Debugging
- Print from rank 0 only
- Set deterministic seeds
- Use NCCL_DEBUG=INFO
- Start with small model

## 5. Performance
- Use mixed precision
- Enable torch.compile
- Profile bottlenecks
- Overlap communication

## 6. Monitoring
- Track loss, accuracy, throughput
- Monitor GPU utilization
- Log learning rate
- Use TensorBoard/wandb

## 7. Reproducibility
- Set all random seeds
- Use deterministic algorithms
- Save full configuration
- Version control code
