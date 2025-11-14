# Distributed Training Glossary

## Terms

**Rank:** Unique ID for each process (0, 1, 2, ...)

**World Size:** Total number of processes

**Local Rank:** Rank within a node

**Process Group:** Collection of communicating processes

**Backend:** Communication library (NCCL, Gloo, MPI)

**AllReduce:** Combine values from all processes

**Broadcast:** Send from one to all

**Scatter/Gather:** Distribute/collect data

**DDP:** DistributedDataParallel

**FSDP:** Fully Sharded Data Parallel

**Pipeline Parallel:** Split model layers across devices

**Tensor Parallel:** Split tensors across devices

**Gradient Accumulation:** Sum gradients over multiple steps

**Mixed Precision:** Use FP16/BF16 for speed, FP32 for stability

**Checkpoint:** Saved model state for recovery

**Epoch:** One pass through entire dataset

**Batch:** Subset of data processed together

**Learning Rate:** Step size for optimizer

**Scheduler:** Adjusts learning rate during training
