"""Profile where time is spent in CUDA hash mode."""
import time
import torch
from bench_utils import create_role_mapping, HEADER_PATH, SPECIES_PATH
from resolve_core import (
    ResolveDataset, DatasetConfig, TargetSpec,
    ResolveModel, ModelConfig, Trainer, TrainConfig,
    SpeciesEncodingMode
)

# Load with CUDA hash
config = DatasetConfig()
config.species_encoding = SpeciesEncodingMode.Hash
config.hash_dim = 512
config.use_cuda_hash = True

target = TargetSpec.regression("Relevé area (m²)")
dataset = ResolveDataset.from_csv(
    HEADER_PATH, SPECIES_PATH, create_role_mapping(), [target], config
)

n_plots = dataset.n_plots
print(f"Dataset: {n_plots} plots")

# Create model
model_config = ModelConfig()
model_config.hash_dim = 512
model_config.hidden_dims = [256, 128]
model = ResolveModel(dataset.schema, model_config)

# Simulate what happens in one epoch
# We need to profile:
# 1. Hash computation time
# 2. Forward pass time
# 3. Backward pass time

print("\n=== Profiling individual operations ===")

# Move data to GPU
device = torch.device("cuda")
batch_size = 4096

# Warm up
for _ in range(3):
    _ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()

# Profile hash computation only (using the CSR kernel)
# We can't directly call the CUDA kernel from Python, but we can estimate
# by timing how long a typical batch takes

# Instead, let's profile the full training loop components
model.to(device)
model.train()

# Create dummy continuous features (what pre-computed mode would have)
continuous_dim = 51 + 512  # coords + covars + hash
dummy_continuous = torch.randn(batch_size, continuous_dim, device=device)
dummy_target = torch.randn(batch_size, device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Profile forward pass only
torch.cuda.synchronize()
t0 = time.perf_counter()
n_iters = 100
for _ in range(n_iters):
    out = model.forward(dummy_continuous, None, None, None, None)
torch.cuda.synchronize()
forward_time = (time.perf_counter() - t0) / n_iters * 1000
print(f"\nForward pass: {forward_time:.3f} ms/batch")

# Profile backward pass
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iters):
    optimizer.zero_grad()
    out = model.forward(dummy_continuous, None, None, None, None)
    loss = out["Relevé area (m²)"].mean()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
full_step_time = (time.perf_counter() - t0) / n_iters * 1000
backward_time = full_step_time - forward_time
print(f"Backward + optimizer: {backward_time:.3f} ms/batch")
print(f"Full step (fwd+bwd+opt): {full_step_time:.3f} ms/batch")

# Estimate epoch time from pure forward/backward
n_batches = (n_plots * 0.8) // batch_size  # 80% train
pure_compute_per_epoch = full_step_time * n_batches / 1000
print(f"\nBatches per epoch: {n_batches}")
print(f"Pure compute time per epoch: {pure_compute_per_epoch:.2f}s")

# Compare with actual measured epoch time
print(f"\nActual CUDA hash epoch time: ~0.82s")
print(f"Actual pre-computed epoch time: ~1.13s")

# The difference must be:
# - Hash computation overhead in CUDA hash mode
# - Memory transfer overhead in pre-computed mode
hash_overhead = 0.82 - pure_compute_per_epoch
precomputed_overhead = 1.13 - pure_compute_per_epoch
print(f"\nEstimated overhead:")
print(f"  CUDA hash mode: {hash_overhead:.2f}s ({hash_overhead/0.82*100:.0f}% of epoch)")
print(f"  Pre-computed mode: {precomputed_overhead:.2f}s ({precomputed_overhead/1.13*100:.0f}% of epoch)")

# Profile memory bandwidth
print("\n=== Memory bandwidth analysis ===")
# Hash embedding size per batch
hash_bytes_per_batch = batch_size * 512 * 4  # float32
print(f"Hash embedding per batch: {hash_bytes_per_batch / 1e6:.2f} MB")

# Species data read per batch (estimated)
avg_species_per_plot = dataset.raw_species_ids.shape[0] / dataset.n_plots
species_bytes_per_batch = batch_size * avg_species_per_plot * (8 + 4)  # int64 + float32
print(f"Species data per batch: {species_bytes_per_batch / 1e6:.2f} MB")
