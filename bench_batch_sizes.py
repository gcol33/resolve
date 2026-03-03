"""Benchmark different batch sizes for CUDA hash mode."""
import time
import torch
from bench_utils import create_role_mapping, HEADER_PATH, SPECIES_PATH
from resolve_core import (
    ResolveDataset, DatasetConfig, TargetSpec,
    ResolveModel, ModelConfig, Trainer, TrainConfig,
    SpeciesEncodingMode
)

# Load dataset with CUDA hash
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
print(f"Max feasible batch size: {int(n_plots * 0.8)} (80% train split)")

# Batch sizes to test
batch_sizes = [4096, 8192, 16384, 32768, 65536]

print("\n" + "="*60)
print("CUDA Hash Mode - Batch Size Benchmark")
print("="*60)

results = []

for batch_size in batch_sizes:
    if batch_size > n_plots * 0.8:
        print(f"\nSkipping batch_size={batch_size} (larger than train set)")
        continue

    print(f"\n--- Batch size: {batch_size} ---")

    # Create model
    model_config = ModelConfig()
    model_config.hash_dim = 512
    model_config.hidden_dims = [256, 128]
    model = ResolveModel(dataset.schema, model_config)

    # Create trainer
    train_config = TrainConfig()
    train_config.max_epochs = 5
    train_config.batch_size = batch_size
    train_config.lr = 1e-3
    train_config.patience = 100  # Don't early stop
    train_config.device = "cuda"

    trainer = Trainer(model, train_config)
    trainer.prepare_data(dataset, 0.2, 42)

    # Warm up
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    result = trainer.fit()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    n_epochs = len(result.train_loss_history)
    time_per_epoch = total_time / n_epochs
    n_batches = int((n_plots * 0.8) // batch_size)
    time_per_batch = time_per_epoch / n_batches * 1000  # ms

    print(f"  Epochs: {n_epochs}")
    print(f"  Time/epoch: {time_per_epoch:.3f}s")
    print(f"  Batches/epoch: {n_batches}")
    print(f"  Time/batch: {time_per_batch:.3f}ms")

    results.append({
        "batch_size": batch_size,
        "time_per_epoch": time_per_epoch,
        "n_batches": n_batches,
        "time_per_batch": time_per_batch
    })

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Batch Size':>12} {'Time/Epoch':>12} {'Batches':>10} {'Time/Batch':>12} {'Speedup':>10}")
print("-"*60)

baseline_time = results[0]["time_per_epoch"] if results else 1.0
for r in results:
    speedup = baseline_time / r["time_per_epoch"]
    print(f"{r['batch_size']:>12} {r['time_per_epoch']:>11.3f}s {r['n_batches']:>10} {r['time_per_batch']:>11.3f}ms {speedup:>9.2f}x")
