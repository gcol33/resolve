"""Benchmark AMP (FP16) vs FP32 for CUDA hash mode."""
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

# Test configurations
configs = [
    {"name": "FP32 (baseline)", "use_amp": False, "batch_size": 32768},
    {"name": "FP16 (AMP)", "use_amp": True, "batch_size": 32768},
    {"name": "FP16 (AMP) + larger batch", "use_amp": True, "batch_size": 65536},
]

print("\n" + "="*60)
print("AMP (FP16) vs FP32 Benchmark")
print("="*60)

results = []

for cfg in configs:
    print(f"\n--- {cfg['name']} ---")

    # Create model
    model_config = ModelConfig()
    model_config.hash_dim = 512
    model_config.hidden_dims = [256, 128]
    model = ResolveModel(dataset.schema, model_config)

    # Create trainer
    train_config = TrainConfig()
    train_config.max_epochs = 5
    train_config.batch_size = cfg["batch_size"]
    train_config.lr = 1e-3
    train_config.patience = 100  # Don't early stop
    train_config.device = "cuda"
    train_config.use_amp = cfg["use_amp"]

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

    print(f"  Epochs: {n_epochs}")
    print(f"  Time/epoch: {time_per_epoch:.3f}s")
    print(f"  Final test loss: {result.test_loss_history[-1]:.4f}")

    results.append({
        "name": cfg["name"],
        "time_per_epoch": time_per_epoch,
        "final_loss": result.test_loss_history[-1]
    })

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Configuration':<30} {'Time/Epoch':>12} {'Speedup':>10} {'Test Loss':>12}")
print("-"*66)

baseline_time = results[0]["time_per_epoch"]
for r in results:
    speedup = baseline_time / r["time_per_epoch"]
    print(f"{r['name']:<30} {r['time_per_epoch']:>11.3f}s {speedup:>9.2f}x {r['final_loss']:>12.4f}")
