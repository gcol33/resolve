"""Benchmark TF32 and cuDNN benchmark mode."""
import sys
sys.path.insert(0, 'src/core/python/src')
sys.path.insert(0, 'src')

import time
import torch
from resolve_core import (
    ResolveDataset, DatasetConfig, RoleMapping, TargetSpec,
    ResolveModel, ModelConfig, Trainer, TrainConfig,
    SpeciesEncodingMode
)

HEADER_PATH = "J:/Phd Local/Gilles_paper_resolve/data/iter_bench_header.csv"
SPECIES_PATH = "J:/Phd Local/Gilles_paper_resolve/data/iter_bench_species.csv"

def create_role_mapping():
    roles = RoleMapping()
    roles.plot_id = "PlotObservationID"
    roles.species_id = "WFO_TAXON"
    roles.abundance = "Cover %"
    roles.longitude = "Longitude"
    roles.latitude = "Latitude"
    roles.genus = "WFO_GENUS"
    roles.family = "WFO_FAMILY"
    return roles

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")

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
    {"name": "TF32=OFF, cuDNN_bench=OFF", "allow_tf32": False, "cudnn_benchmark": False},
    {"name": "TF32=OFF, cuDNN_bench=ON", "allow_tf32": False, "cudnn_benchmark": True},
    {"name": "TF32=ON,  cuDNN_bench=OFF", "allow_tf32": True, "cudnn_benchmark": False},
    {"name": "TF32=ON,  cuDNN_bench=ON (default)", "allow_tf32": True, "cudnn_benchmark": True},
]

print("\n" + "="*70)
print("TF32 + cuDNN Benchmark Mode - Performance Test")
print("="*70)

results = []
batch_size = 32768

for cfg in configs:
    print(f"\n--- {cfg['name']} ---")

    # Create model
    model_config = ModelConfig()
    model_config.hash_dim = 512
    model_config.hidden_dims = [256, 128]
    model = ResolveModel(dataset.schema, model_config)

    # Create trainer
    train_config = TrainConfig()
    train_config.max_epochs = 10  # More epochs for stable timing
    train_config.batch_size = batch_size
    train_config.lr = 1e-3
    train_config.patience = 100  # Don't early stop
    train_config.device = "cuda"
    train_config.allow_tf32 = cfg["allow_tf32"]
    train_config.cudnn_benchmark = cfg["cudnn_benchmark"]

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

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"{'Configuration':<40} {'Time/Epoch':>12} {'Speedup':>10} {'Test Loss':>12}")
print("-"*76)

baseline_time = results[0]["time_per_epoch"]
for r in results:
    speedup = baseline_time / r["time_per_epoch"]
    print(f"{r['name']:<40} {r['time_per_epoch']:>11.3f}s {speedup:>9.2f}x {r['final_loss']:>12.4f}")
