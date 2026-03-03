"""Benchmark CUDA hash embedding computation vs pre-computed."""
import time
import torch
from bench_utils import create_role_mapping, HEADER_PATH, SPECIES_PATH
from resolve_core import (
    ResolveDataset, DatasetConfig, TargetSpec,
    ResolveModel, ModelConfig, Trainer, TrainConfig,
    SpeciesEncodingMode
)

def benchmark_mode(use_cuda_hash: bool, n_epochs: int = 5) -> dict:
    """Benchmark training with or without CUDA hash computation."""
    mode_name = "CUDA hash (on-the-fly)" if use_cuda_hash else "Pre-computed hash"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {mode_name}")
    print(f"{'='*60}")

    # Dataset config
    config = DatasetConfig()
    config.species_encoding = SpeciesEncodingMode.Hash
    config.hash_dim = 512
    config.use_cuda_hash = use_cuda_hash

    # Load dataset
    print("Loading dataset...")
    t0 = time.perf_counter()

    # Define targets
    target = TargetSpec.regression("Relevé area (m²)")
    targets = [target]

    dataset = ResolveDataset.from_csv(
        HEADER_PATH,
        SPECIES_PATH,
        create_role_mapping(),
        targets,
        config
    )
    load_time = time.perf_counter() - t0
    print(f"  Dataset load time: {load_time:.2f}s")
    print(f"  n_plots: {dataset.n_plots}")

    # Model config
    model_config = ModelConfig()
    model_config.hash_dim = 512
    model_config.hidden_dims = [256, 128]
    model_config.dropout = 0.1

    # Create model (Trainer will move to device)
    print("Creating model...")
    model = ResolveModel(dataset.schema, model_config)

    # Train config - use attribute names from bindings
    train_config = TrainConfig()
    train_config.max_epochs = n_epochs
    train_config.batch_size = 4096
    train_config.lr = 1e-3
    train_config.patience = n_epochs + 1  # Disable early stopping
    train_config.device = "cuda"  # Must set this for CUDA hash to work

    # Debug dataset state
    has_raw = dataset.has_raw_species_data()
    print(f"  has_raw_species_data: {has_raw}")

    # Create trainer and prepare data
    print("Preparing data...")
    trainer = Trainer(model, train_config)
    t0 = time.perf_counter()
    trainer.prepare_data(dataset, test_size=0.2, seed=42)
    prep_time = time.perf_counter() - t0
    print(f"  Data prep time: {prep_time:.2f}s")

    # Warm-up GPU
    print("Warming up GPU...")
    _ = torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()

    # Train
    print(f"Training for {n_epochs} epochs...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = trainer.fit()
    torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    time_per_epoch = train_time / n_epochs
    print(f"  Total train time: {train_time:.2f}s")
    print(f"  Time per epoch: {time_per_epoch:.2f}s")
    best_test_loss = min(result.test_loss_history)
    print(f"  Final test loss: {best_test_loss:.4f}")

    return {
        "mode": mode_name,
        "use_cuda_hash": use_cuda_hash,
        "load_time": load_time,
        "prep_time": prep_time,
        "train_time": train_time,
        "time_per_epoch": time_per_epoch,
        "best_test_loss": best_test_loss,
    }

def main():
    print("CUDA Hash Embedding Benchmark")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    n_epochs = 3  # Few epochs for quick benchmark

    # Benchmark pre-computed (baseline)
    baseline = benchmark_mode(use_cuda_hash=False, n_epochs=n_epochs)

    # Benchmark CUDA hash
    cuda_hash = benchmark_mode(use_cuda_hash=True, n_epochs=n_epochs)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Pre-computed':>15} {'CUDA hash':>15} {'Speedup':>10}")
    print("-"*65)

    # Dataset load time (CUDA hash should be faster - no CPU hash computation)
    load_speedup = baseline["load_time"] / cuda_hash["load_time"]
    print(f"{'Dataset load time':<25} {baseline['load_time']:>14.2f}s {cuda_hash['load_time']:>14.2f}s {load_speedup:>9.2f}x")

    # Data prep time
    prep_speedup = baseline["prep_time"] / cuda_hash["prep_time"]
    print(f"{'Data prep time':<25} {baseline['prep_time']:>14.2f}s {cuda_hash['prep_time']:>14.2f}s {prep_speedup:>9.2f}x")

    # Training time per epoch
    train_speedup = baseline["time_per_epoch"] / cuda_hash["time_per_epoch"]
    print(f"{'Time per epoch':<25} {baseline['time_per_epoch']:>14.2f}s {cuda_hash['time_per_epoch']:>14.2f}s {train_speedup:>9.2f}x")

    # Total training time
    total_speedup = baseline["train_time"] / cuda_hash["train_time"]
    print(f"{'Total train time':<25} {baseline['train_time']:>14.2f}s {cuda_hash['train_time']:>14.2f}s {total_speedup:>9.2f}x")

    print("-"*65)
    print(f"{'Best test loss':<25} {baseline['best_test_loss']:>15.4f} {cuda_hash['best_test_loss']:>15.4f}")

    if train_speedup > 1:
        print(f"\nCUDA hash is {train_speedup:.2f}x FASTER per epoch")
    else:
        print(f"\nCUDA hash is {1/train_speedup:.2f}x SLOWER per epoch")

if __name__ == "__main__":
    main()
