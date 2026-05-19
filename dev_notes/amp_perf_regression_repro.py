"""Minimal repro for the C++ AMP perf regression vs the Python POC.

Runs the same matched pipeline on both backends with FP32 then AMP and
prints side-by-side timings. Expected (on the reporter's RTX 5080):

  FP32: C++ ~1.9x faster per epoch than Python POC
  AMP : C++ slows down ~1.9x; Python POC slows down ~1.1x;
        the C++ vs Python advantage collapses to ~1.05x.

See dev_notes/amp_perf_regression.md (or the issue) for context.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

N_PLOTS = 20_000
N_SPECIES = 500
SP_PER_PLOT = 15
EPOCHS = 30
BS = 1024
HASH_DIM = 64
HIDDEN = [512, 256, 128]


def make_fixture(tmp: Path):
    rng = np.random.default_rng(0)
    genera = [f"Genus_{g:02d}" for g in range(40)]
    families = [f"Family_{f:02d}" for f in range(12)]
    species_to_gen = {f"sp_{s:04d}": rng.choice(genera) for s in range(N_SPECIES)}
    species_to_fam = {sp: families[hash(g) % len(families)] for sp, g in species_to_gen.items()}

    plot_ids = [f"P{i:05d}" for i in range(N_PLOTS)]
    header = pd.DataFrame({
        "PlotObservationID": plot_ids,
        "Area": np.exp(rng.normal(4.0, 1.0, N_PLOTS)),
        "Altitude": rng.normal(800, 400, N_PLOTS),
        "Latitude": rng.uniform(40, 55, N_PLOTS),
        "Longitude": rng.uniform(-5, 25, N_PLOTS),
        "Cov1": rng.normal(size=N_PLOTS),
        "Cov2": rng.normal(size=N_PLOTS),
    })
    rows = []
    for pid in plot_ids:
        n = int(np.clip(rng.poisson(SP_PER_PLOT), 3, 60))
        chosen = rng.choice(N_SPECIES, size=min(n, N_SPECIES), replace=False)
        for s in chosen:
            sp = f"sp_{s:04d}"
            rows.append({
                "PlotObservationID": pid,
                "WFO_TAXON": sp,
                "Cover %": float(abs(rng.normal(5, 3))) + 0.01,
                "WFO_GENUS": species_to_gen[sp],
                "WFO_FAMILY": species_to_fam[sp],
            })
    species = pd.DataFrame(rows)

    hp = tmp / "header.csv"; sp_ = tmp / "species.csv"
    header.to_csv(hp, index=False); species.to_csv(sp_, index=False)
    return hp, sp_


def run_python(hp, sp_, use_amp: bool):
    from resolve.data.dataset import ResolveDataset
    from resolve.train.trainer import Trainer
    roles = {
        "plot_id": "PlotObservationID", "species_id": "WFO_TAXON",
        "species_plot_id": "PlotObservationID", "abundance": "Cover %",
        "coords_lat": "Latitude", "coords_lon": "Longitude",
        "taxonomy_genus": "WFO_GENUS", "taxonomy_family": "WFO_FAMILY",
        "covariates": ["Cov1", "Cov2"],
    }
    targets = {
        "Area": {"column": "Area", "task": "regression", "transform": "log1p"},
        "Altitude": {"column": "Altitude", "task": "regression"},
    }
    torch.manual_seed(42); np.random.seed(42)
    ds = ResolveDataset.from_csv(header=hp, species=sp_, roles=roles, targets=targets, verbose=False)
    trainer = Trainer(
        dataset=ds, species_encoding="hash", hash_dim=HASH_DIM, hidden_dims=HIDDEN,
        batch_size=BS, max_epochs=EPOCHS, patience=10**6, lr=1e-3, weight_decay=1e-4,
        device="cuda", use_amp=use_amp, verbose=0, loss_config="mae",
    )
    t = time.perf_counter()
    trainer.fit()
    return time.perf_counter() - t


def run_cpp(hp, sp_, use_amp: bool):
    import resolve_core as rc
    roles = rc.RoleMapping()
    roles.plot_id = "PlotObservationID"; roles.species_id = "WFO_TAXON"
    roles.abundance = "Cover %"; roles.latitude = "Latitude"; roles.longitude = "Longitude"
    roles.genus = "WFO_GENUS"; roles.family = "WFO_FAMILY"
    roles.covariates = ["Cov1", "Cov2"]

    ts_area = rc.TargetSpec.regression(column="Area", transform=rc.TransformType.Log1p)
    ts_alt = rc.TargetSpec.regression(column="Altitude")
    ds_cfg = rc.DatasetConfig(); ds_cfg.hash_dim = HASH_DIM

    torch.manual_seed(42); np.random.seed(42)
    ds = rc.ResolveDataset.from_csv(
        header_path=str(hp), species_path=str(sp_),
        roles=roles, targets=[ts_area, ts_alt], config=ds_cfg,
    )
    mc = rc.ModelConfig()
    mc.hash_dim = HASH_DIM; mc.hidden_dims = list(HIDDEN); mc.dropout = 0.3
    mc.species_encoding = rc.SpeciesEncodingMode.Hash
    mc.encoder_architecture = rc.EncoderArchitecture.MLP
    model = rc.ResolveModel(ds.schema, mc)

    tc = rc.TrainConfig()
    tc.batch_size = BS; tc.max_epochs = EPOCHS; tc.patience = 10**6
    tc.lr = 1e-3; tc.weight_decay = 1e-4; tc.device = "cuda"; tc.use_amp = use_amp
    trainer = rc.Trainer(model, tc)
    trainer.prepare_data(ds, test_size=0.2, seed=42)
    t = time.perf_counter()
    trainer.fit()
    return time.perf_counter() - t


def main():
    tmp = Path(tempfile.mkdtemp(prefix="amp_repro_"))
    try:
        hp, sp_ = make_fixture(tmp)
        results = {}
        for amp in (False, True):
            tag = "AMP" if amp else "FP32"
            results[("py", tag)] = run_python(hp, sp_, amp)
            results[("cpp", tag)] = run_cpp(hp, sp_, amp)
            print(f"{tag:5s}  py={results[('py', tag)]:.2f}s  cpp={results[('cpp', tag)]:.2f}s  "
                  f"speedup={results[('py', tag)]/results[('cpp', tag)]:.2f}x")
        py_reg = results[("py", "AMP")] / results[("py", "FP32")]
        cpp_reg = results[("cpp", "AMP")] / results[("cpp", "FP32")]
        print(f"\nAMP regression vs FP32:  Python={py_reg:.2f}x  C++={cpp_reg:.2f}x")
        if cpp_reg / py_reg > 1.4:
            print("BUG REPRODUCED: C++ AMP regression > 1.4x worse than Python POC AMP regression")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
