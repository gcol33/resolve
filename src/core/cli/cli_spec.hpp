#pragma once

// The RESOLVE CLI's flag tables: one CommandSpec per subcommand.
//
// This is the single place a flag is declared. The parser accepts exactly what
// is listed here (anything else exits non-zero), `resolve help` is rendered
// from these rows, and each command reads its values back by the same name.
// Adding a knob is one row here plus one read in the command body.
//
// Header-only and torch-free so the tables can be exercised from a unit test
// without spawning a process.

#include "arg_parser.hpp"

namespace resolve_cli {

// Roles shared by `train` and `predict`: both read the same two CSVs with the
// same column semantics, so the rows live in one function rather than in two
// tables that can drift.
inline void append_role_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--plot-id", Arity::Value, "COL", "plot_id",
                     "Column name for plot ID"});
    flags.push_back({"--species-id", Arity::Value, "COL", "species_id",
                     "Column name for species ID"});
    flags.push_back({"--abundance", Arity::Value, "COL", "",
                     "Column name for abundance (optional)"});
    flags.push_back({"--lon", Arity::Value, "COL", "",
                     "Column name for longitude (optional)"});
    flags.push_back({"--lat", Arity::Value, "COL", "",
                     "Column name for latitude (optional)"});
    flags.push_back({"--genus", Arity::Value, "COL", "",
                     "Column name for genus (optional)"});
    flags.push_back({"--family", Arity::Value, "COL", "",
                     "Column name for family (optional)"});
    flags.push_back({"--covariate", Arity::Repeatable, "COL", "",
                     "Header column to use as a continuous covariate.\n"
                     "Repeat for each one; order defines the model's\n"
                     "covariate order."});
    flags.push_back({"--categorical", Arity::Repeatable, "COL", "",
                     "Header column to use as a categorical covariate.\n"
                     "Factorized at load time and embedded. Repeat for\n"
                     "each one. Must not also be a --covariate."});
}

inline const CommandSpec& train_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--header", Arity::Value, "PATH", "",
                         "Path to header CSV file (plot-level data)"});
        flags.push_back({"--species", Arity::Value, "PATH", "",
                         "Path to species CSV file (species occurrences)"});
        flags.push_back({"--output", Arity::Value, "PATH", "model.pt",
                         "Output path for the trained model"});
        append_role_flags(flags);
        flags.push_back({"--target", Arity::Repeatable, "SPEC", "",
                         "Target column and type. Repeat per target:\n"
                         "  COL:regression          continuous target\n"
                         "  COL:regression:log1p    fit on log1p(y),\n"
                         "                          predictions inverted\n"
                         "  COL:classification:N    N-class target\n"
                         "A bare COL means COL:regression."});

        // Dataset / encoding
        flags.push_back({"--encoding", Arity::Value, "MODE", "hash",
                         "Species encoding: hash, embed, sparse, rank_pool,\n"
                         "transformer"});
        flags.push_back({"--hash-dim", Arity::Value, "N", "32",
                         "Hash dimension"});
        flags.push_back({"--top-k", Arity::Value, "N", "3",
                         "Top-k species per taxonomy slot"});
        flags.push_back({"--top-k-species", Arity::Value, "N", "10",
                         "Species slots kept per plot in embed mode"});
        flags.push_back({"--selection", Arity::Value, "MODE", "top",
                         "Which species to keep: top, bottom, top_bottom, all"});
        flags.push_back({"--representation", Arity::Value, "MODE", "abundance",
                         "Species values: abundance or presence_absence"});
        flags.push_back({"--normalization", Arity::Value, "MODE", "raw",
                         "Abundance normalization: raw, norm, log1p"});
        flags.push_back({"--aggregation", Arity::Value, "MODE", "abundance",
                         "Taxonomy aggregation: abundance or count"});
        flags.push_back({"--no-taxonomy", Arity::Flag, "", "",
                         "Ignore the genus/family columns even when present"});
        flags.push_back({"--no-unknown-fraction", Arity::Flag, "", "",
                         "Drop the unknown-species-fraction feature (on by\n"
                         "default)"});
        flags.push_back({"--unknown-count", Arity::Flag, "", "",
                         "Add the unknown-species-count feature (off by\n"
                         "default)"});
        flags.push_back({"--use-cuda-hash", Arity::Flag, "", "",
                         "Hash species on the GPU each batch instead of\n"
                         "precomputing the embedding. Hash encoding only."});
        flags.push_back({"--pool-weighting", Arity::Value, "W", "log1p",
                         "Per-species pooling weight for rank_pool /\n"
                         "transformer encoders: binary, abundance, log1p,\n"
                         "norm, rank. Ignored for hash/embed/sparse."});
        flags.push_back({"--pool-species-cap", Arity::Value, "N", "0",
                         "Species-per-plot cap for the pool encoders:\n"
                         "0 = no cap, -1 = auto p99, >0 = manual cap"});

        // Model architecture
        flags.push_back({"--encoder-architecture", Arity::Value, "A", "mlp",
                         "Encoder architecture: mlp, ft_transformer, tabnet,\n"
                         "saint, trait_net, gnn, excelformer,\n"
                         "heterogeneous_gnn"});
        flags.push_back({"--hidden-dims", Arity::Value, "LIST", "2048,1024,512,256,128,64",
                         "Comma-separated MLP hidden layer widths"});
        flags.push_back({"--dropout", Arity::Value, "FLOAT", "0.3",
                         "Dropout rate in the encoder MLP"});
        flags.push_back({"--cover-dropout", Arity::Value, "FLOAT", "0.0",
                         "Probability of dropping a plot's cover values in\n"
                         "the rank_pool / transformer encoders"});
        flags.push_back({"--d-model", Arity::Value, "N", "128",
                         "Transformer token dimension"});
        flags.push_back({"--n-heads", Arity::Value, "N", "4",
                         "Transformer attention heads"});
        flags.push_back({"--n-attention-layers", Arity::Value, "N", "2",
                         "Transformer self-attention layers. Required >= 1\n"
                         "when --transformer-pooling cls."});
        flags.push_back({"--transformer-ff-dim", Arity::Value, "N", "256",
                         "Feed-forward width inside each transformer block"});
        flags.push_back({"--transformer-pooling", Arity::Value, "P", "attention",
                         "Transformer pooling: attention or cls"});
        flags.push_back({"--transformer-dropout", Arity::Value, "FLOAT", "0.1",
                         "Dropout inside transformer blocks"});

        // Training
        flags.push_back({"--batch-size", Arity::Value, "N", "4096",
                         "Batch size"});
        flags.push_back({"--batch-size-floor", Arity::Value, "N", "1024",
                         "Smallest batch size the auto-halve-on-OOM retry in\n"
                         "Trainer::fit is allowed to drop to. On CUDA\n"
                         "OutOfMemoryError the trainer releases optimizer /\n"
                         "AMP / GPU caches, halves batch_size, and restarts\n"
                         "from epoch 0; below this floor the OOM is\n"
                         "rethrown. Especially relevant on Windows, where\n"
                         "the allocator's expandable_segments option is\n"
                         "unavailable."});
        flags.push_back({"--max-epochs", Arity::Value, "N", "500",
                         "Maximum epochs"});
        flags.push_back({"--patience", Arity::Value, "N", "50",
                         "Early stopping patience"});
        flags.push_back({"--lr", Arity::Value, "FLOAT", "0.001",
                         "Learning rate"});
        flags.push_back({"--weight-decay", Arity::Value, "FLOAT", "0.0001",
                         "AdamW weight decay"});
        flags.push_back({"--lr-scheduler", Arity::Value, "S", "none",
                         "Learning rate schedule: none, step, cosine"});
        flags.push_back({"--lr-step-size", Arity::Value, "N", "100",
                         "Epochs between decays, for --lr-scheduler step"});
        flags.push_back({"--lr-gamma", Arity::Value, "FLOAT", "0.1",
                         "Decay factor, for --lr-scheduler step"});
        flags.push_back({"--lr-min", Arity::Value, "FLOAT", "0.000001",
                         "Floor learning rate, for --lr-scheduler cosine"});
        flags.push_back({"--loss-config", Arity::Value, "M", "combined",
                         "Loss recipe: mae, smape, combined, nca"});
        flags.push_back({"--band-threshold", Arity::Value, "FLOAT", "0.25",
                         "Tolerance band the phase-3 loss penalty optimizes\n"
                         "toward (predictions outside [1-t, 1+t] times the\n"
                         "target are penalized)"});
        flags.push_back({"--band-thresholds", Arity::Value, "LIST", "0.1,0.25,0.5",
                         "Comma-separated band accuracies to REPORT. Does\n"
                         "not change what training optimizes."});
        flags.push_back({"--nca-temperature", Arity::Value, "FLOAT", "0.1",
                         "Scale of the stochastic-neighbour softmax in the\n"
                         "NCA term, i.e. the effective number of neighbours\n"
                         "each sample spreads over. Acts only under\n"
                         "--loss-config nca."});
        flags.push_back({"--nca-neighbors", Arity::Value, "N", "32",
                         "Neighbours per sample the NCA term sums over, its\n"
                         "most similar in-batch samples. 0 or less keeps the\n"
                         "whole batch. Acts only under --loss-config nca."});
        flags.push_back({"--nca-weight", Arity::Value, "FLOAT", "0.1",
                         "Weight of the NCA term against the cross-entropy\n"
                         "it is added to. Acts only under --loss-config nca."});
        flags.push_back({"--checkpoint-dir", Arity::Value, "PATH", "",
                         "Directory for per-epoch checkpoints and the\n"
                         "progress file (empty = disabled)"});
        flags.push_back({"--checkpoint-every", Arity::Value, "N", "0",
                         "Also checkpoint every N epochs (0 = best only).\n"
                         "Requires --checkpoint-dir."});
        flags.push_back({"--amp", Arity::Flag, "", "",
                         "Enable CUDA automatic mixed precision (off by\n"
                         "default; helps transformer encoders, rarely MLPs)"});
        flags.push_back({"--no-amp", Arity::Flag, "", "",
                         "Disable AMP explicitly (the default)"});
        flags.push_back({"--cudnn-benchmark", Arity::Flag, "", "",
                         "Enable the cuDNN autotuner (the default)"});
        flags.push_back({"--no-cudnn-benchmark", Arity::Flag, "", "",
                         "Disable the cuDNN autotuner, for run-to-run\n"
                         "determinism"});
        flags.push_back({"--no-tf32", Arity::Flag, "", "",
                         "Disable TF32 matmuls on Ampere+ GPUs"});
        flags.push_back({"--test-size", Arity::Value, "FLOAT", "0.2",
                         "Test split ratio"});
        flags.push_back({"--seed", Arity::Value, "N", "42",
                         "Random seed. Seeds the global torch RNG before the\n"
                         "model is constructed (weight init) and drives the\n"
                         "train/test split, the epoch shuffle, and the\n"
                         "cross-validation folds, so two identical runs\n"
                         "produce the same model."});
        flags.push_back({"--cuda", Arity::Flag, "", "",
                         "Use CUDA if available"});
        flags.push_back({"--vram-fraction", Arity::Value, "FLOAT", "1.0",
                         "Fraction of GPU VRAM the PyTorch caching allocator\n"
                         "may use. Dedicated training jobs on a solo GPU use\n"
                         "the full device. Pass an explicit lower value\n"
                         "(e.g. 0.80) when sharing the GPU with a desktop /\n"
                         "GUI to leave headroom."});

        // Cross-validation
        flags.push_back({"--cv-folds", Arity::Value, "N", "0",
                         "Run N-fold cross-validation BEFORE the final fit\n"
                         "and report per-fold plus mean +/- std metrics\n"
                         "(0 = skip). Each fold starts from the untrained\n"
                         "weights; the trainer's split and weights are\n"
                         "restored afterwards, so the saved model is the\n"
                         "same one a run without this flag produces."});
        flags.push_back({"--cv-spatial", Arity::Flag, "", "",
                         "Make the folds spatial blocks instead of random\n"
                         "rows. Requires --lon / --lat."});
        flags.push_back({"--cv-lat-size", Arity::Value, "FLOAT", "1.0",
                         "Spatial block height in degrees latitude"});
        flags.push_back({"--cv-lon-size", Arity::Value, "FLOAT", "1.0",
                         "Spatial block width in degrees longitude"});
        flags.push_back({"--cv-balance", Arity::Flag, "", "",
                         "Assign spatial blocks to folds by greedy\n"
                         "bin-packing instead of round-robin"});

        return CommandSpec("train", "Train Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& predict_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--model", Arity::Value, "PATH", "",
                         "Path to the trained model"});
        flags.push_back({"--header", Arity::Value, "PATH", "",
                         "Path to header CSV file"});
        flags.push_back({"--species", Arity::Value, "PATH", "",
                         "Path to species CSV file"});
        flags.push_back({"--output", Arity::Value, "PATH", "predictions.csv",
                         "Output path for predictions"});
        append_role_flags(flags);
        flags.push_back({"--cuda", Arity::Flag, "", "",
                         "Use CUDA if available (default: CPU). Predict on a\n"
                         "5M-param MLP over 300k plots is ~12s on CPU vs ~1s\n"
                         "on GPU; the OOM and bookkeeping cost of GPU\n"
                         "predict on 16 GiB-class cards usually outweighs\n"
                         "the speedup."});
        flags.push_back({"--predict-batch-size", Arity::Value, "N", "4096",
                         "Forward-pass batch size for inference. Pass -1 to\n"
                         "disable chunking (one forward over the entire\n"
                         "dataset; can OOM on >150k plots at typical hidden\n"
                         "sizes)."});
        flags.push_back({"--vram-fraction", Arity::Value, "FLOAT", "1.0",
                         "Fraction of GPU VRAM the PyTorch caching allocator\n"
                         "may use"});
        return CommandSpec("predict", "Predict Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& info_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--model", Arity::Value, "PATH", "",
                         "Path to the trained model"});
        return CommandSpec("info", "Info Options:", std::move(flags));
    }();
    return spec;
}

}  // namespace resolve_cli
