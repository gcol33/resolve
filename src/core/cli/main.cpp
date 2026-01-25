// RESOLVE CLI - Command-line interface for training and inference
// Usage:
//   resolve train --header h.csv --species s.csv --output model.pt [options]
//   resolve predict --model model.pt --header h.csv --species s.csv --output predictions.csv
//   resolve info --model model.pt

#include <iostream>
#include <string>
#include <vector>
#include <optional>

#include "resolve/resolve.hpp"

// Forward declarations for command handlers
int train_command(
    const std::string& header_path,
    const std::string& species_path,
    const std::string& output_path,
    const std::string& plot_id_col,
    const std::string& species_id_col,
    const std::optional<std::string>& abundance_col,
    const std::optional<std::string>& lon_col,
    const std::optional<std::string>& lat_col,
    const std::optional<std::string>& genus_col,
    const std::optional<std::string>& family_col,
    const std::vector<std::string>& target_cols,
    const std::vector<std::string>& target_types,
    const std::string& species_encoding,
    int hash_dim,
    int top_k,
    int batch_size,
    int max_epochs,
    int patience,
    float lr,
    float test_size,
    bool use_cuda
);

int predict_command(
    const std::string& model_path,
    const std::string& header_path,
    const std::string& species_path,
    const std::string& output_path,
    const std::string& plot_id_col,
    const std::string& species_id_col,
    const std::optional<std::string>& abundance_col,
    const std::optional<std::string>& lon_col,
    const std::optional<std::string>& lat_col,
    const std::optional<std::string>& genus_col,
    const std::optional<std::string>& family_col,
    bool use_cuda
);

int info_command(const std::string& model_path);

void print_usage() {
    std::cout << R"(
RESOLVE - Species composition-based prediction

Usage:
  resolve train [options]     Train a new model
  resolve predict [options]   Make predictions with a trained model
  resolve info [options]      Display model information

Train Options:
  --header PATH          Path to header CSV file (plot-level data)
  --species PATH         Path to species CSV file (species occurrences)
  --output PATH          Output path for trained model (default: model.pt)
  --plot-id COL          Column name for plot ID (default: plot_id)
  --species-id COL       Column name for species ID (default: species_id)
  --abundance COL        Column name for abundance (optional)
  --lon COL              Column name for longitude (optional)
  --lat COL              Column name for latitude (optional)
  --genus COL            Column name for genus (optional)
  --family COL           Column name for family (optional)
  --target COL:TYPE      Target column and type (regression/classification:N)
                         Can be specified multiple times
  --encoding MODE        Species encoding: hash, embed, sparse (default: hash)
  --hash-dim N           Hash dimension (default: 32)
  --top-k N              Top-k species for encoding (default: 3)
  --batch-size N         Batch size (default: 4096)
  --max-epochs N         Maximum epochs (default: 500)
  --patience N           Early stopping patience (default: 50)
  --lr FLOAT             Learning rate (default: 0.001)
  --test-size FLOAT      Test split ratio (default: 0.2)
  --cuda                 Use CUDA if available

Predict Options:
  --model PATH           Path to trained model
  --header PATH          Path to header CSV file
  --species PATH         Path to species CSV file
  --output PATH          Output path for predictions (default: predictions.csv)
  --plot-id COL          Column name for plot ID (default: plot_id)
  --species-id COL       Column name for species ID (default: species_id)
  --abundance COL        Column name for abundance (optional)
  --lon COL              Column name for longitude (optional)
  --lat COL              Column name for latitude (optional)
  --genus COL            Column name for genus (optional)
  --family COL           Column name for family (optional)
  --cuda                 Use CUDA if available

Info Options:
  --model PATH           Path to trained model

Examples:
  resolve train --header plots.csv --species occurrences.csv \
                --target area:regression --target habitat:classification:9 \
                --output model.pt

  resolve predict --model model.pt --header new_plots.csv \
                  --species new_occurrences.csv --output predictions.csv
)" << std::endl;
}

// Simple argument parser
class ArgParser {
public:
    ArgParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            args_.push_back(argv[i]);
        }
    }

    bool has(const std::string& flag) const {
        return std::find(args_.begin(), args_.end(), flag) != args_.end();
    }

    std::string get(const std::string& flag, const std::string& default_val = "") const {
        auto it = std::find(args_.begin(), args_.end(), flag);
        if (it != args_.end() && (it + 1) != args_.end()) {
            return *(it + 1);
        }
        return default_val;
    }

    std::optional<std::string> get_optional(const std::string& flag) const {
        auto it = std::find(args_.begin(), args_.end(), flag);
        if (it != args_.end() && (it + 1) != args_.end()) {
            return *(it + 1);
        }
        return std::nullopt;
    }

    std::vector<std::string> get_all(const std::string& flag) const {
        std::vector<std::string> result;
        for (auto it = args_.begin(); it != args_.end(); ++it) {
            if (*it == flag && (it + 1) != args_.end()) {
                result.push_back(*(it + 1));
            }
        }
        return result;
    }

    std::string command() const {
        if (!args_.empty() && args_[0][0] != '-') {
            return args_[0];
        }
        return "";
    }

private:
    std::vector<std::string> args_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    ArgParser args(argc, argv);
    std::string cmd = args.command();

    if (cmd == "train") {
        // Parse target specifications
        auto target_specs = args.get_all("--target");
        std::vector<std::string> target_cols;
        std::vector<std::string> target_types;

        for (const auto& spec : target_specs) {
            auto pos = spec.find(':');
            if (pos != std::string::npos) {
                target_cols.push_back(spec.substr(0, pos));
                target_types.push_back(spec.substr(pos + 1));
            } else {
                target_cols.push_back(spec);
                target_types.push_back("regression");
            }
        }

        return train_command(
            args.get("--header"),
            args.get("--species"),
            args.get("--output", "model.pt"),
            args.get("--plot-id", "plot_id"),
            args.get("--species-id", "species_id"),
            args.get_optional("--abundance"),
            args.get_optional("--lon"),
            args.get_optional("--lat"),
            args.get_optional("--genus"),
            args.get_optional("--family"),
            target_cols,
            target_types,
            args.get("--encoding", "hash"),
            std::stoi(args.get("--hash-dim", "32")),
            std::stoi(args.get("--top-k", "3")),
            std::stoi(args.get("--batch-size", "4096")),
            std::stoi(args.get("--max-epochs", "500")),
            std::stoi(args.get("--patience", "50")),
            std::stof(args.get("--lr", "0.001")),
            std::stof(args.get("--test-size", "0.2")),
            args.has("--cuda")
        );
    }
    else if (cmd == "predict") {
        return predict_command(
            args.get("--model"),
            args.get("--header"),
            args.get("--species"),
            args.get("--output", "predictions.csv"),
            args.get("--plot-id", "plot_id"),
            args.get("--species-id", "species_id"),
            args.get_optional("--abundance"),
            args.get_optional("--lon"),
            args.get_optional("--lat"),
            args.get_optional("--genus"),
            args.get_optional("--family"),
            args.has("--cuda")
        );
    }
    else if (cmd == "info") {
        return info_command(args.get("--model"));
    }
    else if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }
    else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage();
        return 1;
    }
}
