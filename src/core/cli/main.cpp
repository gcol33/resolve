// RESOLVE CLI - Command-line interface for training and inference
// Usage:
//   resolve train --header h.csv --species s.csv --output model.pt [options]
//   resolve predict --model model.pt --header h.csv --species s.csv --output predictions.csv
//   resolve info --model model.pt
//
// Flags are declared once per subcommand in cli_spec.hpp. This file only
// routes: it picks the subcommand's table, hands the remaining tokens to
// parse_args (which rejects anything the table does not declare), and passes
// the result to the command implementation.

#include <iostream>
#include <string>
#include <vector>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "cli_spec.hpp"

using resolve_cli::ArgError;
using resolve_cli::ParsedArgs;

// Command handlers. Each reads its values from the ParsedArgs by the same flag
// names its CommandSpec declares; reading an undeclared name throws.
int train_command(const ParsedArgs& args);
int predict_command(const ParsedArgs& args);
int info_command(const ParsedArgs& args);

namespace {

void print_usage() {
    std::cout << R"(
RESOLVE - Species composition-based prediction

Usage:
  resolve train [options]     Train a new model
  resolve predict [options]   Make predictions with a trained model
  resolve info [options]      Display model information
  resolve version             Print version

Pass --help after a subcommand for that command's flags only.

)";
    std::cout << resolve_cli::render_usage(resolve_cli::train_spec()) << "\n";
    std::cout << resolve_cli::render_usage(resolve_cli::predict_spec()) << "\n";
    std::cout << resolve_cli::render_usage(resolve_cli::info_spec()) << "\n";
    std::cout << R"(Prediction output columns:
  plot_id, then one column per schema target in schema order. A classification
  target gets TWO columns: <target> carries the original class label from the
  checkpoint's class vocabulary (the integer code when the checkpoint has no
  label vocabulary, or the column was already integer-coded) and <target>_code
  always carries the integer code the model predicted.

Examples:
  resolve train --header plots.csv --species occurrences.csv \
                --target area:regression:log1p \
                --target habitat:classification:9 \
                --covariate elevation --covariate slope \
                --categorical bedrock \
                --seed 42 --output model.pt

  resolve predict --model model.pt --header new_plots.csv \
                  --species new_occurrences.csv --output predictions.csv
)" << std::endl;
}

// `--help` / `-h` after a subcommand prints that command's block. Handled here
// rather than as a table row so every command gets it without repeating a row,
// and so it works even alongside an otherwise invalid flag.
bool wants_command_help(const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (token == "--help" || token == "-h") return true;
    }
    return false;
}

int run_command(const resolve_cli::CommandSpec& spec,
                const std::vector<std::string>& tokens,
                int (*handler)(const ParsedArgs&)) {
    if (wants_command_help(tokens)) {
        std::cout << resolve_cli::render_usage(spec);
        return 0;
    }
    return handler(resolve_cli::parse_args(spec, tokens));
}

int run_cli(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string cmd = argv[1];
    const std::vector<std::string> tokens(argv + 2, argv + argc);

    if (cmd == "train") {
        return run_command(resolve_cli::train_spec(), tokens, &train_command);
    }
    if (cmd == "predict") {
        return run_command(resolve_cli::predict_spec(), tokens, &predict_command);
    }
    if (cmd == "info") {
        return run_command(resolve_cli::info_spec(), tokens, &info_command);
    }
    if (cmd == "version" || cmd == "--version") {
        std::cout << "resolve " << resolve::VERSION << std::endl;
        return 0;
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    std::cerr << "Unknown command: " << cmd << std::endl;
    print_usage();
    return 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Issue #19: a native fault during a (possibly overnight, headless) run must
    // fail fast with the fault's NTSTATUS, never hang on the Windows JIT
    // debugger. Arm the handler before any engine work. No-op off Windows.
    resolve::install_crash_handler(0);

    int rc;
    try {
        rc = run_cli(argc, argv);
    } catch (const ArgError& e) {
        // Unknown flag, missing value, stray positional, or a non-numeric value
        // for a numeric flag. Every one of these used to be ignored, which for
        // a research CLI is a wrong run that looks fine.
        std::cerr << "Error: " << e.what() << std::endl;
        rc = 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        rc = 1;
    }

    // Issue #18: the command is done and its output is flushed; from here a
    // libtorch teardown fault is a benign artifact. Carry the command's real
    // exit code through, then mark work complete so the handler exits with it.
    resolve::install_crash_handler(rc);
    resolve::signal_work_complete();
    return rc;
}
