// RESOLVE CLI - what `train`, `predict` and `predict --suite` share: reading
// the role flags, choosing the device, and loading the plots from a header and
// a species table or from a species table alone.

#ifndef RESOLVE_CLI_COMMON_HPP
#define RESOLVE_CLI_COMMON_HPP

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "resolve/dataset.hpp"
#include "resolve/gpu.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/suite.hpp"

#include "arg_parser.hpp"

namespace resolve_cli {

// A role flag's value when it was given. A column a run reads only when named
// (abundance, coordinates, taxonomy) stays unset otherwise, and a suite keeps
// the manifest's name for a column the command line does not rename.
inline std::optional<std::string> role_flag(const ParsedArgs& args, const char* flag) {
    std::string value;
    if (args.get_if_present(flag, value)) return value;
    return std::nullopt;
}

// The roles of `train` and `predict`: the plot and species columns (with their
// table defaults), the optional columns that were named, and the covariate and
// categorical columns in command-line order.
inline resolve::RoleMapping read_roles(const ParsedArgs& args) {
    resolve::RoleMapping roles;
    roles.plot_id = args.get("--plot-id");
    roles.species_id = args.get("--species-id");
    roles.abundance = role_flag(args, "--abundance");
    roles.longitude = role_flag(args, "--lon");
    roles.latitude = role_flag(args, "--lat");
    roles.genus = role_flag(args, "--genus");
    roles.family = role_flag(args, "--family");
    roles.covariates = args.get_all("--covariate");
    roles.categoricals = args.get_all("--categorical");
    return roles;
}

// The same flags as renames of a suite's contract columns.
inline resolve::SuiteColumns read_suite_columns(const ParsedArgs& args) {
    resolve::SuiteColumns columns;
    columns.plot_id = role_flag(args, "--plot-id");
    columns.species_id = role_flag(args, "--species-id");
    columns.abundance = role_flag(args, "--abundance");
    columns.genus = role_flag(args, "--genus");
    columns.family = role_flag(args, "--family");
    columns.latitude = role_flag(args, "--lat");
    columns.longitude = role_flag(args, "--lon");
    if (args.has("--covariate")) columns.covariates = args.get_all("--covariate");
    if (args.has("--categorical")) columns.categoricals = args.get_all("--categorical");
    return columns;
}

// The device --device names (resolve::parse_device). A CUDA device the machine
// does not have is an error, not a quiet run on the CPU.
inline torch::Device choose_device(const ParsedArgs& args) {
    torch::Device device(torch::kCPU);
    try {
        device = resolve::parse_device(args.get("--device"));
    } catch (const std::invalid_argument& e) {
        throw ArgError(std::string("--device: ") + e.what());
    }
    if (device.is_cuda()) {
        const auto n_devices = static_cast<int>(torch::cuda::device_count());
        if (n_devices == 0) throw ArgError("--device: no CUDA device is available");
        if (device.has_index() && device.index() >= n_devices) {
            throw ArgError("--device: " + resolve::device_name(device) + " does not exist (" +
                           std::to_string(n_devices) + " CUDA device(s) present)");
        }
    }
    std::cout << "Using " << resolve::device_name(device) << std::endl;
    return device;
}

// The plots of a header and a species CSV, or of a species CSV alone when no
// header is given; in the namespace of `vocabs` when given (a trained model's
// vocabularies), otherwise with vocabularies fitted on this data.
inline resolve::ResolveDataset load_dataset(const std::string& header_path,
                                            const std::string& species_path,
                                            const resolve::RoleMapping& roles,
                                            const std::vector<resolve::TargetSpec>& targets,
                                            const resolve::DatasetConfig& config,
                                            const resolve::ExternalVocabs* vocabs = nullptr) {
    std::optional<resolve::TableSource> header;
    if (!header_path.empty()) header = resolve::TableSource::csv(header_path);
    return resolve::ResolveDataset::load(header, resolve::TableSource::csv(species_path), roles,
                                         targets, config, vocabs);
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_COMMON_HPP
