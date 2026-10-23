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

// CUDA with --cuda, the CPU otherwise. Asking for CUDA on a machine without a
// usable device is an error, not a quiet run on the CPU.
inline torch::Device choose_device(const ParsedArgs& args) {
    if (!args.has("--cuda")) {
        std::cout << "Using CPU" << std::endl;
        return torch::kCPU;
    }
    if (!torch::cuda::is_available()) {
        throw ArgError("--cuda: no CUDA device is available");
    }
    std::cout << "Using CUDA" << std::endl;
    return torch::kCUDA;
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
    using resolve::ResolveDataset;
    if (header_path.empty()) {
        return vocabs != nullptr
                   ? ResolveDataset::from_species_csv_with_vocabs(species_path, roles, targets,
                                                                  *vocabs, config)
                   : ResolveDataset::from_species_csv(species_path, roles, targets, config);
    }
    return vocabs != nullptr
               ? ResolveDataset::from_csv_with_vocabs(header_path, species_path, roles, targets,
                                                      *vocabs, config)
               : ResolveDataset::from_csv(header_path, species_path, roles, targets, config);
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_COMMON_HPP
