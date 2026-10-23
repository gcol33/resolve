#pragma once

// Python calls that carry model inputs. Every such method is bound as
// (self, *args, **kwargs) and reads its arguments through CallArgs, which
// builds the PlotInputs / ContinuousInputs from the engine's own field lists:
// a new model input reaches every binding without an edit here.

#include "bindings_common.hpp"

#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

// The PlotInputs field names, in positional order.
inline std::vector<std::string> plot_input_names() {
    std::vector<std::string> names;
    for (const auto& field : resolve::kPlotInputFields) names.emplace_back(field.name);
    return names;
}

// The PlotInputs field names except `continuous`, then nothing else: the
// model inputs of a call that passes the continuous block as its parts.
inline std::vector<std::string> species_input_names() {
    std::vector<std::string> names;
    for (const auto& field : resolve::kPlotInputFields) {
        if (std::string(field.name) != "continuous") names.emplace_back(field.name);
    }
    return names;
}

// The ContinuousInputs field names.
inline std::vector<std::string> continuous_part_names() {
    std::vector<std::string> names;
    resolve::ContinuousInputs{}.for_each(
        [&](const char* name, const torch::Tensor&) { names.emplace_back(name); });
    return names;
}

// A docstring that outlives the module: nanobind keeps the pointer it is
// given, so the text is stored here for the life of the process.
inline const char* persistent_doc(std::string text) {
    static std::deque<std::string> store;
    store.push_back(std::move(text));
    return store.back().c_str();
}

// "name=None, ..." for a list of optional arguments; the first `n_required`
// carry no default.
inline std::string argument_list(const std::vector<std::string>& names,
                                 std::size_t n_required = 0) {
    std::string out;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out += ", ";
        out += names[i];
        if (i >= n_required) out += "=None";
    }
    return out;
}

// One Python call's arguments by name. Positional arguments bind to
// `positional` in order; keywords name any accepted argument. Reading an
// argument consumes it, and finish() raises TypeError for any argument the
// method did not read, so a misspelt keyword is an error rather than an input
// silently left out.
class CallArgs {
public:
    CallArgs(std::string method, const nb::args& args, const nb::kwargs& kwargs,
             const std::vector<std::string>& positional)
        : method_(std::move(method)) {
        if (args.size() > positional.size()) {
            throw nb::type_error((method_ + "() takes at most " +
                                  std::to_string(positional.size()) +
                                  " positional arguments (" +
                                  std::to_string(args.size()) + " given)").c_str());
        }
        for (std::size_t i = 0; i < args.size(); ++i) {
            values_[positional[i]] = nb::borrow(
                nb::handle(PyTuple_GET_ITEM(args.ptr(), static_cast<Py_ssize_t>(i))));
        }
        for (auto item : kwargs) {
            auto key = nb::cast<std::string>(item.first);
            if (values_.count(key) != 0) {
                throw nb::type_error((method_ + "() got multiple values for argument '" +
                                      key + "'").c_str());
            }
            values_[key] = nb::borrow(item.second);
        }
    }

    // The argument `name`, or None when the call did not pass it.
    nb::object take(const std::string& name) {
        auto it = values_.find(name);
        if (it == values_.end()) return nb::none();
        nb::object out = it->second;
        values_.erase(it);
        return out;
    }

    template <typename T>
    T take(const std::string& name, T default_value) {
        nb::object obj = take(name);
        return obj.is_none() ? default_value : nb::cast<T>(obj);
    }

    // A required argument.
    template <typename T>
    T require(const std::string& name) {
        nb::object obj = take(name);
        if (obj.is_none()) {
            throw nb::type_error((method_ + "() missing required argument '" + name +
                                  "'").c_str());
        }
        return nb::cast<T>(obj);
    }

    // The PlotInputs fields the call passed. `continuous` is read only when
    // `with_continuous`, and is then required.
    resolve::PlotInputs plot_inputs(bool with_continuous) {
        resolve::PlotInputs inputs;
        inputs.for_each([&](const resolve::PlotInputField& field, torch::Tensor& t) {
            const std::string name = field.name;
            if (name == "continuous") {
                if (with_continuous) t = unpack_required_tensor(take(name), field.name);
            } else {
                t = optional_tensor(name);
            }
        });
        return inputs;
    }

    // The ContinuousInputs parts the call passed.
    resolve::ContinuousInputs continuous_parts() {
        resolve::ContinuousInputs parts;
        parts.for_each([&](const char* name, torch::Tensor& t) { t = optional_tensor(name); });
        return parts;
    }

    void finish() const {
        if (values_.empty()) return;
        std::string names;
        for (const auto& [name, value] : values_) {
            if (!names.empty()) names += ", ";
            names += "'" + name + "'";
        }
        throw nb::type_error((method_ + "() got unexpected argument(s) " + names).c_str());
    }

private:
    // An optional tensor argument: undefined for None or absent, an error for
    // anything that is not a tensor.
    at::Tensor optional_tensor(const std::string& name) {
        nb::object obj = take(name);
        return obj.is_none() ? at::Tensor() : unpack_required_tensor(obj, name.c_str());
    }

    std::string method_;
    std::unordered_map<std::string, nb::object> values_;
};
