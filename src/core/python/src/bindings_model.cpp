#include "bindings_common.hpp"

void register_model(nb::module_& m) {
    nb::class_<resolve::ResolveModel>(m, "ResolveModel")
        .def(nb::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
             nb::arg("schema"), nb::arg("config") = resolve::ModelConfig{})
        .def("forward", [](resolve::ResolveModel& self,
                          torch::Tensor continuous,
                          torch::Tensor genus_ids,
                          torch::Tensor family_ids,
                          torch::Tensor species_ids,
                          torch::Tensor species_vector) {
            return tensor_map_to_dict(self->forward(continuous, genus_ids, family_ids, species_ids, species_vector));
        }, nb::arg("continuous"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("species_ids"),
           nb::arg("species_vector"))
        .def("get_latent", [](resolve::ResolveModel& self,
                              torch::Tensor continuous,
                              torch::Tensor genus_ids,
                              torch::Tensor family_ids,
                              torch::Tensor species_ids,
                              torch::Tensor species_vector) {
            return self->get_latent(continuous, genus_ids, family_ids, species_ids, species_vector);
        }, nb::arg("continuous"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("species_ids"),
           nb::arg("species_vector"))
        .def("train", [](resolve::ResolveModel& self, bool mode) { self->train(mode); }, nb::arg("mode") = true)
        .def("eval", [](resolve::ResolveModel& self) { self->eval(); })
        .def("to", [](resolve::ResolveModel& self, const std::string& device) {
            if (device == "cuda") {
                self->to(torch::kCUDA);
            } else {
                self->to(torch::kCPU);
            }
        })
        .def_prop_ro("schema", [](resolve::ResolveModel& self) { return self->schema(); })
        .def_prop_ro("config", [](resolve::ResolveModel& self) { return self->config(); })
        .def_prop_ro("latent_dim", [](resolve::ResolveModel& self) { return self->latent_dim(); })
        .def_prop_ro("species_encoding", [](resolve::ResolveModel& self) { return self->species_encoding(); })
        .def_prop_ro("uses_explicit_vector", [](resolve::ResolveModel& self) { return self->uses_explicit_vector(); });

    m.attr("SpaccModel") = m.attr("ResolveModel");
}
