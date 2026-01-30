#include "bindings_common.hpp"

void register_metrics(nb::module_& m) {
    nb::class_<resolve::ClassificationMetrics>(m, "ClassificationMetrics")
        .def(nb::init<>())
        .def_ro("accuracy", &resolve::ClassificationMetrics::accuracy)
        .def_ro("macro_f1", &resolve::ClassificationMetrics::macro_f1)
        .def_ro("weighted_f1", &resolve::ClassificationMetrics::weighted_f1)
        .def_ro("per_class_precision", &resolve::ClassificationMetrics::per_class_precision)
        .def_ro("per_class_recall", &resolve::ClassificationMetrics::per_class_recall)
        .def_ro("per_class_f1", &resolve::ClassificationMetrics::per_class_f1)
        .def_ro("per_class_support", &resolve::ClassificationMetrics::per_class_support)
        .def_ro("confusion_matrix", &resolve::ClassificationMetrics::confusion_matrix);

    nb::class_<resolve::ConfidenceMetrics>(m, "ConfidenceMetrics")
        .def(nb::init<>())
        .def_ro("accuracy", &resolve::ConfidenceMetrics::accuracy)
        .def_ro("coverage", &resolve::ConfidenceMetrics::coverage)
        .def_ro("n_samples", &resolve::ConfidenceMetrics::n_samples)
        .def_ro("n_total", &resolve::ConfidenceMetrics::n_total);

    nb::class_<resolve::Metrics>(m, "Metrics")
        .def_static("band_accuracy", &resolve::Metrics::band_accuracy,
                    nb::arg("pred"), nb::arg("target"), nb::arg("threshold") = 0.25f)
        .def_static("mae", &resolve::Metrics::mae)
        .def_static("rmse", &resolve::Metrics::rmse)
        .def_static("smape", &resolve::Metrics::smape,
                    nb::arg("pred"), nb::arg("target"), nb::arg("eps") = 1e-8f)
        .def_static("r_squared", &resolve::Metrics::r_squared,
                    nb::arg("pred"), nb::arg("target"))
        .def_static("accuracy", &resolve::Metrics::accuracy)
        .def_static("confusion_matrix", &resolve::Metrics::confusion_matrix,
                    nb::arg("pred"), nb::arg("target"), nb::arg("num_classes"))
        .def_static("classification_metrics", &resolve::Metrics::classification_metrics,
                    nb::arg("pred"), nb::arg("target"), nb::arg("num_classes"))
        .def_static("accuracy_at_threshold", &resolve::Metrics::accuracy_at_threshold,
                    nb::arg("pred"), nb::arg("target"), nb::arg("confidence"), nb::arg("threshold"))
        .def_static("accuracy_coverage_curve", &resolve::Metrics::accuracy_coverage_curve,
                    nb::arg("pred"), nb::arg("target"), nb::arg("confidence"),
                    nb::arg("thresholds") = std::vector<float>{0.0f, 0.5f, 0.8f, 0.9f, 0.95f})
        .def_static("compute", &resolve::Metrics::compute,
                    nb::arg("pred"), nb::arg("target"), nb::arg("task"),
                    nb::arg("transform") = resolve::TransformType::None,
                    nb::arg("band_thresholds") = std::vector<float>{0.25f, 0.50f, 0.75f},
                    nb::arg("num_classes") = 0);
}
