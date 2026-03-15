#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "occulus/features.hpp"
#include "occulus/filters.hpp"
#include "occulus/kdtree.hpp"
#include "occulus/normals.hpp"
#include "occulus/registration.hpp"
#include "occulus/segmentation.hpp"

namespace py = pybind11;

// std::vector<bool> is a bitfield — no .data(). Copy to uint8_t array.
static py::array_t<bool> bool_vec_to_array(const std::vector<bool>& v) {
    py::array_t<bool> arr(v.size());
    auto ptr = arr.mutable_data();
    for (size_t i = 0; i < v.size(); ++i) ptr[i] = v[i];
    return arr;
}

// --------------------------------------------------------------------------
// kdtree submodule
// --------------------------------------------------------------------------
void bind_kdtree(py::module_& m) {
    auto sub = m.def_submodule("kdtree", "k-d tree spatial queries");

    py::class_<occulus::KDTree>(sub, "KDTree",
        "3D k-d tree for radius and kNN queries.")
        .def(py::init([](py::array_t<double> pts) {
            auto buf = pts.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error("Expected (N, 3) array");
            return new occulus::KDTree(
                static_cast<const double*>(buf.ptr),
                static_cast<size_t>(buf.shape[0]));
        }), py::arg("points"))
        .def("radius_search", [](const occulus::KDTree& tree,
                                  py::array_t<double> query, double radius) {
            auto buf = query.request();
            if (buf.size != 3) throw std::runtime_error("Query must be length 3");
            const double* d = static_cast<const double*>(buf.ptr);
            occulus::Point3 q = {d[0], d[1], d[2]};
            return tree.radius_search(q, radius);
        }, py::arg("query"), py::arg("radius"))
        .def("knn_search", [](const occulus::KDTree& tree,
                               py::array_t<double> query, size_t k) {
            auto buf = query.request();
            if (buf.size != 3) throw std::runtime_error("Query must be length 3");
            const double* d = static_cast<const double*>(buf.ptr);
            occulus::Point3 q = {d[0], d[1], d[2]};
            return tree.knn_search(q, k);
        }, py::arg("query"), py::arg("k"))
        .def_property_readonly("size", &occulus::KDTree::size);
}

// --------------------------------------------------------------------------
// filters submodule
// --------------------------------------------------------------------------
void bind_filters(py::module_& m) {
    auto sub = m.def_submodule("filters", "Point cloud filtering");

    sub.def("voxel_downsample", [](py::array_t<double> pts, double voxel_size) {
        auto buf = pts.request();
        if (buf.ndim != 2 || buf.shape[1] != 3)
            throw std::runtime_error("Expected (N, 3) array");
        auto result = occulus::voxel_downsample(
            static_cast<const double*>(buf.ptr),
            static_cast<size_t>(buf.shape[0]),
            voxel_size);
        size_t m_pts = result.size() / 3;
        return py::array_t<double>({m_pts, size_t(3)}, result.data());
    }, py::arg("points"), py::arg("voxel_size"),
    "Voxel grid downsampling. Returns (M, 3) centroids.");

    sub.def("statistical_outlier_removal",
        [](py::array_t<double> pts, int nb_neighbors, double std_ratio) {
            auto buf = pts.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error("Expected (N, 3) array");
            auto mask = occulus::statistical_outlier_removal(
                static_cast<const double*>(buf.ptr),
                static_cast<size_t>(buf.shape[0]),
                nb_neighbors, std_ratio);
            return bool_vec_to_array(mask);
        }, py::arg("points"), py::arg("nb_neighbors"), py::arg("std_ratio"),
        "Statistical outlier removal. Returns boolean inlier mask.");

    sub.def("radius_outlier_removal",
        [](py::array_t<double> pts, double radius, int min_neighbors) {
            auto buf = pts.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error("Expected (N, 3) array");
            auto mask = occulus::radius_outlier_removal(
                static_cast<const double*>(buf.ptr),
                static_cast<size_t>(buf.shape[0]),
                radius, min_neighbors);
            return bool_vec_to_array(mask);
        }, py::arg("points"), py::arg("radius"), py::arg("min_neighbors"),
        "Radius outlier removal. Returns boolean inlier mask.");
}

// --------------------------------------------------------------------------
// normals submodule
// --------------------------------------------------------------------------
void bind_normals(py::module_& m) {
    auto sub = m.def_submodule("normals", "Normal estimation");

    sub.def("estimate_normals",
        [](py::array_t<double> pts, double radius, int max_nn) {
            auto buf = pts.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error("Expected (N, 3) array");
            size_t n = static_cast<size_t>(buf.shape[0]);
            auto result = occulus::estimate_normals(
                static_cast<const double*>(buf.ptr), n, radius, max_nn);
            return py::array_t<double>({n, size_t(3)}, result.data());
        }, py::arg("points"), py::arg("radius") = 0.1, py::arg("max_nn") = 30,
        "PCA normal estimation. Returns (N, 3) unit normals.");

    sub.def("orient_normals_to_viewpoint",
        [](py::array_t<double> normals, py::array_t<double> pts,
           double vx, double vy, double vz) {
            auto n_buf = normals.mutable_unchecked<2>();
            auto p_buf = pts.unchecked<2>();
            size_t n = static_cast<size_t>(n_buf.shape(0));
            occulus::orient_normals_to_viewpoint(
                n_buf.mutable_data(0, 0),
                p_buf.data(0, 0), n, vx, vy, vz);
        }, py::arg("normals"), py::arg("points"),
        py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("vz") = 1e6,
        "Orient normals toward a viewpoint (modifies in place).");
}

// --------------------------------------------------------------------------
// registration submodule
// --------------------------------------------------------------------------
void bind_registration(py::module_& m) {
    auto sub = m.def_submodule("registration", "ICP registration");

    py::class_<occulus::ICPResult>(sub, "ICPResult")
        .def_readonly("transformation", &occulus::ICPResult::transformation)
        .def_readonly("fitness", &occulus::ICPResult::fitness)
        .def_readonly("inlier_rmse", &occulus::ICPResult::inlier_rmse)
        .def_readonly("converged", &occulus::ICPResult::converged)
        .def_readonly("n_iterations", &occulus::ICPResult::n_iterations);

    sub.def("icp_point_to_point",
        [](py::array_t<double> source, py::array_t<double> target,
           int max_iter, double tol, double max_dist,
           py::object init_transform) {
            auto s = source.request();
            auto t = target.request();
            const double* init = nullptr;
            std::array<double, 16> init_buf;
            if (!init_transform.is_none()) {
                auto it = py::cast<py::array_t<double>>(init_transform);
                auto ibuf = it.request();
                std::memcpy(init_buf.data(), ibuf.ptr, 16 * sizeof(double));
                init = init_buf.data();
            }
            return occulus::icp_point_to_point(
                static_cast<const double*>(s.ptr), static_cast<size_t>(s.shape[0]),
                static_cast<const double*>(t.ptr), static_cast<size_t>(t.shape[0]),
                max_iter, tol, max_dist, init);
        },
        py::arg("source"), py::arg("target"),
        py::arg("max_iterations") = 50, py::arg("tolerance") = 1e-6,
        py::arg("max_distance") = 1.0, py::arg("init_transform") = py::none(),
        "Point-to-point ICP registration.");
}

// --------------------------------------------------------------------------
// segmentation submodule
// --------------------------------------------------------------------------
void bind_segmentation(py::module_& m) {
    auto sub = m.def_submodule("segmentation", "Ground classification");

    sub.def("classify_ground_csf",
        [](py::array_t<double> pts, double cloth_res, double threshold, int max_iter) {
            auto buf = pts.request();
            size_t n = static_cast<size_t>(buf.shape[0]);
            auto result = occulus::classify_ground_csf(
                static_cast<const double*>(buf.ptr), n,
                cloth_res, threshold, max_iter);
            return py::array_t<int>(n, result.data());
        },
        py::arg("points"), py::arg("cloth_resolution") = 1.0,
        py::arg("class_threshold") = 0.5, py::arg("max_iterations") = 500,
        "CSF ground classification. Returns int array (2=ground, 1=other).");

    sub.def("classify_ground_pmf",
        [](py::array_t<double> pts, int max_window, double slope,
           double initial_dist, double max_dist, double cell_size) {
            auto buf = pts.request();
            size_t n = static_cast<size_t>(buf.shape[0]);
            auto result = occulus::classify_ground_pmf(
                static_cast<const double*>(buf.ptr), n,
                max_window, slope, initial_dist, max_dist, cell_size);
            return py::array_t<int>(n, result.data());
        },
        py::arg("points"), py::arg("max_window") = 18,
        py::arg("slope") = 1.0, py::arg("initial_dist") = 0.15,
        py::arg("max_dist") = 2.5, py::arg("cell_size") = 1.0,
        "PMF ground classification. Returns int array (2=ground, 1=other).");
}

// --------------------------------------------------------------------------
// features submodule
// --------------------------------------------------------------------------
void bind_features(py::module_& m) {
    auto sub = m.def_submodule("features", "Geometric feature extraction");

    sub.def("detect_plane_ransac",
        [](py::array_t<double> pts, double dist_thresh, int max_iter) {
            auto buf = pts.request();
            size_t n = static_cast<size_t>(buf.shape[0]);
            auto result = occulus::detect_plane_ransac(
                static_cast<const double*>(buf.ptr), n, dist_thresh, max_iter);
            py::dict out;
            out["model"] = py::array_t<double>(4, result.model.data());
            out["inliers"] = bool_vec_to_array(result.inliers);
            out["score"] = result.score;
            return out;
        },
        py::arg("points"), py::arg("distance_threshold") = 0.01,
        py::arg("max_iterations") = 1000,
        "RANSAC plane detection.");

    sub.def("compute_geometric_features",
        [](py::array_t<double> pts, double radius) {
            auto buf = pts.request();
            size_t n = static_cast<size_t>(buf.shape[0]);
            auto result = occulus::compute_geometric_features(
                static_cast<const double*>(buf.ptr), n, radius);
            return py::array_t<double>({n, size_t(7)}, result.data());
        },
        py::arg("points"), py::arg("radius") = 0.5,
        "Eigenvalue geometric features (N, 7): linearity, planarity, "
        "sphericity, omnivariance, anisotropy, eigenentropy, curvature.");
}

// --------------------------------------------------------------------------
// mesh submodule (placeholder — delegates to Open3D in Python)
// --------------------------------------------------------------------------
void bind_mesh(py::module_& m) {
    m.def_submodule("mesh",
        "Surface reconstruction (Poisson, BPA) — implemented via Open3D in Python.");
}

// --------------------------------------------------------------------------
// Module definition
// --------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "Occulus C++ backend — accelerated point cloud operations.";

    bind_kdtree(m);
    bind_filters(m);
    bind_normals(m);
    bind_registration(m);
    bind_segmentation(m);
    bind_features(m);
    bind_mesh(m);
}
