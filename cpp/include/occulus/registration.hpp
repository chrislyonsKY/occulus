#pragma once

/// @file registration.hpp
/// @brief ICP point cloud registration (point-to-point and point-to-plane).

#include <array>
#include <cstddef>
#include <vector>

namespace occulus {

/// Result of an ICP registration.
struct ICPResult {
    std::array<double, 16> transformation;  ///< 4x4 rigid transform (row-major)
    double fitness;                         ///< Fraction of source points with correspondences
    double inlier_rmse;                     ///< RMS error of inlier correspondences
    bool converged;                         ///< Whether convergence tolerance was met
    int n_iterations;                       ///< Number of iterations executed
};

/// Point-to-point ICP.
///
/// Iteratively aligns `source` to `target` by minimizing the sum of
/// squared distances between corresponding point pairs (closest-point).
///
/// @param source          Flat (Ns*3) source points.
/// @param ns              Number of source points.
/// @param target          Flat (Nt*3) target points.
/// @param nt              Number of target points.
/// @param max_iterations  Maximum number of ICP iterations.
/// @param tolerance       Convergence threshold (change in RMSE).
/// @param max_dist        Maximum correspondence distance.
/// @param init_transform  Optional 4x4 initial transform (nullptr = identity).
/// @return ICPResult with the estimated rigid transformation.
ICPResult icp_point_to_point(const double* source, size_t ns,
                             const double* target, size_t nt,
                             int max_iterations, double tolerance,
                             double max_dist,
                             const double* init_transform);

/// Point-to-plane ICP.
///
/// Requires target normals. Minimizes the point-to-plane distance
/// metric, which converges faster than point-to-point on smooth surfaces.
///
/// @param source          Flat (Ns*3) source points.
/// @param ns              Number of source points.
/// @param target          Flat (Nt*3) target points.
/// @param target_normals  Flat (Nt*3) target normals.
/// @param nt              Number of target points.
/// @param max_iterations  Maximum number of ICP iterations.
/// @param tolerance       Convergence threshold.
/// @param max_dist        Maximum correspondence distance.
/// @param init_transform  Optional 4x4 initial transform (nullptr = identity).
/// @return ICPResult with the estimated rigid transformation.
ICPResult icp_point_to_plane(const double* source, size_t ns,
                             const double* target, size_t nt,
                             const double* target_normals,
                             int max_iterations, double tolerance,
                             double max_dist,
                             const double* init_transform);

}  // namespace occulus
