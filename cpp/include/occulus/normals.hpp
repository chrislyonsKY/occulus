#pragma once

/// @file normals.hpp
/// @brief PCA-based normal estimation for 3D point clouds.

#include <cstddef>
#include <vector>

namespace occulus {

/// Estimate surface normals via PCA on local neighborhoods.
///
/// For each point, finds neighbors within `radius` (or at most `max_nn`)
/// using a k-d tree, then computes the eigenvector corresponding to the
/// smallest eigenvalue of the local covariance matrix.
///
/// @param data    Flat (N*3) xyz array.
/// @param n       Number of points.
/// @param radius  Neighborhood search radius.
/// @param max_nn  Maximum number of neighbors to use.
/// @return Flat (N*3) array of unit normals.
std::vector<double> estimate_normals(const double* data, size_t n,
                                     double radius, int max_nn);

/// Orient normals toward a viewpoint.
///
/// Flips each normal so that it points toward the given viewpoint.
///
/// @param normals  Flat (N*3) normal array (modified in place).
/// @param data     Flat (N*3) xyz array.
/// @param n        Number of points.
/// @param vx, vy, vz  Viewpoint coordinates.
void orient_normals_to_viewpoint(double* normals, const double* data,
                                 size_t n,
                                 double vx, double vy, double vz);

}  // namespace occulus
