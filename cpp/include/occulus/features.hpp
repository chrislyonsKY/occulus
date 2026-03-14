#pragma once

/// @file features.hpp
/// @brief Geometric feature extraction — RANSAC plane/cylinder fitting.

#include <array>
#include <cstddef>
#include <vector>

namespace occulus {

/// Result of a RANSAC plane fit.
struct PlaneResult {
    std::array<double, 4> model;  ///< Plane equation: ax + by + cz + d = 0
    std::vector<bool> inliers;    ///< Boolean mask of inlier points
    double score;                 ///< Fraction of inliers
};

/// RANSAC plane detection.
///
/// Finds the dominant plane in the point cloud using RANSAC.
///
/// @param data           Flat (N*3) xyz array.
/// @param n              Number of points.
/// @param distance_threshold  Max distance from plane to count as inlier.
/// @param max_iterations      Number of RANSAC iterations.
/// @return PlaneResult with the best-fit plane model and inlier mask.
PlaneResult detect_plane_ransac(const double* data, size_t n,
                                double distance_threshold,
                                int max_iterations);

/// Compute eigenvalue-based geometric features.
///
/// For each point, computes the covariance matrix of its local neighborhood
/// and extracts eigenvalue ratios: linearity, planarity, sphericity,
/// omnivariance, anisotropy, eigenentropy, and curvature.
///
/// @param data    Flat (N*3) xyz array.
/// @param n       Number of points.
/// @param radius  Neighborhood search radius.
/// @return Flat (N*7) array: [linearity, planarity, sphericity,
///         omnivariance, anisotropy, eigenentropy, curvature] per point.
std::vector<double> compute_geometric_features(const double* data, size_t n,
                                               double radius);

}  // namespace occulus
