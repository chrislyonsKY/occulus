#pragma once

/// @file filters.hpp
/// @brief Point cloud filtering — voxel downsample, SOR, radius outlier.

#include <cstddef>
#include <vector>

namespace occulus {

/// Voxel grid downsampling.
///
/// Assigns each point to a voxel cell and keeps one representative point
/// per occupied cell (the centroid of all points in that cell).
///
/// @param data       Flat (N*3) xyz array.
/// @param n          Number of input points.
/// @param voxel_size Edge length of cubic voxels.
/// @return Flat (M*3) array of downsampled centroids.
std::vector<double> voxel_downsample(const double* data, size_t n,
                                     double voxel_size);

/// Statistical outlier removal.
///
/// For each point, computes the mean distance to its k nearest neighbors.
/// Points whose mean distance exceeds (global_mean + std_ratio * global_std)
/// are marked as outliers.
///
/// @param data        Flat (N*3) xyz array.
/// @param n           Number of input points.
/// @param nb_neighbors Number of neighbors (k).
/// @param std_ratio   Standard deviation multiplier threshold.
/// @return Boolean mask (N elements) — true = inlier, false = outlier.
std::vector<bool> statistical_outlier_removal(const double* data, size_t n,
                                              int nb_neighbors,
                                              double std_ratio);

/// Radius outlier removal.
///
/// For each point, counts neighbors within `radius`. Points with fewer
/// than `min_neighbors` are marked as outliers.
///
/// @param data          Flat (N*3) xyz array.
/// @param n             Number of input points.
/// @param radius        Search radius.
/// @param min_neighbors Minimum neighbor count to keep.
/// @return Boolean mask (N elements) — true = inlier, false = outlier.
std::vector<bool> radius_outlier_removal(const double* data, size_t n,
                                         double radius, int min_neighbors);

}  // namespace occulus
