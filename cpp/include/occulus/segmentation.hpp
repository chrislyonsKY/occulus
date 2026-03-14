#pragma once

/// @file segmentation.hpp
/// @brief Ground classification — CSF and PMF algorithms.

#include <cstddef>
#include <vector>

namespace occulus {

/// Cloth Simulation Filter (CSF) ground classification.
///
/// Simulates a rigid cloth draped over an inverted point cloud to
/// separate ground from non-ground returns. Returns ASPRS class codes
/// (2 = ground, 1 = unclassified).
///
/// @param data              Flat (N*3) xyz array.
/// @param n                 Number of points.
/// @param cloth_resolution  Grid resolution of the simulated cloth.
/// @param class_threshold   Distance threshold to classify as ground.
/// @param max_iterations    Maximum cloth simulation iterations.
/// @return Vector of N classification codes (2=ground, 1=other).
std::vector<int> classify_ground_csf(const double* data, size_t n,
                                     double cloth_resolution,
                                     double class_threshold,
                                     int max_iterations);

/// Progressive Morphological Filter (PMF) ground classification.
///
/// Applies iteratively increasing morphological opening operations to
/// separate ground from elevated objects. Well-suited for flat to
/// moderate terrain.
///
/// @param data          Flat (N*3) xyz array.
/// @param n             Number of points.
/// @param max_window    Maximum window size (in grid cells).
/// @param slope         Terrain slope parameter (degrees).
/// @param initial_dist  Initial elevation difference threshold.
/// @param max_dist      Maximum elevation difference threshold.
/// @param cell_size     Grid cell size for rasterization.
/// @return Vector of N classification codes (2=ground, 1=other).
std::vector<int> classify_ground_pmf(const double* data, size_t n,
                                     int max_window, double slope,
                                     double initial_dist, double max_dist,
                                     double cell_size);

}  // namespace occulus
