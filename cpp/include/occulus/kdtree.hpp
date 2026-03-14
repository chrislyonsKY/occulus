#pragma once

/// @file kdtree.hpp
/// @brief k-d tree for fast spatial queries on 3D point clouds.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace occulus {

/// A 3D point (x, y, z).
using Point3 = std::array<double, 3>;

/// Simple k-d tree for 3D point clouds.
///
/// Supports radius search and k-nearest-neighbor queries.
/// Construction is O(N log N), queries are O(log N) expected.
class KDTree {
public:
    /// Build a k-d tree from N points stored in a flat (N*3) array.
    ///
    /// @param data  Pointer to contiguous x,y,z,x,y,z,... data.
    /// @param n     Number of points.
    explicit KDTree(const double* data, size_t n);

    /// Find all points within `radius` of `query`.
    ///
    /// @param query   3D query point.
    /// @param radius  Search radius.
    /// @return Vector of (index, squared_distance) pairs.
    std::vector<std::pair<size_t, double>> radius_search(
        const Point3& query, double radius) const;

    /// Find the k nearest neighbors to `query`.
    ///
    /// @param query  3D query point.
    /// @param k      Number of neighbors.
    /// @return Vector of (index, squared_distance) pairs, sorted by distance.
    std::vector<std::pair<size_t, double>> knn_search(
        const Point3& query, size_t k) const;

    /// Number of points in the tree.
    size_t size() const { return n_points_; }

private:
    struct Node {
        size_t index;           // index into original point array
        int split_dim;          // 0=x, 1=y, 2=z
        int left  = -1;        // child node indices (-1 = leaf)
        int right = -1;
    };

    std::vector<Point3> points_;
    std::vector<Node> nodes_;
    size_t n_points_;

    int build_(std::vector<size_t>& indices, int depth, size_t lo, size_t hi);

    void radius_search_(int node_idx, const Point3& query,
                        double radius_sq,
                        std::vector<std::pair<size_t, double>>& results) const;

    void knn_search_(int node_idx, const Point3& query, size_t k,
                     std::vector<std::pair<size_t, double>>& heap) const;
};

}  // namespace occulus
