#include "occulus/filters.hpp"
#include "occulus/kdtree.hpp"

#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>

namespace occulus {

// Hash for voxel grid keys
struct VoxelHash {
    size_t operator()(const std::array<int, 3>& v) const {
        size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

std::vector<double> voxel_downsample(const double* data, size_t n,
                                     double voxel_size) {
    double inv = 1.0 / voxel_size;

    // Accumulate points per voxel cell
    struct Accum {
        double sx = 0, sy = 0, sz = 0;
        size_t count = 0;
    };
    std::unordered_map<std::array<int, 3>, Accum, VoxelHash> grid;

    for (size_t i = 0; i < n; ++i) {
        double x = data[i * 3];
        double y = data[i * 3 + 1];
        double z = data[i * 3 + 2];
        std::array<int, 3> key = {
            static_cast<int>(std::floor(x * inv)),
            static_cast<int>(std::floor(y * inv)),
            static_cast<int>(std::floor(z * inv)),
        };
        auto& acc = grid[key];
        acc.sx += x;
        acc.sy += y;
        acc.sz += z;
        acc.count++;
    }

    std::vector<double> result;
    result.reserve(grid.size() * 3);
    for (const auto& [key, acc] : grid) {
        double inv_count = 1.0 / static_cast<double>(acc.count);
        result.push_back(acc.sx * inv_count);
        result.push_back(acc.sy * inv_count);
        result.push_back(acc.sz * inv_count);
    }
    return result;
}

std::vector<bool> statistical_outlier_removal(const double* data, size_t n,
                                              int nb_neighbors,
                                              double std_ratio) {
    KDTree tree(data, n);

    // Compute mean neighbor distances
    std::vector<double> mean_dists(n);
    for (size_t i = 0; i < n; ++i) {
        Point3 query = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        auto neighbors = tree.knn_search(query, static_cast<size_t>(nb_neighbors) + 1);

        double sum = 0;
        int count = 0;
        for (const auto& [idx, dist_sq] : neighbors) {
            if (idx != i) {
                sum += std::sqrt(dist_sq);
                count++;
            }
        }
        mean_dists[i] = count > 0 ? sum / count : 0;
    }

    // Global statistics
    double global_mean = std::accumulate(mean_dists.begin(), mean_dists.end(), 0.0)
                         / static_cast<double>(n);
    double variance = 0;
    for (double d : mean_dists) {
        double diff = d - global_mean;
        variance += diff * diff;
    }
    double global_std = std::sqrt(variance / static_cast<double>(n));
    double threshold = global_mean + std_ratio * global_std;

    std::vector<bool> mask(n);
    for (size_t i = 0; i < n; ++i) {
        mask[i] = mean_dists[i] <= threshold;
    }
    return mask;
}

std::vector<bool> radius_outlier_removal(const double* data, size_t n,
                                         double radius, int min_neighbors) {
    KDTree tree(data, n);

    std::vector<bool> mask(n);
    for (size_t i = 0; i < n; ++i) {
        Point3 query = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        auto neighbors = tree.radius_search(query, radius);
        // Subtract self
        int count = static_cast<int>(neighbors.size()) - 1;
        mask[i] = count >= min_neighbors;
    }
    return mask;
}

}  // namespace occulus
