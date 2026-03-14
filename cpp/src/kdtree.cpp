#include "occulus/kdtree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace occulus {

KDTree::KDTree(const double* data, size_t n) : n_points_(n) {
    points_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        points_[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
    }

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    nodes_.reserve(n);
    build_(indices, 0, 0, n);
}

int KDTree::build_(std::vector<size_t>& indices, int depth,
                   size_t lo, size_t hi) {
    if (lo >= hi) return -1;

    int dim = depth % 3;
    size_t mid = (lo + hi) / 2;

    std::nth_element(
        indices.begin() + static_cast<long>(lo),
        indices.begin() + static_cast<long>(mid),
        indices.begin() + static_cast<long>(hi),
        [&](size_t a, size_t b) {
            return points_[a][dim] < points_[b][dim];
        });

    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back({indices[mid], dim, -1, -1});

    nodes_[node_idx].left = build_(indices, depth + 1, lo, mid);
    nodes_[node_idx].right = build_(indices, depth + 1, mid + 1, hi);

    return node_idx;
}

std::vector<std::pair<size_t, double>> KDTree::radius_search(
    const Point3& query, double radius) const {
    std::vector<std::pair<size_t, double>> results;
    if (!nodes_.empty()) {
        radius_search_(0, query, radius * radius, results);
    }
    return results;
}

void KDTree::radius_search_(
    int node_idx, const Point3& query, double radius_sq,
    std::vector<std::pair<size_t, double>>& results) const {
    if (node_idx < 0) return;

    const auto& node = nodes_[node_idx];
    const auto& pt = points_[node.index];

    double dx = pt[0] - query[0];
    double dy = pt[1] - query[1];
    double dz = pt[2] - query[2];
    double dist_sq = dx * dx + dy * dy + dz * dz;

    if (dist_sq <= radius_sq) {
        results.emplace_back(node.index, dist_sq);
    }

    double diff = query[node.split_dim] - pt[node.split_dim];
    double diff_sq = diff * diff;

    int first = diff < 0 ? node.left : node.right;
    int second = diff < 0 ? node.right : node.left;

    radius_search_(first, query, radius_sq, results);
    if (diff_sq <= radius_sq) {
        radius_search_(second, query, radius_sq, results);
    }
}

std::vector<std::pair<size_t, double>> KDTree::knn_search(
    const Point3& query, size_t k) const {
    // Max-heap: largest distance on top
    std::vector<std::pair<size_t, double>> heap;
    if (!nodes_.empty()) {
        knn_search_(0, query, k, heap);
    }
    // Sort by distance ascending
    std::sort(heap.begin(), heap.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    return heap;
}

void KDTree::knn_search_(
    int node_idx, const Point3& query, size_t k,
    std::vector<std::pair<size_t, double>>& heap) const {
    if (node_idx < 0) return;

    const auto& node = nodes_[node_idx];
    const auto& pt = points_[node.index];

    double dx = pt[0] - query[0];
    double dy = pt[1] - query[1];
    double dz = pt[2] - query[2];
    double dist_sq = dx * dx + dy * dy + dz * dz;

    if (heap.size() < k) {
        heap.emplace_back(node.index, dist_sq);
        std::push_heap(heap.begin(), heap.end(),
                       [](const auto& a, const auto& b) { return a.second < b.second; });
    } else if (dist_sq < heap.front().second) {
        std::pop_heap(heap.begin(), heap.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
        heap.back() = {node.index, dist_sq};
        std::push_heap(heap.begin(), heap.end(),
                       [](const auto& a, const auto& b) { return a.second < b.second; });
    }

    double diff = query[node.split_dim] - pt[node.split_dim];
    double diff_sq = diff * diff;

    int first = diff < 0 ? node.left : node.right;
    int second = diff < 0 ? node.right : node.left;

    knn_search_(first, query, k, heap);

    double worst = heap.size() < k
                       ? std::numeric_limits<double>::max()
                       : heap.front().second;
    if (diff_sq < worst) {
        knn_search_(second, query, k, heap);
    }
}

}  // namespace occulus
