#include "occulus/features.hpp"
#include "occulus/kdtree.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace occulus {

PlaneResult detect_plane_ransac(const double* data, size_t n,
                                double distance_threshold,
                                int max_iterations) {
    PlaneResult best;
    best.score = 0;
    best.inliers.resize(n, false);

    if (n < 3) return best;

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Sample 3 random points
        size_t i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
        if (i0 == i1 || i0 == i2 || i1 == i2) continue;

        double ax = data[i0 * 3], ay = data[i0 * 3 + 1], az = data[i0 * 3 + 2];
        double bx = data[i1 * 3], by = data[i1 * 3 + 1], bz = data[i1 * 3 + 2];
        double cx = data[i2 * 3], cy = data[i2 * 3 + 1], cz = data[i2 * 3 + 2];

        // Compute plane normal via cross product
        double ux = bx - ax, uy = by - ay, uz = bz - az;
        double vx = cx - ax, vy = cy - ay, vz = cz - az;
        double nx = uy * vz - uz * vy;
        double ny = uz * vx - ux * vz;
        double nz = ux * vy - uy * vx;
        double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-12) continue;
        nx /= len; ny /= len; nz /= len;
        double d = -(nx * ax + ny * ay + nz * az);

        // Count inliers
        int count = 0;
        for (size_t i = 0; i < n; ++i) {
            double dist_val = std::abs(nx * data[i * 3] + ny * data[i * 3 + 1] +
                                       nz * data[i * 3 + 2] + d);
            if (dist_val <= distance_threshold) count++;
        }

        double score = static_cast<double>(count) / static_cast<double>(n);
        if (score > best.score) {
            best.score = score;
            best.model = {nx, ny, nz, d};
            for (size_t i = 0; i < n; ++i) {
                double dist_val = std::abs(nx * data[i * 3] + ny * data[i * 3 + 1] +
                                           nz * data[i * 3 + 2] + d);
                best.inliers[i] = dist_val <= distance_threshold;
            }
        }
    }
    return best;
}

std::vector<double> compute_geometric_features(const double* data, size_t n,
                                               double radius) {
    KDTree tree(data, n);
    // 7 features per point: linearity, planarity, sphericity,
    //                        omnivariance, anisotropy, eigenentropy, curvature
    std::vector<double> features(n * 7, 0.0);

    for (size_t i = 0; i < n; ++i) {
        Point3 query = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        auto neighbors = tree.radius_search(query, radius);

        if (neighbors.size() < 3) continue;

        // Centroid
        double cx = 0, cy = 0, cz = 0;
        for (const auto& [idx, _] : neighbors) {
            cx += data[idx * 3];
            cy += data[idx * 3 + 1];
            cz += data[idx * 3 + 2];
        }
        double inv_n = 1.0 / static_cast<double>(neighbors.size());
        cx *= inv_n; cy *= inv_n; cz *= inv_n;

        // Covariance (upper triangle)
        double cov[6] = {};
        for (const auto& [idx, _] : neighbors) {
            double dx = data[idx * 3] - cx;
            double dy = data[idx * 3 + 1] - cy;
            double dz = data[idx * 3 + 2] - cz;
            cov[0] += dx * dx; cov[1] += dx * dy; cov[2] += dx * dz;
            cov[3] += dy * dy; cov[4] += dy * dz; cov[5] += dz * dz;
        }

        // Eigenvalues via characteristic equation of 3x3 symmetric
        double a11 = cov[0], a12 = cov[1], a13 = cov[2];
        double a22 = cov[3], a23 = cov[4], a33 = cov[5];

        double p1 = a12 * a12 + a13 * a13 + a23 * a23;
        double q = (a11 + a22 + a33) / 3.0;
        double p2 = (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) +
                    (a33 - q) * (a33 - q) + 2 * p1;
        double p = std::sqrt(p2 / 6.0);

        double e1, e2, e3;  // eigenvalues sorted descending
        if (p < 1e-15) {
            e1 = e2 = e3 = q;
        } else {
            double inv_p = 1.0 / p;
            double b11 = inv_p * (a11 - q), b12 = inv_p * a12, b13 = inv_p * a13;
            double b22 = inv_p * (a22 - q), b23 = inv_p * a23;
            double b33 = inv_p * (a33 - q);

            double det_b = b11 * (b22 * b33 - b23 * b23)
                          - b12 * (b12 * b33 - b13 * b23)
                          + b13 * (b12 * b23 - b13 * b22);
            double r = det_b / 2.0;
            double phi;
            if (r <= -1) phi = M_PI / 3.0;
            else if (r >= 1) phi = 0;
            else phi = std::acos(r) / 3.0;

            e1 = q + 2 * p * std::cos(phi);
            e3 = q + 2 * p * std::cos(phi + 2 * M_PI / 3.0);
            e2 = 3 * q - e1 - e3;

            // Sort descending
            if (e1 < e2) std::swap(e1, e2);
            if (e1 < e3) std::swap(e1, e3);
            if (e2 < e3) std::swap(e2, e3);
        }

        // Clamp eigenvalues to non-negative
        e1 = std::max(e1, 0.0);
        e2 = std::max(e2, 0.0);
        e3 = std::max(e3, 0.0);
        double sum_e = e1 + e2 + e3;

        if (sum_e > 1e-15 && e1 > 1e-15) {
            features[i * 7 + 0] = (e1 - e2) / e1;                          // linearity
            features[i * 7 + 1] = (e2 - e3) / e1;                          // planarity
            features[i * 7 + 2] = e3 / e1;                                 // sphericity
            features[i * 7 + 3] = std::cbrt(e1 * e2 * e3);                 // omnivariance
            features[i * 7 + 4] = (e1 - e3) / e1;                          // anisotropy

            // eigenentropy
            double entropy = 0;
            for (double e : {e1, e2, e3}) {
                double p_e = e / sum_e;
                if (p_e > 1e-15) entropy -= p_e * std::log(p_e);
            }
            features[i * 7 + 5] = entropy;

            features[i * 7 + 6] = e3 / sum_e;                              // curvature
        }
    }
    return features;
}

}  // namespace occulus
