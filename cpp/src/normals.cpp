#include "occulus/normals.hpp"
#include "occulus/kdtree.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace occulus {

namespace {

/// Compute eigenvectors of a 3x3 symmetric matrix via Jacobi iteration.
/// Returns eigenvalues sorted ascending and corresponding eigenvectors.
void eigen_3x3_symmetric(const double cov[6],
                         double eigenvalues[3],
                         double eigenvectors[9]) {
    // cov stored as [xx, xy, xz, yy, yz, zz]
    // Build full symmetric matrix
    double a[3][3] = {
        {cov[0], cov[1], cov[2]},
        {cov[1], cov[3], cov[4]},
        {cov[2], cov[4], cov[5]},
    };

    // Initialize eigenvectors to identity
    double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    // Jacobi rotation — 20 sweeps is more than enough for 3x3
    for (int sweep = 0; sweep < 20; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(a[p][q]) < 1e-15) continue;

                double tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                double t = (tau >= 0 ? 1.0 : -1.0)
                           / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                double a_pq = a[p][q];
                a[p][q] = 0;
                a[p][p] -= t * a_pq;
                a[q][q] += t * a_pq;

                for (int r = 0; r < 3; ++r) {
                    if (r != p && r != q) {
                        double a_rp = a[r][p];
                        double a_rq = a[r][q];
                        a[r][p] = a[p][r] = c * a_rp - s * a_rq;
                        a[r][q] = a[q][r] = s * a_rp + c * a_rq;
                    }
                }

                for (int r = 0; r < 3; ++r) {
                    double v_rp = v[r][p];
                    double v_rq = v[r][q];
                    v[r][p] = c * v_rp - s * v_rq;
                    v[r][q] = s * v_rp + c * v_rq;
                }
            }
        }
    }

    // Sort eigenvalues ascending
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3,
              [&](int i, int j) { return a[i][i] < a[j][j]; });

    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = a[order[i]][order[i]];
        for (int r = 0; r < 3; ++r) {
            eigenvectors[i * 3 + r] = v[r][order[i]];
        }
    }
}

}  // anonymous namespace

std::vector<double> estimate_normals(const double* data, size_t n,
                                     double radius, int max_nn) {
    KDTree tree(data, n);
    std::vector<double> normals(n * 3, 0.0);

    for (size_t i = 0; i < n; ++i) {
        Point3 query = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
        auto neighbors = tree.radius_search(query, radius);

        // Limit to max_nn
        if (static_cast<int>(neighbors.size()) > max_nn) {
            std::partial_sort(neighbors.begin(),
                              neighbors.begin() + max_nn,
                              neighbors.end(),
                              [](const auto& a, const auto& b) {
                                  return a.second < b.second;
                              });
            neighbors.resize(max_nn);
        }

        if (neighbors.size() < 3) {
            normals[i * 3 + 2] = 1.0;  // default up
            continue;
        }

        // Compute centroid
        double cx = 0, cy = 0, cz = 0;
        for (const auto& [idx, _] : neighbors) {
            cx += data[idx * 3];
            cy += data[idx * 3 + 1];
            cz += data[idx * 3 + 2];
        }
        double inv_n = 1.0 / static_cast<double>(neighbors.size());
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        // Covariance matrix (upper triangle: xx, xy, xz, yy, yz, zz)
        double cov[6] = {};
        for (const auto& [idx, _] : neighbors) {
            double dx = data[idx * 3] - cx;
            double dy = data[idx * 3 + 1] - cy;
            double dz = data[idx * 3 + 2] - cz;
            cov[0] += dx * dx;
            cov[1] += dx * dy;
            cov[2] += dx * dz;
            cov[3] += dy * dy;
            cov[4] += dy * dz;
            cov[5] += dz * dz;
        }

        double eigenvalues[3];
        double eigenvectors[9];
        eigen_3x3_symmetric(cov, eigenvalues, eigenvectors);

        // Normal is the eigenvector of the smallest eigenvalue (index 0)
        double len = std::sqrt(eigenvectors[0] * eigenvectors[0] +
                               eigenvectors[1] * eigenvectors[1] +
                               eigenvectors[2] * eigenvectors[2]);
        if (len > 1e-12) {
            normals[i * 3]     = eigenvectors[0] / len;
            normals[i * 3 + 1] = eigenvectors[1] / len;
            normals[i * 3 + 2] = eigenvectors[2] / len;
        } else {
            normals[i * 3 + 2] = 1.0;
        }
    }
    return normals;
}

void orient_normals_to_viewpoint(double* normals, const double* data,
                                 size_t n,
                                 double vx, double vy, double vz) {
    for (size_t i = 0; i < n; ++i) {
        double dx = vx - data[i * 3];
        double dy = vy - data[i * 3 + 1];
        double dz = vz - data[i * 3 + 2];

        double dot = normals[i * 3] * dx +
                     normals[i * 3 + 1] * dy +
                     normals[i * 3 + 2] * dz;

        if (dot < 0) {
            normals[i * 3]     = -normals[i * 3];
            normals[i * 3 + 1] = -normals[i * 3 + 1];
            normals[i * 3 + 2] = -normals[i * 3 + 2];
        }
    }
}

}  // namespace occulus
