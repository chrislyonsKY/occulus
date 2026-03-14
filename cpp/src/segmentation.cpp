#include "occulus/segmentation.hpp"
#include "occulus/kdtree.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace occulus {

std::vector<int> classify_ground_csf(const double* data, size_t n,
                                     double cloth_resolution,
                                     double class_threshold,
                                     int max_iterations) {
    if (n == 0) return {};

    // Find bounding box
    double xmin = data[0], xmax = data[0];
    double ymin = data[1], ymax = data[1];
    double zmin = data[2], zmax = data[2];
    for (size_t i = 1; i < n; ++i) {
        double x = data[i * 3], y = data[i * 3 + 1], z = data[i * 3 + 2];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
    }

    // Create cloth grid
    int nx = static_cast<int>(std::ceil((xmax - xmin) / cloth_resolution)) + 1;
    int ny = static_cast<int>(std::ceil((ymax - ymin) / cloth_resolution)) + 1;

    // Initialize cloth at max height (inverted: ground is "up")
    double z_inv_max = -zmin;  // Invert: flip Z
    std::vector<double> cloth(nx * ny, z_inv_max);
    std::vector<bool> fixed(nx * ny, false);

    // Invert the point cloud (negate Z)
    std::vector<double> inv_z(n);
    for (size_t i = 0; i < n; ++i) {
        inv_z[i] = -data[i * 3 + 2];
    }

    // For each grid cell, find the highest inverted point (= lowest original Z)
    std::vector<double> max_inv_z(nx * ny, -1e30);
    for (size_t i = 0; i < n; ++i) {
        int ix = static_cast<int>((data[i * 3] - xmin) / cloth_resolution);
        int iy = static_cast<int>((data[i * 3 + 1] - ymin) / cloth_resolution);
        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);
        if (inv_z[i] > max_inv_z[iy * nx + ix]) {
            max_inv_z[iy * nx + ix] = inv_z[i];
        }
    }

    // Simulate cloth falling under gravity
    double gravity = 0.3;
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Move cloth particles down
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = j * nx + i;
                if (!fixed[idx]) {
                    cloth[idx] -= gravity;
                    // Collision with terrain
                    if (max_inv_z[idx] > -1e29 && cloth[idx] < max_inv_z[idx]) {
                        cloth[idx] = max_inv_z[idx];
                        fixed[idx] = true;
                    }
                }
            }
        }

        // Internal spring forces (smooth the cloth)
        std::vector<double> new_cloth = cloth;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = j * nx + i;
                if (fixed[idx]) continue;

                double sum = 0;
                int count = 0;
                if (i > 0)      { sum += cloth[idx - 1]; count++; }
                if (i < nx - 1) { sum += cloth[idx + 1]; count++; }
                if (j > 0)      { sum += cloth[idx - nx]; count++; }
                if (j < ny - 1) { sum += cloth[idx + nx]; count++; }

                if (count > 0) {
                    new_cloth[idx] = sum / count;
                    // Re-check collision
                    if (max_inv_z[idx] > -1e29 && new_cloth[idx] < max_inv_z[idx]) {
                        new_cloth[idx] = max_inv_z[idx];
                        fixed[idx] = true;
                    }
                }
            }
        }
        cloth = new_cloth;
    }

    // Classify points: compare each point to nearest cloth particle
    std::vector<int> classification(n, 1);  // default: unclassified
    for (size_t i = 0; i < n; ++i) {
        int ix = static_cast<int>((data[i * 3] - xmin) / cloth_resolution);
        int iy = static_cast<int>((data[i * 3 + 1] - ymin) / cloth_resolution);
        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);

        double cloth_z = -cloth[iy * nx + ix];  // Convert back from inverted
        double dist = std::abs(data[i * 3 + 2] - cloth_z);

        if (dist <= class_threshold) {
            classification[i] = 2;  // Ground (ASPRS class 2)
        }
    }

    return classification;
}

std::vector<int> classify_ground_pmf(const double* data, size_t n,
                                     int max_window, double slope,
                                     double initial_dist, double max_dist,
                                     double cell_size) {
    if (n == 0) return {};

    // Find bounding box
    double xmin = data[0], xmax = data[0];
    double ymin = data[1], ymax = data[1];
    for (size_t i = 1; i < n; ++i) {
        if (data[i * 3] < xmin) xmin = data[i * 3];
        if (data[i * 3] > xmax) xmax = data[i * 3];
        if (data[i * 3 + 1] < ymin) ymin = data[i * 3 + 1];
        if (data[i * 3 + 1] > ymax) ymax = data[i * 3 + 1];
    }

    int nx = static_cast<int>(std::ceil((xmax - xmin) / cell_size)) + 1;
    int ny = static_cast<int>(std::ceil((ymax - ymin) / cell_size)) + 1;

    // Build minimum elevation grid
    std::vector<double> grid(nx * ny, 1e30);
    for (size_t i = 0; i < n; ++i) {
        int ix = static_cast<int>((data[i * 3] - xmin) / cell_size);
        int iy = static_cast<int>((data[i * 3 + 1] - ymin) / cell_size);
        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);
        grid[iy * nx + ix] = std::min(grid[iy * nx + ix], data[i * 3 + 2]);
    }

    // Progressive morphological opening
    std::vector<double> opened = grid;
    double slope_rad = slope * M_PI / 180.0;

    for (int w = 1; w <= max_window; w *= 2) {
        double dh = std::min(initial_dist + slope_rad * w * cell_size, max_dist);
        std::vector<double> eroded(nx * ny);
        std::vector<double> dilated(nx * ny);

        // Erosion (min filter)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double val = 1e30;
                for (int dj = -w; dj <= w; ++dj) {
                    for (int di = -w; di <= w; ++di) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < nx && nj >= 0 && nj < ny) {
                            val = std::min(val, opened[nj * nx + ni]);
                        }
                    }
                }
                eroded[j * nx + i] = val;
            }
        }

        // Dilation (max filter on eroded)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double val = -1e30;
                for (int dj = -w; dj <= w; ++dj) {
                    for (int di = -w; di <= w; ++di) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < nx && nj >= 0 && nj < ny) {
                            val = std::max(val, eroded[nj * nx + ni]);
                        }
                    }
                }
                dilated[j * nx + i] = val;
            }
        }

        opened = dilated;
    }

    // Classify: points close to the opened surface are ground
    std::vector<int> classification(n, 1);
    for (size_t i = 0; i < n; ++i) {
        int ix = static_cast<int>((data[i * 3] - xmin) / cell_size);
        int iy = static_cast<int>((data[i * 3 + 1] - ymin) / cell_size);
        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);

        double ground_z = opened[iy * nx + ix];
        if (ground_z < 1e29 && std::abs(data[i * 3 + 2] - ground_z) <= max_dist) {
            classification[i] = 2;
        }
    }

    return classification;
}

}  // namespace occulus
