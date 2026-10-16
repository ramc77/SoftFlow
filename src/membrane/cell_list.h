#pragma once
#include "../core/types.h"
#include "capsule_system.h"
#include <vector>
#include <cmath>

namespace softflow {

/// Uniform-grid cell list for O(N) neighbor lookups in 2D.
///
/// Standard molecular-dynamics technique: the domain is divided into a grid of
/// cells whose side length equals the interaction cutoff r_cut.  Each membrane
/// node is assigned to a cell based on its position.  When computing pairwise
/// interactions, only nodes in the same or adjacent cells (3x3 stencil in 2D)
/// need to be checked, reducing the complexity from O(N^2) to O(N * avg_neighbors).
///
/// Periodic boundary handling:
///   - x-direction: periodic wrapping of cell indices when Lx > 0
///   - y-direction: non-periodic (walls), out-of-range cells are simply skipped
class CellList {
public:
    /// A (capsule_index, node_index) pair identifying a membrane node.
    struct NodeRef {
        int capsule;
        int node;
    };

    CellList() = default;

    /// Build the cell list from the current capsule positions.
    /// @param system   capsule system with current node positions
    /// @param r_cut    interaction cutoff (= cell side length)
    /// @param Lx       periodic domain width in x (0 = non-periodic)
    /// @param y_min    lower y bound of the domain
    /// @param y_max    upper y bound of the domain
    void build(const CapsuleSystem& system, Real r_cut,
               Real Lx, Real y_min, Real y_max) {
        r_cut_ = r_cut;
        Lx_ = Lx;
        y_min_ = y_min;
        y_max_ = y_max;

        // Determine grid dimensions.
        // x-extent: if periodic, use Lx; otherwise scan for actual extent.
        Real x_extent, x_origin;
        if (Lx > 0.0) {
            x_extent = Lx;
            x_origin = 0.0;
        } else {
            // Find bounding box of all nodes in x
            Real xmin = 1e30, xmax = -1e30;
            int ncaps = system.numCapsules();
            for (int c = 0; c < ncaps; ++c) {
                const auto& pos = system[c].positions();
                for (const auto& p : pos) {
                    if (p.x < xmin) xmin = p.x;
                    if (p.x > xmax) xmax = p.x;
                }
            }
            x_origin = xmin - r_cut;
            x_extent = (xmax - xmin) + 2.0 * r_cut;
        }
        x_origin_ = x_origin;

        ncx_ = std::max(1, static_cast<int>(std::floor(x_extent / r_cut)));
        ncy_ = std::max(1, static_cast<int>(std::floor((y_max - y_min) / r_cut)));

        // Cell dimensions (may be slightly larger than r_cut)
        dx_ = x_extent / static_cast<Real>(ncx_);
        dy_ = (y_max - y_min) / static_cast<Real>(ncy_);

        int total_cells = ncx_ * ncy_;
        cells_.assign(total_cells, std::vector<NodeRef>());

        // Insert all membrane nodes
        int ncaps = system.numCapsules();
        for (int c = 0; c < ncaps; ++c) {
            const auto& pos = system[c].positions();
            int nnodes = system[c].numNodes();
            for (int k = 0; k < nnodes; ++k) {
                int ci = cellIndex(pos[k]);
                if (ci >= 0 && ci < total_cells) {
                    cells_[ci].push_back({c, k});
                }
            }
        }
    }

    /// Number of cells in x and y.
    int ncx() const { return ncx_; }
    int ncy() const { return ncy_; }

    /// Access the node list in a given cell.
    const std::vector<NodeRef>& cell(int cx, int cy) const {
        return cells_[cy * ncx_ + cx];
    }

    /// Iterate over all (cell, neighbor_cell) pairs relevant for a given
    /// target cell (cx, cy).  Returns the 3x3 stencil of neighbor cell
    /// indices (with periodic wrapping in x if applicable).
    /// Writes up to 9 neighbor cell flat-indices into `out` and returns count.
    int neighborCells(int cx, int cy, int out[9]) const {
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy;
            if (ny < 0 || ny >= ncy_) continue;  // y is non-periodic
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx;
                // Periodic wrapping in x
                if (Lx_ > 0.0) {
                    if (nx < 0) nx += ncx_;
                    if (nx >= ncx_) nx -= ncx_;
                } else {
                    if (nx < 0 || nx >= ncx_) continue;
                }
                out[count++] = ny * ncx_ + nx;
            }
        }
        return count;
    }

    /// Total number of cells.
    int totalCells() const { return ncx_ * ncy_; }

    /// Get the flat cell index for a position.
    int cellIndex(const Vec2d& pos) const {
        // Guard against NaN/Inf (prevents infinite while loops)
        if (std::isnan(pos.x) || std::isnan(pos.y) ||
            std::isinf(pos.x) || std::isinf(pos.y))
            return 0;

        Real px = pos.x - x_origin_;
        // Periodic wrap in x (use fmod to avoid infinite loops)
        if (Lx_ > 0.0) {
            px = std::fmod(px, Lx_);
            if (px < 0.0) px += Lx_;
        }
        int cx = static_cast<int>(std::floor(px / dx_));
        int cy = static_cast<int>(std::floor((pos.y - y_min_) / dy_));
        // Clamp (safety)
        if (cx < 0) cx = 0;
        if (cx >= ncx_) cx = ncx_ - 1;
        if (cy < 0) cy = 0;
        if (cy >= ncy_) cy = ncy_ - 1;
        return cy * ncx_ + cx;
    }

private:
    Real r_cut_ = 0.0;
    Real Lx_ = 0.0;
    Real y_min_ = 0.0;
    Real y_max_ = 0.0;
    Real x_origin_ = 0.0;
    Real dx_ = 1.0;
    Real dy_ = 1.0;
    int ncx_ = 0;
    int ncy_ = 0;
    std::vector<std::vector<NodeRef>> cells_;
};

} // namespace softflow
