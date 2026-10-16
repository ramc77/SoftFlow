"""
Pre-processing and geometry generation utilities for SoftFlow simulations.

Import this module for geometry helpers without loading the simulation engine:

    from pysoftflow.preprocess import (
        staggered_pillars,
        inline_pillars,
        stenosis_polygons,
        wavy_wall_polygons,
        bifurcation_polygons,
        random_obstacles,
    )

All functions return geometry descriptions (lists of dicts or vertex lists)
that can be passed to ``SoftFlowSimulation.obstacle()``.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════
# Pillar Arrays
# ══════════════════════════════════════════════════════════════

def staggered_pillars(
    nx: int,
    ny: int,
    radius: float = 8.0,
    spacing: float = 40.0,
    margin: Optional[float] = None,
) -> List[dict]:
    """Generate a staggered array of circular pillars.

    Parameters
    ----------
    nx, ny : int
        Domain dimensions.
    radius : float
        Pillar radius.
    spacing : float
        Center-to-center spacing between pillars.
    margin : float, optional
        Margin from domain edges. Default: spacing/2.

    Returns
    -------
    list of dict
        Each dict: ``{"shape": "circle", "center": (cx, cy), "radius": r}``.
    """
    if margin is None:
        margin = spacing * 0.5

    pillars = []
    row = 0
    cy = margin
    while cy < ny - margin:
        offset = (spacing * 0.5) if row % 2 == 1 else 0.0
        cx = margin + offset
        while cx < nx - margin:
            pillars.append({
                "shape": "circle",
                "center": (cx, cy),
                "radius": radius,
            })
            cx += spacing
        cy += spacing
        row += 1

    return pillars


def inline_pillars(
    nx: int,
    ny: int,
    radius: float = 8.0,
    spacing_x: float = 40.0,
    spacing_y: float = 40.0,
    margin: Optional[float] = None,
) -> List[dict]:
    """Generate an inline (regular grid) array of circular pillars.

    Parameters
    ----------
    nx, ny : int
        Domain dimensions.
    radius : float
        Pillar radius.
    spacing_x, spacing_y : float
        Spacing in x and y directions.
    margin : float, optional
        Margin from domain edges. Default: spacing_x/2.

    Returns
    -------
    list of dict
        Each dict: ``{"shape": "circle", "center": (cx, cy), "radius": r}``.
    """
    if margin is None:
        margin = max(spacing_x, spacing_y) * 0.5

    pillars = []
    cy = margin
    while cy < ny - margin:
        cx = margin
        while cx < nx - margin:
            pillars.append({
                "shape": "circle",
                "center": (cx, cy),
                "radius": radius,
            })
            cx += spacing_x
        cy += spacing_y

    return pillars


def random_obstacles(
    nx: int,
    ny: int,
    count: int = 10,
    radius_range: Tuple[float, float] = (5.0, 12.0),
    margin: float = 15.0,
    min_gap: float = 5.0,
    seed: int = 42,
) -> List[dict]:
    """Generate randomly placed non-overlapping circular obstacles.

    Parameters
    ----------
    nx, ny : int
        Domain dimensions.
    count : int
        Number of obstacles to place.
    radius_range : (min, max)
        Range of obstacle radii.
    margin : float
        Minimum distance from domain edges.
    min_gap : float
        Minimum gap between obstacle surfaces.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict: ``{"shape": "circle", "center": (cx, cy), "radius": r}``.
    """
    rng = np.random.RandomState(seed)
    rmin, rmax = radius_range
    obstacles = []
    max_attempts = count * 100

    for _ in range(max_attempts):
        if len(obstacles) >= count:
            break

        r = rng.uniform(rmin, rmax)
        cx = rng.uniform(margin + r, nx - margin - r)
        cy = rng.uniform(margin + r, ny - margin - r)

        # Check overlap with existing obstacles
        overlap = False
        for obs in obstacles:
            dx = cx - obs["center"][0]
            dy = cy - obs["center"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < r + obs["radius"] + min_gap:
                overlap = True
                break

        if not overlap:
            obstacles.append({
                "shape": "circle",
                "center": (cx, cy),
                "radius": r,
            })

    return obstacles


# ══════════════════════════════════════════════════════════════
# Polygon-Based Geometry
# ══════════════════════════════════════════════════════════════

def stenosis_polygons(
    ny: int,
    center_x: float,
    width: float = 40.0,
    severity: float = 0.4,
) -> List[List[Tuple[float, float]]]:
    """Generate top and bottom polygon vertices for a symmetric stenosis.

    Parameters
    ----------
    ny : int
        Channel height.
    center_x : float
        X-position of stenosis center.
    width : float
        Total length of stenosis region.
    severity : float
        Fraction of half-channel blocked (0-1). 0.4 = 40% constriction.

    Returns
    -------
    list of list of (x, y) tuples
        Two polygons: [bottom_vertices, top_vertices].
    """
    h = severity * ny / 2.0
    x0 = center_x - width
    x1 = center_x - width / 3
    x2 = center_x + width / 3
    x3 = center_x + width

    bottom = [(x0, 0), (x1, h), (x2, h), (x3, 0)]
    top = [(x0, ny), (x1, ny - h), (x2, ny - h), (x3, ny)]

    return [bottom, top]


def wavy_wall_polygons(
    nx: int,
    ny: int,
    amplitude: float = 10.0,
    wavelength: float = 50.0,
    wall: str = "both",
    n_points: int = 100,
) -> List[List[Tuple[float, float]]]:
    """Generate polygon vertices for sinusoidal wavy walls.

    Parameters
    ----------
    nx : int
        Domain length.
    ny : int
        Domain height.
    amplitude : float
        Wave amplitude in lattice units.
    wavelength : float
        Wavelength in lattice units.
    wall : str
        ``"bottom"``, ``"top"``, or ``"both"``.
    n_points : int
        Number of points along the wall.

    Returns
    -------
    list of list of (x, y) tuples
        Polygon vertex lists for each wall.
    """
    x_pts = np.linspace(0, nx, n_points)
    wave = amplitude * np.sin(2 * np.pi * x_pts / wavelength)

    polygons = []

    if wall in ("bottom", "both"):
        verts = [(float(x_pts[0]), 0.0)]
        for i in range(n_points):
            verts.append((float(x_pts[i]), float(wave[i] + amplitude)))
        verts.append((float(x_pts[-1]), 0.0))
        polygons.append(verts)

    if wall in ("top", "both"):
        verts = [(float(x_pts[0]), float(ny))]
        for i in range(n_points):
            verts.append((float(x_pts[i]), float(ny - wave[i] - amplitude)))
        verts.append((float(x_pts[-1]), float(ny)))
        polygons.append(verts)

    return polygons


def bifurcation_polygons(
    nx: int,
    ny: int,
    branch_x: float = None,
    angle: float = 30.0,
    branch_width: float = None,
) -> List[List[Tuple[float, float]]]:
    """Generate polygon vertices for a Y-shaped bifurcation.

    Creates a central divider polygon that splits the channel into
    two branches starting at ``branch_x``.

    Parameters
    ----------
    nx : int
        Domain length.
    ny : int
        Domain height.
    branch_x : float, optional
        X-position where bifurcation starts. Default: nx/3.
    angle : float
        Half-angle of bifurcation in degrees.
    branch_width : float, optional
        Width of the central divider at the outlet. Default: ny/4.

    Returns
    -------
    list of list of (x, y) tuples
        Polygon vertex list for the central divider.
    """
    if branch_x is None:
        branch_x = nx / 3.0
    if branch_width is None:
        branch_width = ny / 4.0

    cy = ny / 2.0
    half_w = branch_width / 2.0

    # Central divider: wedge shape
    tip_x = branch_x
    end_x = nx

    divider = [
        (tip_x, cy),
        (end_x, cy + half_w),
        (end_x, cy - half_w),
    ]

    return [divider]


def constriction_rect(
    ny: int,
    center_x: float,
    width: float = 20.0,
    gap: float = 20.0,
) -> List[dict]:
    """Generate two rectangular obstacles forming a constriction.

    Parameters
    ----------
    ny : int
        Channel height.
    center_x : float
        X-center of constriction.
    width : float
        Width of each rectangular obstacle.
    gap : float
        Gap between top and bottom obstacles.

    Returns
    -------
    list of dict
        Two rect obstacle dicts with p1, p2.
    """
    half_gap = gap / 2.0
    cy = ny / 2.0

    return [
        {
            "shape": "rect",
            "p1": (center_x - width / 2, 0),
            "p2": (center_x + width / 2, cy - half_gap),
        },
        {
            "shape": "rect",
            "p1": (center_x - width / 2, cy + half_gap),
            "p2": (center_x + width / 2, ny),
        },
    ]


def diamond_obstacle(
    cx: float,
    cy: float,
    size: float = 10.0,
) -> List[Tuple[float, float]]:
    """Generate vertices for a diamond (rotated square) obstacle.

    Parameters
    ----------
    cx, cy : float
        Center position.
    size : float
        Half-diagonal length.

    Returns
    -------
    list of (x, y) tuples
        Four vertices of the diamond.
    """
    return [
        (cx, cy + size),
        (cx + size, cy),
        (cx, cy - size),
        (cx - size, cy),
    ]


def triangle_obstacle(
    cx: float,
    cy: float,
    size: float = 10.0,
    orientation: str = "right",
) -> List[Tuple[float, float]]:
    """Generate vertices for a triangular obstacle.

    Parameters
    ----------
    cx, cy : float
        Center position.
    size : float
        Half-height of the triangle.
    orientation : str
        ``"right"`` (pointing right), ``"left"``, ``"up"``, ``"down"``.

    Returns
    -------
    list of (x, y) tuples
        Three vertices of the triangle.
    """
    if orientation == "right":
        return [(cx - size, cy + size), (cx + size, cy), (cx - size, cy - size)]
    elif orientation == "left":
        return [(cx + size, cy + size), (cx - size, cy), (cx + size, cy - size)]
    elif orientation == "up":
        return [(cx - size, cy - size), (cx, cy + size), (cx + size, cy - size)]
    elif orientation == "down":
        return [(cx - size, cy + size), (cx, cy - size), (cx + size, cy + size)]
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")
