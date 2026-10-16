"""
Matplotlib-based visualization helpers for SoftFlow simulations.

All functions accept an optional *ax* argument.  If ``None`` a new figure
and axes are created automatically.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from .simulation import Simulation


def _get_ax(ax: Optional["matplotlib.axes.Axes"] = None):
    """Return (*fig*, *ax*), creating them if *ax* is ``None``."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    else:
        fig = ax.get_figure()
    return fig, ax


# ── Fluid field plotting ─────────────────────────────────────

def plot_fluid(
    sim: "Simulation",
    field: str = "density",
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    **imshow_kwargs,
) -> "matplotlib.axes.Axes":
    """Plot a scalar fluid field.

    Parameters
    ----------
    sim : Simulation
        An *initialized* Simulation instance.
    field : str
        One of ``"density"``, ``"velocity_x"``, ``"velocity_y"``,
        or ``"speed"`` (magnitude of velocity).
    ax : Axes, optional
        Matplotlib axes to draw on.
    cmap : str
        Matplotlib colourmap name.
    colorbar : bool
        Whether to add a colourbar.
    **imshow_kwargs
        Passed through to ``ax.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _get_ax(ax)

    if field == "density":
        data = sim.fluid_density()
        label = r"$\rho$"
    elif field == "velocity_x":
        data = sim.fluid_velocity()[0]
        label = r"$u_x$"
    elif field == "velocity_y":
        data = sim.fluid_velocity()[1]
        label = r"$u_y$"
    elif field == "speed":
        ux, uy = sim.fluid_velocity()
        data = np.sqrt(ux ** 2 + uy ** 2)
        label = r"$|\mathbf{u}|$"
    else:
        raise ValueError(f"Unknown field: {field!r}")

    kwargs = dict(origin="lower", cmap=cmap, aspect="equal")
    kwargs.update(imshow_kwargs)
    im = ax.imshow(data, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Fluid {field} (step {sim.current_step})")
    if colorbar:
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    return ax


# ── Capsule overlay ──────────────────────────────────────────

def plot_capsules(
    sim: "Simulation",
    color_by: str = "type",
    ax: Optional["matplotlib.axes.Axes"] = None,
    linewidth: float = 1.5,
    centroid_marker: bool = True,
    **plot_kwargs,
) -> "matplotlib.axes.Axes":
    """Draw capsule membranes.

    Parameters
    ----------
    sim : Simulation
        An *initialized* Simulation instance.
    color_by : str
        ``"type"`` colours each capsule by its type_id,
        ``"index"`` by its ordinal position.
    ax : Axes, optional
        Matplotlib axes to draw on.
    linewidth : float
        Line width for membrane curves.
    centroid_marker : bool
        Mark each centroid with a dot.
    **plot_kwargs
        Passed through to ``ax.plot`` for each capsule.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax = _get_ax(ax)
    cs = sim.core.capsules()

    cmap = plt.cm.tab10
    for i in range(cs.numCapsules()):
        cap = cs[i]
        positions = cap.node_positions_array()  # (num_nodes, 2) numpy
        # Close the ring for drawing
        xs = np.concatenate([positions[:, 0], positions[:1, 0]])
        ys = np.concatenate([positions[:, 1], positions[:1, 1]])

        if color_by == "type":
            colour = cmap(cap.getTypeId() % 10)
        else:
            colour = cmap(i % 10)

        ax.plot(xs, ys, color=colour, linewidth=linewidth, **plot_kwargs)

        if centroid_marker:
            c = cap.centroid()
            ax.plot(c.x, c.y, "o", color=colour, markersize=3)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Capsules (step {sim.current_step})")
    return ax


# ── Combined snapshot ────────────────────────────────────────

def plot_snapshot(
    sim: "Simulation",
    fluid_field: str = "speed",
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = "viridis",
    **kwargs,
) -> "matplotlib.axes.Axes":
    """Plot fluid field with capsule membranes overlaid.

    Parameters
    ----------
    sim : Simulation
        An *initialized* Simulation instance.
    fluid_field : str
        Which scalar field to show in the background (see :func:`plot_fluid`).
    ax : Axes, optional
        Matplotlib axes.
    cmap : str
        Colourmap for the fluid.
    **kwargs
        Extra arguments forwarded to :func:`plot_fluid`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = plot_fluid(sim, field=fluid_field, ax=ax, cmap=cmap, **kwargs)
    plot_capsules(sim, ax=ax, color_by="type", linewidth=1.2)
    ax.set_title(f"Snapshot (step {sim.current_step})")
    return ax


# ── Animation ────────────────────────────────────────────────

def animate(
    sim: "Simulation",
    num_steps: int,
    interval: int = 10,
    filename: Optional[str] = None,
    fluid_field: str = "speed",
    cmap: str = "viridis",
    fps: int = 30,
    dpi: int = 100,
):
    """Create a matplotlib animation of the simulation.

    The simulation is advanced *num_steps* timesteps; a frame is captured
    every *interval* steps.

    Parameters
    ----------
    sim : Simulation
        Must already be initialized.
    num_steps : int
        Total timesteps to run.
    interval : int
        Capture a frame every *interval* steps.
    filename : str, optional
        If given, save the animation to this file (e.g. ``"sim.mp4"``
        or ``"sim.gif"``).  Requires ``ffmpeg`` or ``pillow``.
    fluid_field : str
        Fluid scalar field for background (``"density"``, ``"speed"``, etc.).
    cmap : str
        Colourmap.
    fps : int
        Frames per second for saved video.
    dpi : int
        Resolution of saved video.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Initial frame
    plot_snapshot(sim, fluid_field=fluid_field, ax=ax, cmap=cmap)
    im = ax.images[0] if ax.images else None

    total_frames = num_steps // interval

    def _update(frame_idx):
        # Advance the simulation
        for _ in range(interval):
            sim.step()

        ax.cla()
        plot_snapshot(sim, fluid_field=fluid_field, ax=ax, cmap=cmap)
        return ax.get_children()

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=total_frames,
        blit=False,
        repeat=False,
    )

    if filename is not None:
        if filename.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer, dpi=dpi)

    return anim
