"""
Pre-configured simulation scenarios using the SoftFlowSimulation API.

Each function returns a fully configured (but not yet initialized)
:class:`pysoftflow.SoftFlowSimulation` that the caller can tweak
further before calling ``.run()``.
"""

from __future__ import annotations

from .setup import SoftFlowSimulation


def blood_flow_channel(
    channel_length: int = 400,
    channel_width: int = 80,
    rbc_count: int = 30,
    rbc_radius: float = 4.0,
    inlet_velocity: float = 0.02,
    tau: float = 0.8,
    output_dir: str = "output_blood",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Straight channel with Skalak RBC capsules and lubrication.

    Uses realistic membrane model (Skalak) with viscosity contrast.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=channel_length, ny=channel_width)
    sim.boundary(x="inlet_outlet", y="wall")
    sim.fluid(tau=tau, density=1.0,
              inlet_velocity=inlet_velocity, outlet_density=1.0)
    sim.ibm(iterations=2)
    sim.lubrication(enabled=True)

    sim.particle_type("rbc", model="skalak", G_s=0.06, C_skalak=10.0,
                      k_bend=0.003, k_area=0.8, k_perimeter=0.08,
                      gamma_visc=0.008, viscosity_ratio=5.0)

    sim.interaction("particle-particle", style="morse",
                    epsilon=0.003, sigma=0.8, r_cut=2.0, power=6)
    sim.interaction("particle-wall", style="morse",
                    epsilon=0.005, sigma=0.8, r_cut=2.0, power=6)

    margin_x = channel_length * 0.1
    margin_y = rbc_radius + 2.0
    sim.region("seed", x=(margin_x, channel_length - margin_x),
               y=(margin_y, channel_width - margin_y))
    sim.generate("rbc", count=rbc_count, region="seed", radius=rbc_radius)

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=2000)

    return sim


def porous_media(
    nx: int = 300,
    ny: int = 100,
    pillar_arrangement: str = "staggered",
    pillar_radius: float = 8.0,
    pillar_spacing: float = 40.0,
    capsule_count: int = 15,
    capsule_radius: float = 3.5,
    tau: float = 0.8,
    body_force_x: float = 5e-6,
    output_dir: str = "output_porous",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Periodic channel with a regular array of circular pillars."""
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=tau, density=1.0)
    sim.body_force(body_force_x, 0.0)

    # Generate pillar positions
    margin = pillar_spacing * 0.5
    row = 0
    cy = margin
    while cy < ny - margin:
        offset = (pillar_spacing * 0.5) if (
            pillar_arrangement == "staggered" and row % 2 == 1) else 0.0
        cx = margin + offset
        while cx < nx - margin:
            sim.obstacle("circle", center=(cx, cy), radius=pillar_radius)
            cx += pillar_spacing
        cy += pillar_spacing
        row += 1

    sim.particle_type("soft", model="neo_hookean", G_s=0.08,
                      k_bend=0.004, k_area=0.8, k_perimeter=0.08,
                      gamma_visc=0.008)

    sim.interaction("particle-particle", style="morse",
                    epsilon=0.003, sigma=0.8, r_cut=2.0, power=6)
    sim.interaction("particle-wall", style="morse",
                    epsilon=0.005, sigma=0.8, r_cut=2.0, power=6)

    seed_x1 = min(nx * 0.25, margin - pillar_radius)
    seed_x1 = max(seed_x1, 2.0 * capsule_radius + 2.0)
    sim.region("seed", x=(2.0, seed_x1),
               y=(capsule_radius + 2.0, ny - capsule_radius - 2.0))
    sim.generate("soft", count=capsule_count, region="seed",
                 radius=capsule_radius)

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=2000)

    return sim


def segregation_study(
    nx: int = 400,
    ny: int = 80,
    rbc_count: int = 25,
    platelet_count: int = 15,
    tau: float = 0.8,
    body_force_x: float = 5e-6,
    output_dir: str = "output_segregation",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """RBC + platelet segregation study with quantitative metrics.

    Models the Fahraeus-Lindqvist effect with Skalak RBCs and
    Neo-Hookean platelets. Computes margination, CFL, and mixing entropy.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=tau, density=1.0)
    sim.body_force(body_force_x, 0.0)
    sim.ibm(iterations=2)
    sim.lubrication(enabled=True)
    sim.metrics(interval=10000)

    # RBCs: large, deformable Skalak membrane
    sim.particle_type("rbc", model="skalak", G_s=0.06, C_skalak=10.0,
                      k_bend=0.003, k_area=0.8, k_perimeter=0.08,
                      gamma_visc=0.01)

    # Platelets: small, stiff Neo-Hookean
    sim.particle_type("platelet", model="neo_hookean", G_s=0.3,
                      k_bend=0.02, k_area=2.0, k_perimeter=0.2,
                      gamma_visc=0.02)

    sim.interaction("particle-particle", style="morse",
                    epsilon=0.003, sigma=0.8, r_cut=2.0, power=6)
    sim.interaction("particle-wall", style="morse",
                    epsilon=0.005, sigma=0.8, r_cut=2.0, power=6)

    sim.region("full", x=(10, nx - 10), y=(10, ny - 10))
    sim.generate("rbc", count=rbc_count, region="full", radius=(4.0, 5.0))
    sim.generate("platelet", count=platelet_count, region="full", radius=(1.5, 2.5))

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=5000)

    return sim


def tumor_cell_adhesion(
    nx: int = 400,
    ny: int = 80,
    rbc_count: int = 20,
    tumor_count: int = 5,
    tau: float = 0.8,
    body_force_x: float = 5e-6,
    output_dir: str = "output_tumor",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Tumor cell cluster formation with Bell model adhesion.

    Models circulating tumor cells (CTCs) forming clusters with
    platelets and RBCs under flow. Uses stochastic bond formation.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=tau, density=1.0, collision="regularized")
    sim.body_force(body_force_x, 0.0)
    sim.ibm(iterations=2)
    sim.lubrication(enabled=True)
    sim.adhesion(enabled=True, k_on=0.002, k_off=0.005, k_bond=0.05,
                 d_bond=2.5, F_crit=0.01)
    sim.metrics(interval=5000)

    # RBCs
    sim.particle_type("rbc", model="skalak", G_s=0.06, C_skalak=10.0,
                      k_bend=0.003, k_area=0.8, k_perimeter=0.08)

    # Tumor cells: stiffer than RBCs
    sim.particle_type("tumor", model="neo_hookean", G_s=0.15,
                      k_bend=0.01, k_area=1.5, k_perimeter=0.15)

    sim.interaction("particle-particle", style="morse",
                    epsilon=0.003, sigma=0.8, r_cut=2.0, power=6)

    sim.region("full", x=(10, nx - 10), y=(10, ny - 10))
    sim.generate("rbc", count=rbc_count, region="full", radius=(4.0, 5.0))
    sim.generate("tumor", count=tumor_count, region="full", radius=(4.0, 6.0))

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=5000)

    return sim


def drug_delivery(
    nx: int = 400,
    ny: int = 80,
    carrier_count: int = 10,
    tau: float = 0.8,
    inlet_velocity: float = 0.015,
    diffusivity: float = 0.01,
    output_dir: str = "output_drug",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Drug delivery simulation with scalar transport.

    Models drug-carrying capsules releasing a scalar field
    (drug concentration) as they flow through a channel.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="inlet_outlet", y="wall")
    sim.fluid(tau=tau, density=1.0, inlet_velocity=inlet_velocity,
              outlet_density=1.0)
    sim.ibm(iterations=2)
    sim.scalar_transport(enabled=True, diffusivity=diffusivity, n_species=1)

    # Drug carriers: soft, deformable capsules
    sim.particle_type("carrier", model="skalak", G_s=0.05, C_skalak=5.0,
                      k_bend=0.003, k_area=0.8, k_perimeter=0.08)
    sim.scalar_source("carrier", release_rate=0.001)

    sim.region("inlet", x=(20, 100), y=(10, ny - 10))
    sim.generate("carrier", count=carrier_count, region="inlet",
                 radius=(3.5, 4.5))

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=2000)

    return sim


def couette_flow(
    nx: int = 150,
    ny: int = 61,
    top_wall_velocity: float = 0.02,
    capsule_radius: float = 6.0,
    tau: float = 0.8,
    output_dir: str = "output_couette",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Couette flow with moving wall and a single capsule.

    Studies capsule tank-treading behavior under simple shear
    using Helfrich bending model.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=tau, density=1.0)
    sim.moving_wall(top_velocity=top_wall_velocity)

    sim.particle_type("cell", model="skalak", G_s=0.1, C_skalak=10.0,
                      k_bend=0.008, k_area=1.0, k_perimeter=0.1,
                      gamma_visc=0.01, use_helfrich=True, kappa_0=0.0)

    sim.particle("cell", center=(nx // 2, ny // 2), radius=capsule_radius)

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=5000)

    return sim


def stenosis(
    nx: int = 300,
    ny: int = 80,
    stenosis_center: int = 150,
    stenosis_width: int = 40,
    stenosis_height: int = 18,
    capsule_count: int = 8,
    tau: float = 0.8,
    inlet_velocity: float = 0.015,
    output_dir: str = "output_stenosis",
    output_interval: int = 200,
) -> SoftFlowSimulation:
    """Vascular stenosis using polygon obstacles.

    Models a narrowed blood vessel segment where capsules must
    deform to pass through the constriction.
    """
    sim = SoftFlowSimulation()

    sim.domain(nx=nx, ny=ny)
    sim.boundary(x="inlet_outlet", y="wall")
    sim.fluid(tau=tau, density=1.0, inlet_velocity=inlet_velocity,
              outlet_density=1.0)
    sim.ibm(iterations=2)
    sim.lubrication(enabled=True)

    # Build stenosis from polygon obstacles
    x0 = stenosis_center - stenosis_width
    x1 = stenosis_center - stenosis_width // 3
    x2 = stenosis_center + stenosis_width // 3
    x3 = stenosis_center + stenosis_width

    # Bottom stenosis bump
    sim.obstacle("polygon", vertices=[
        (x0, 0), (x1, stenosis_height),
        (x2, stenosis_height), (x3, 0),
    ])
    # Top stenosis bump
    sim.obstacle("polygon", vertices=[
        (x0, ny), (x1, ny - stenosis_height),
        (x2, ny - stenosis_height), (x3, ny),
    ])

    sim.particle_type("rbc", model="skalak", G_s=0.08, C_skalak=10.0,
                      k_bend=0.004, k_area=1.0, k_perimeter=0.1,
                      gamma_visc=0.01)

    margin = 7.0
    sim.region("upstream", x=(20, stenosis_center - stenosis_width - 10),
               y=(margin, ny - margin))
    sim.generate("rbc", count=capsule_count, region="upstream",
                 radius=(3.5, 4.5))

    sim.output(format="vtk", directory=output_dir, interval=output_interval)
    sim.thermo(interval=5000)

    return sim
