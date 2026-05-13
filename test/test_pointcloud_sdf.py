"""Python-only example for PointCloudSDF.

Run with: `pixi run test-pointcloud-sdf`

Builds a synthetic obstacle as a point cloud, visualizes the resulting
KDTree-backed SDF on the z=0 slice, and runs the FK CBF with the
PointCloudSDF as the env signed-distance function.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pinocchio
from urdf_parser_py import urdf

from pinocchio_ik.qpcbf import PinocchioFKCBF, PointCloudSDF


def sample_sphere_surface(center, radius, n_points, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, n_points)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    s = np.sqrt(1.0 - u * u)
    pts = np.stack([s * np.cos(theta), s * np.sin(theta), u], axis=-1) * radius
    return pts + center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--buffer", type=float, default=0.0)
    parser.add_argument("--meshcat", action="store_true", help="Animate the trajectory and point cloud in Meshcat")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    urdf_file = args.urdf or os.path.join(here, "two_manipulator_arm.urdf")

    center = np.array([0.5, 0.5, 0.0])
    radius = 0.075
    points = sample_sphere_surface(center, radius, args.n_points)

    sdf = PointCloudSDF(points=points, buffer=args.buffer)

    # 1) Single-query sanity check.
    d, g = sdf(np.array([[0.5, 0.5 - 0.2, 0.0]]))
    print(f"query at 0.2 m from obstacle surface: d={d[0]:.4f} (expected ~{0.2 - radius - args.buffer:.4f}), |g|={np.linalg.norm(g[0]):.4f}")

    # 2) Run the FK CBF rolling out from start to goal.
    urdf_model = urdf.URDF.from_xml_file(urdf_file)
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_file)
    controller = PinocchioFKCBF(model, urdf_model, sdf)
    print(f"model nq={model.nq}, controlled joints={controller.controlled_joint_names}")

    start_config = np.array([-1.0] * model.nq)
    goal_config = np.array([1.0] * model.nq)

    def nominal(q):
        diff = goal_config - q
        n = np.linalg.norm(diff)
        return diff / n if n > 1e-9 else np.zeros_like(diff)

    solutions, _ = controller.controller.solve_ode(start_config, nominal, 100, 0.05)
    print(f"rolled out {solutions.shape[0]} steps, final q = {solutions[-1]}")

    _save_plot(controller, sdf, points, center, radius, solutions, here)

    if args.meshcat:
        _animate_meshcat(model, collision_model, visual_model, points, solutions)


def _animate_meshcat(model, collision_model, visual_model, points, solutions):
    import meshcat
    import meshcat.geometry as mg
    from pinocchio.visualize import MeshcatVisualizer

    # Spawn the server ourselves so we can print its URL up front.
    server = meshcat.Visualizer()
    print()
    print("=" * 60)
    print(f"Open Meshcat at: {server.url()}")
    print("=" * 60)
    print()

    viz = MeshcatVisualizer(model, collision_model=collision_model, visual_model=visual_model)
    viz.initViewer(viewer=server, loadModel=True)
    viz.displayCollisions(True)
    viz.displayVisuals(True)
    viz.display(solutions[0])

    points_t = points.T.astype(np.float32)
    viz.viewer["obstacle/cloud"].set_object(
        mg.Points(
            mg.PointsGeometry(position=points_t),
            mg.PointsMaterial(size=0.02, color=0xff3333),
        )
    )

    # Give the user a chance to open the URL before animating.
    print("Press Enter to start the animation loop (Ctrl-C to exit when done).")
    try:
        input()
    except EOFError:
        pass

    idx = 0
    try:
        while True:
            viz.display(solutions[idx])
            time.sleep(0.05)
            idx = (idx + 1) % solutions.shape[0]
    except KeyboardInterrupt:
        pass


def _save_plot(controller, sdf, points, center, radius, solutions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pinocchio as pin

    grid_lo, grid_hi, n_grid = -0.2, 1.2, 100
    xs = np.linspace(grid_lo, grid_hi, n_grid)
    ys = np.linspace(grid_lo, grid_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    grid_points = np.stack([X, Y, Z], axis=-1)
    grid_flat = grid_points.reshape(-1, 3)
    d, _ = sdf(grid_flat)
    d = d.reshape(X.shape)

    ee_xy = np.empty((solutions.shape[0], 2))
    for i, q in enumerate(solutions):
        pin.forwardKinematics(controller.model, controller.data, q)
        pin.updateFramePlacements(controller.model, controller.data)
        ee_xy[i] = controller.get_frame_pose("end_effector").translation[:2]

    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(X, Y, d, levels=30, cmap="viridis")
    ax.contour(X, Y, d, levels=[0.0], colors="red", linewidths=1.5)
    fig.colorbar(cf, ax=ax, label="signed distance [m]")

    in_slice = np.abs(points[:, 2]) < 0.02
    ax.scatter(points[in_slice, 0], points[in_slice, 1], c="white", s=4, label="cloud (|z|<2cm)")

    theta = np.linspace(0, 2 * np.pi, 64)
    ax.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
            "r--", linewidth=1, label="true surface")

    ax.plot(ee_xy[:, 0], ee_xy[:, 1], "k-", marker=".", markersize=3, label="EE trajectory")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("PointCloudSDF (z=0 slice) + FK CBF rollout")
    ax.set_aspect("equal")
    ax.set_xlim(grid_lo, grid_hi)
    ax.set_ylim(grid_lo, grid_hi)
    ax.legend(loc="lower right", fontsize=8)

    out = os.path.join(out_dir, "test_pointcloud_sdf.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
