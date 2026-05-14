"""xArm6 FK CBF demo with an analytic spherical obstacle.

Run with: `pixi run test-xarm6-cbf`

The nominal controller drives the arm from a start configuration that has
the end-effector on one side of the workspace to a mirror configuration on
the other side. A spherical obstacle is placed on the midline so that
straight joint-space interpolation would crash the EE into it; the FK CBF
deflects the rollout up and over the obstacle.

Flags:
    --meshcat / --no-meshcat   animate the trajectory in Meshcat (default: on)
    --plot                     also save a matplotlib figure of the trajectory
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pinocchio
from urdf_parser_py import urdf

from pinocchio_ik.qpcbf import PinocchioFKCBF, extract_spheres_from_urdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meshcat",
        dest="meshcat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Animate the trajectory in Meshcat (default: on; pass --no-meshcat to disable)",
    )
    parser.add_argument("--plot", action="store_true",
                        help="Also save a matplotlib figure of the trajectory")
    parser.add_argument("--urdf", default=None,
                        help="Path to URDF (default: launch/xarm6_with_gripper_spherized.urdf)")
    args = parser.parse_args()

    # Always save the plot when running headless (no meshcat) so the demo
    # leaves something behind to look at.
    if not args.meshcat:
        args.plot = True

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    urdf_file = args.urdf or os.path.join(repo_root, "launch", "xarm6_with_gripper_spherized.urdf")

    dt = 0.05
    steps = 50

    # Sphere obstacle sits on the EE's nominal swing line; radius is chosen
    # so the unconstrained trajectory would clearly clip it.
    center = np.array([0.40, 0.0, 0.30])
    radius = 0.1

    def env_sdf(pos):
        return np.linalg.norm(pos - center, axis=-1) - radius

    def env_grad(pos):
        diff = pos - center
        return diff / np.linalg.norm(diff, axis=-1, keepdims=True)

    urdf_model = urdf.URDF.from_xml_file(urdf_file)
    model = pinocchio.buildModelFromUrdf(urdf_file)

    controller = PinocchioFKCBF(model, urdf_model, env_sdf, env_grad,
                                ee_frame_name="xarm6_ee_tip")
    print(f"model nq={model.nq}, controlled joints={controller.controlled_joint_names}")

    # Mirror-image reach poses on either side of the obstacle.
    start_config = np.array([ 0.9, -0.2, -1.0, 0.0, 1.2, 0.0])
    goal_config  = np.array([-0.9, -0.2, -1.0, 0.0, 1.2, 0.0])

    def nominal_controller(q):
        diff = goal_config - q
        n = np.linalg.norm(diff)
        return diff / n if n > 1e-9 else np.zeros_like(diff)

    solutions, controls = controller.controller.solve_ode(
        start_config, nominal_controller, steps, dt,
    )
    print(f"rolled out {solutions.shape[0]} configurations, final q = {solutions[-1]}")

    min_dist = _min_distance_along_trajectory(controller, solutions, env_sdf)
    print(f"min sphere-to-obstacle clearance along trajectory: {min_dist:.4f} m")

    if args.plot:
        _save_plot(solutions, controller, center, radius, here)

    if args.meshcat:
        _animate_meshcat(model, urdf_model, solutions, center, radius)


def _min_distance_along_trajectory(controller, solutions, env_sdf):
    worst = np.inf
    for q in solutions:
        pinocchio.forwardKinematics(controller.model, controller.data, q)
        pinocchio.updateFramePlacements(controller.model, controller.data)
        for link_name, local_pos in controller.sphere_positions.items():
            if local_pos.size == 0 or link_name == controller.urdf_model.get_root():
                continue
            pose = controller.get_frame_pose(link_name)
            world = (pose.rotation @ local_pos[..., np.newaxis]
                     + pose.translation[:, np.newaxis])[..., 0]
            d = env_sdf(world) - controller.sphere_radii[link_name]
            worst = min(worst, float(np.min(d)))
    return worst


def _save_plot(solutions, controller, center, radius, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    tip_xyz = np.empty((solutions.shape[0], 3))
    for i, q in enumerate(solutions):
        pinocchio.forwardKinematics(controller.model, controller.data, q)
        pinocchio.updateFramePlacements(controller.model, controller.data)
        tip_xyz[i] = controller.get_frame_pose("xarm6_ee_tip").translation

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    sx = center[0] + radius * np.cos(u) * np.sin(v)
    sy = center[1] + radius * np.sin(u) * np.sin(v)
    sz = center[2] + radius * np.cos(v)
    ax.plot_surface(sx, sy, sz, color="red", alpha=0.3, linewidth=0)

    ax.plot(tip_xyz[:, 0], tip_xyz[:, 1], tip_xyz[:, 2], "k-", marker=".",
            markersize=2, label="EE trajectory")
    ax.scatter(*tip_xyz[0], color="green", s=40, label="start")
    ax.scatter(*tip_xyz[-1], color="blue", s=40, label="goal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("xArm6 FK CBF: EE trajectory over a spherical obstacle")
    ax.legend()
    ax.set_box_aspect((1, 1, 0.7))

    out = os.path.join(out_dir, "test_xarm6_cbf_trajectory.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def _animate_meshcat(model, urdf_model, solutions, center, radius):
    import meshcat
    import meshcat.geometry as mg
    import meshcat.transformations as mt

    viewer = meshcat.Visualizer().open()

    # Draw collision spheres for each link, updated each frame from FK.
    data = model.createData()
    sphere_positions, sphere_radii = extract_spheres_from_urdf(urdf_model)
    sphere_handles = []
    for link_name, local_pos in sphere_positions.items():
        if local_pos.size == 0 or link_name == urdf_model.get_root():
            continue
        for k, (p_local, r) in enumerate(zip(local_pos, sphere_radii[link_name])):
            path = viewer[f"robot/{link_name}/{k}"]
            path.set_object(mg.Sphere(float(r)),
                            mg.MeshLambertMaterial(color=0x5588ff, opacity=0.7))
            sphere_handles.append((link_name, p_local, path))

    viewer["obstacle"].set_object(
        mg.Sphere(float(radius)),
        mg.MeshLambertMaterial(color=0xff3333, opacity=0.6),
    )
    viewer["obstacle"].set_transform(mt.translation_matrix(center))

    print("Looping Meshcat animation. Ctrl-C to exit.")
    idx = 0
    try:
        while True:
            q = solutions[idx]
            pinocchio.forwardKinematics(model, data, q)
            pinocchio.updateFramePlacements(model, data)
            for link_name, p_local, path in sphere_handles:
                fid = model.getFrameId(link_name)
                pose = data.oMf[fid]
                world = pose.rotation @ p_local + pose.translation
                path.set_transform(mt.translation_matrix(world))
            time.sleep(0.05)
            idx = (idx + 1) % solutions.shape[0]
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
