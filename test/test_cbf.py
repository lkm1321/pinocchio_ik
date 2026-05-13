"""Python-only smoke test / demo for the Pinocchio FK CBF.

Run with: `pixi run test-cbf`  (or `pixi run python test/test_cbf.py`)

Flags:
    --meshcat   open a Meshcat browser viewer to animate the trajectory
    --plot      save a matplotlib figure of the trajectory (default)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pinocchio
from urdf_parser_py import urdf

from pinocchio_ik.qpcbf import PinocchioFKCBF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshcat", action="store_true", help="Animate in Meshcat")
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figure (default)")
    parser.add_argument("--urdf", default=None, help="Path to URDF (default: test/two_manipulator_arm.urdf relative to repo root)")
    args = parser.parse_args()

    if not args.meshcat and not args.plot:
        args.plot = True

    here = os.path.dirname(os.path.abspath(__file__))
    urdf_file = args.urdf or os.path.join(here, "two_manipulator_arm.urdf")

    dt = 0.05
    steps = 100

    center = np.array([0.5, 0.5, 0.0])
    radius = 0.075

    def env_sdf(pos):
        return np.linalg.norm(pos - center, axis=-1) - radius

    def env_grad(pos):
        return (pos - center) / np.linalg.norm(pos - center, axis=-1, keepdims=True)

    urdf_model = urdf.URDF.from_xml_file(urdf_file)
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_file)

    controller = PinocchioFKCBF(model, urdf_model, env_sdf, env_grad)
    print(f"model nq={model.nq}, controlled joints={controller.controlled_joint_names}")

    start_config = np.array([-1.0] * model.nq)
    goal_config = np.array([1.0] * model.nq)

    def nominal_controller(q):
        diff = goal_config - q
        n = np.linalg.norm(diff)
        return diff / n if n > 1e-9 else np.zeros_like(diff)

    solutions, controls = controller.controller.solve_ode(
        start_config, nominal_controller, steps, dt,
    )
    print(f"rolled out {solutions.shape[0]} configurations, final q = {solutions[-1]}")

    min_dist = _min_distance_along_trajectory(controller, solutions, env_sdf)
    print(f"min sphere-center to obstacle-center distance (minus sphere/obs radii) along trajectory: {min_dist:.4f}")

    if args.plot:
        _save_plot(solutions, controller, env_sdf, here)

    if args.meshcat:
        from pinocchio.visualize import MeshcatVisualizer as Visualizer
        viz = Visualizer(model, collision_model=collision_model, visual_model=visual_model)
        viz.initViewer(loadModel=True, open=True)
        viz.displayVisuals(False)
        viz.displayCollisions(True)
        print("Looping Meshcat animation. Ctrl-C to exit.")
        idx = 0
        try:
            while True:
                viz.display(solutions[idx])
                time.sleep(0.05)
                idx = (idx + 1) % solutions.shape[0]
        except KeyboardInterrupt:
            pass


def _min_distance_along_trajectory(controller, solutions, env_sdf):
    pin = __import__("pinocchio")
    worst = np.inf
    for q in solutions:
        pin.forwardKinematics(controller.model, controller.data, q)
        pin.updateFramePlacements(controller.model, controller.data)
        for link_name, local_pos in controller.sphere_positions.items():
            if local_pos.size == 0 or link_name == controller.urdf_model.get_root():
                continue
            pose = controller.get_frame_pose(link_name)
            world = (pose.rotation @ local_pos[..., np.newaxis] + pose.translation[:, np.newaxis])[..., 0]
            d = env_sdf(world) - controller.sphere_radii[link_name]
            worst = min(worst, float(np.min(d)))
    return worst


def _save_plot(solutions, controller, env_sdf, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pin = __import__("pinocchio")

    ee_xy = np.empty((solutions.shape[0], 2))
    for i, q in enumerate(solutions):
        pin.forwardKinematics(controller.model, controller.data, q)
        pin.updateFramePlacements(controller.model, controller.data)
        ee_pose = controller.get_frame_pose("end_effector")
        ee_xy[i] = ee_pose.translation[:2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(solutions[:, 0], solutions[:, 1], marker=".")
    axes[0].set_xlabel("joint1 [rad]")
    axes[0].set_ylabel("joint2 [rad]")
    axes[0].set_title("Configuration trajectory")
    axes[0].grid(True)

    axes[1].plot(ee_xy[:, 0], ee_xy[:, 1], marker=".")
    theta = np.linspace(0, 2 * np.pi, 64)
    obs_x = 0.5 + 0.075 * np.cos(theta)
    obs_y = 0.5 + 0.075 * np.sin(theta)
    axes[1].plot(obs_x, obs_y, "r-", label="obstacle")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("End-effector workspace trace")
    axes[1].set_aspect("equal")
    axes[1].legend()
    axes[1].grid(True)

    out = os.path.join(out_dir, "test_cbf_trajectory.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
