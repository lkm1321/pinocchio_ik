# pinocchio_ik

Pinocchio-based velocity IK and forward-kinematics Control Barrier Function (FK CBF)
for collision avoidance. Ships as a ROS 2 (`ament_python`) package, but the CBF core
is pure Python and can be exercised standalone via the included pixi environment.

## Layout

```
pinocchio_ik/
  qpcbf.py           QPCBF, PinocchioFKCBF, PointCloudSDF, table_sdf   (pure Python)
  meshcat_utils.py   Meshcat browser wrapper with a WebGL probe        (pure Python)
  velocity_ik.py     ROS 2 node: velocity IK from a desired EE twist
  cbf_node.py        ROS 2 nodes: point-cloud KDTree node and CBF node
test/
  two_manipulator_arm.urdf   Toy planar 2-link arm used by the demos
  test_cbf.py                FK CBF demo with an analytic obstacle SDF
  test_pointcloud_sdf.py     FK CBF demo with a KDTree SDF over a sampled obstacle
  test_webgl.py              Generates webgl_check.html for browser-side WebGL diagnosis
launch/
  *.launch                   ROS 2 launch files
  xarm6_with_gripper_spherized.urdf   xArm6 with collision spheres
```

## ROS 2 use

Standard `colcon build` into a ROS 2 workspace. Entry points (`setup.py`):

- `velocity_ik`  — velocity IK from `desired_twist` to `joint_velocity_cmd`.
- `distance_cbf` — filters `nominal_joint_velocity` through the FK CBF QP.

Launch files in `launch/` wire these up against the xArm6 URDF.

## Python-only use (via pixi)

The CBF core (`QPCBF`, `PinocchioFKCBF`, `PointCloudSDF`) doesn't import any ROS
modules — you can drive it directly from Python. The pixi env in `pyproject.toml`
pulls pinocchio, cvxpy, osqp, meshcat-python, matplotlib, and `urdf-parser-py`
from conda-forge / PyPI.

```bash
pixi install
pixi run test-cbf                       # FK CBF with an analytic sphere SDF
pixi run test-pointcloud-sdf            # FK CBF with a KDTree SDF over a point cloud
pixi run test-pointcloud-sdf -- --meshcat   # same, plus live meshcat viewer
pixi run test-webgl                     # browser-side WebGL diagnostic
```

Each demo prints what it computed and (where applicable) writes a matplotlib
PNG into `test/`:

- `test/test_cbf_trajectory.png` — configuration trajectory + EE workspace trace
  around an analytic spherical obstacle.
- `test/test_pointcloud_sdf.png` — the `PointCloudSDF` evaluated on a z=0 slice
  with the cloud samples and the FK CBF rollout overlaid.

## Minimal Python API

```python
import numpy as np
import pinocchio, urdf_parser_py.urdf as urdf
from pinocchio_ik.qpcbf import PinocchioFKCBF, PointCloudSDF

# Build an SDF over a static point cloud (returns (distance, gradient)).
sdf = PointCloudSDF(points=my_xyz_points, buffer=0.0)

# Load the robot and construct the CBF.
urdf_model = urdf.URDF.from_xml_file("test/two_manipulator_arm.urdf")
model = pinocchio.buildModelFromUrdf("test/two_manipulator_arm.urdf")
cbf = PinocchioFKCBF(model, urdf_model, sdf)   # env_grad=None: sdf returns (d, g)

q = np.zeros(model.nq)
u_nominal = np.array([1.0, 0.5])
u_safe = cbf.get_control(q, u_nominal)         # QP-filtered joint velocity
```

`PinocchioFKCBF` also accepts `controlled_joint_names=[...]` and
`ee_frame_name="..."`; defaults are all model joints and identity weighting.

## Meshcat / WebGL

`pixi run test-pointcloud-sdf -- --meshcat` opens a browser to a thin wrapper
page (`test/meshcat_wrapper.html`) that iframes the live meshcat URL and runs a
WebGL probe in the outer page. If WebGL is unavailable, a yellow banner appears
above the iframe instead of leaving the user staring at a blank canvas. For
deeper diagnosis run `pixi run test-webgl` and open the resulting
`test/webgl_check.html` in the same browser.
