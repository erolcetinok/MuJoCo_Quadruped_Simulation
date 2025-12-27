import os
import time
import numpy as np
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("nmocap =", model.nmocap)  # should be 1

# IDs
foot_site_id   = model.site("foot_site").id
target_site_id = model.site("target_site").id

# DOF indices (velocity-space) for each hinge joint
dof_shoulder = model.joint("shoulder_joint").dofadr[0]
dof_wing     = model.joint("wing_joint").dofadr[0]
dof_knee     = model.joint("knee_joint").dofadr[0]
dof_idxs = np.array([dof_shoulder, dof_wing, dof_knee], dtype=int)

# QPOS indices (position-space) for each hinge joint
qpos_shoulder = model.joint("shoulder_joint").qposadr[0]
qpos_wing     = model.joint("wing_joint").qposadr[0]
qpos_knee     = model.joint("knee_joint").qposadr[0]
qpos_idxs = np.array([qpos_shoulder, qpos_wing, qpos_knee], dtype=int)

# Joint IDs for clamping
jids = np.array([
    model.joint("shoulder_joint").id,
    model.joint("wing_joint").id,
    model.joint("knee_joint").id
], dtype=int)

def clamp_to_limits(q):
    q_clamped = q.copy()
    for i, jid in enumerate(jids):
        lo, hi = model.jnt_range[jid]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped

# IK params
alpha = 0.6
damping = 1e-2
tol = 1e-2

# Fallback target motion params
dt = 0.01
t = 0.0

# Choose a center position for the target (in YOUR model units)
# Start with whatever you used in XML: body ik_target pos="0 0 150"
target_center = np.array([0.0, 0.0, 150.0])
target_amp = np.array([30.0, 0.0, 0.0])   # move in +X/-X by 30 units
target_freq = 0.6                         # Hz-ish (feel free to tweak)

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)

    while viewer.is_running():

        if model.nmocap > 0:
            data.mocap_pos[0] = target_center + target_amp * np.sin(2*np.pi*target_freq*t)


        mujoco.mj_forward(model, data)


        p_foot   = data.site_xpos[foot_site_id].copy()
        p_target = data.site_xpos[target_site_id].copy()

        err = (p_target - p_foot)

        if np.linalg.norm(err) > tol:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, foot_site_id)

            J = jacp[:, dof_idxs]

            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)

            q = data.qpos[qpos_idxs].copy()
            q_new = q + alpha * dq
            q_new = clamp_to_limits(q_new)

            data.qpos[qpos_idxs] = q_new

            mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(dt)
        t += dt