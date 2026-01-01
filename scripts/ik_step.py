import os
import time
import numpy as np
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("nmocap =", model.nmocap)

foot_site_id = model.site("foot_site").id
target_site_id = model.site("target_site").id

dof_shoulder = model.joint("shoulder_joint").dofadr[0]  # Index in velocity array for shoulder
dof_wing = model.joint("wing_joint").dofadr[0]          # Index in velocity array for wing
dof_knee = model.joint("knee_joint").dofadr[0]          # Index in velocity array for knee
dof_idxs = np.array([dof_shoulder, dof_wing, dof_knee], dtype=int)

qpos_shoulder = model.joint("shoulder_joint").qposadr[0]  # Index in position array for shoulder
qpos_wing = model.joint("wing_joint").qposadr[0]          # Index in position array for wing
qpos_knee = model.joint("knee_joint").qposadr[0]          # Index in position array for knee
qpos_idxs = np.array([qpos_shoulder, qpos_wing, qpos_knee], dtype=int)

jids = np.array([
    model.joint("shoulder_joint").id,  # Joint ID for shoulder
    model.joint("wing_joint").id,      # Joint ID for wing
    model.joint("knee_joint").id       # Joint ID for knee
], dtype=int)

def clamp_to_limits(q):
    q_clamped = q.copy()
    for i, jid in enumerate(jids):
        lo, hi = model.jnt_range[jid]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped

alpha = 0.6
damping = 1e-2

tol = 1e-2

dt = 0.01
t = 0.0

target_center = np.array([0.0, 0.0, 150.0])

PATH = "step"

amp_xyz = np.array([30.0, 0.0, 0.0])
freq_xyz = np.array([0.55, 0.37, 0.73])
phase_xyz = np.array([0.0, np.pi/3, np.pi/6])

step_freq = 0.6
step_amp_x = 35.0
step_amp_y = 15.0
lift_amp_z = 25.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)

    while viewer.is_running():

        if model.nmocap > 0:
            if PATH == "lissajous":
                x = amp_xyz[0] * np.sin(2*np.pi*freq_xyz[0]*t + phase_xyz[0])
                y = amp_xyz[1] * np.sin(2*np.pi*freq_xyz[1]*t + phase_xyz[1])
                z = amp_xyz[2] * np.sin(2*np.pi*freq_xyz[2]*t + phase_xyz[2])
                data.mocap_pos[0] = target_center + np.array([x, y, z])

            elif PATH == "step":
                theta = 2*np.pi*step_freq*t
                x = step_amp_x * np.sin(theta)
                y = step_amp_y * np.cos(theta)
                lift = 0.5 * (1.0 - np.cos(theta))
                z = lift_amp_z * lift
                data.mocap_pos[0] = target_center + np.array([x, y, z])

        mujoco.mj_forward(model, data)

        p_foot = data.site_xpos[foot_site_id].copy()
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