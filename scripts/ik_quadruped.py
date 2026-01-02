import os
import time
import numpy as np
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "quadruped.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

LEGS = ["FL", "FR", "BL", "BR"]

leg_info = {}
for leg in LEGS:
    leg_info[leg] = {
        'foot_site_id': model.site(f"foot_site_{leg}").id,
        'target_site_id': model.site(f"target_site_{leg}").id,
        'mocap_id': model.body(f"ik_target_{leg}").mocapid[0],
        'dof_idxs': np.array([
            model.joint(f"shoulder_joint_{leg}").dofadr[0],
            model.joint(f"wing_joint_{leg}").dofadr[0],
            model.joint(f"knee_joint_{leg}").dofadr[0]
        ], dtype=int),
        'qpos_idxs': np.array([
            model.joint(f"shoulder_joint_{leg}").qposadr[0],
            model.joint(f"wing_joint_{leg}").qposadr[0],
            model.joint(f"knee_joint_{leg}").qposadr[0]
        ], dtype=int),
        'jids': np.array([
            model.joint(f"shoulder_joint_{leg}").id,
            model.joint(f"wing_joint_{leg}").id,
            model.joint(f"knee_joint_{leg}").id
        ], dtype=int)
    }

def clamp_to_limits(leg, q):
    q_clamped = q.copy()
    for i, jid in enumerate(leg_info[leg]['jids']):
        lo, hi = model.jnt_range[jid]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped

# IK parameters
alpha = 0.3
damping = 1e-2
tol = 0.5
dt = 0.01

# Gait parameters
motion_mode = "walk"  # "walk" or "trot"

trot_params = {
    'step_length': 30.0,
    'step_height': 15.0,
    'step_frequency': 0.8,
}

walk_params = {
    'step_length': 30.0,
    'step_height': 15.0,
    'step_frequency': 0.6,
}

def compute_gait_offset(leg, leg_phase, params):
    """Compute foot position offset for a leg based on gait phase"""
    diag_component = params['step_length'] / np.sqrt(2.0)
    
    if leg_phase < np.pi:
        # Swing phase: foot lifts and moves forward
        swing_progress = leg_phase / np.pi
        diag_offset = diag_component * (swing_progress - 0.5)
        z_offset = params['step_height'] * np.sin(swing_progress * np.pi)
    else:
        # Stance phase: foot on ground, moves backward
        stance_progress = (leg_phase - np.pi) / np.pi
        diag_offset = diag_component * (0.5 - stance_progress)
        z_offset = 0.0
    
    # All legs move along positive x=y diagonal
    x_offset = diag_offset
    y_offset = diag_offset
    
    return np.array([x_offset, y_offset, z_offset])

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)
    t = 0.0
    
    # Store initial mocap positions at startup
    initial_mocap_positions = {}
    for leg in LEGS:
        mocap_id = leg_info[leg]['mocap_id']
        initial_mocap_positions[leg] = data.mocap_pos[mocap_id].copy()

    while viewer.is_running():
        if motion_mode == "trot":
            phase = 2 * np.pi * trot_params['step_frequency'] * t
            for leg in LEGS:
                mocap_id = leg_info[leg]['mocap_id']
                initial_pos = initial_mocap_positions[leg]
                
                # Diagonal pairs: FL+BR in phase, FR+BL 180° out of phase
                leg_phase = phase if leg in ["FL", "BR"] else phase + np.pi
                leg_phase = leg_phase % (2 * np.pi)
                
                offset = compute_gait_offset(leg, leg_phase, trot_params)
                data.mocap_pos[mocap_id] = initial_pos + offset
                
        elif motion_mode == "walk":
            phase = 2 * np.pi * walk_params['step_frequency'] * t
            leg_phase_offsets = {
                "FL": 0.0,
                "FR": np.pi / 2,
                "BL": np.pi,
                "BR": 3 * np.pi / 2
            }
            
            for leg in LEGS:
                mocap_id = leg_info[leg]['mocap_id']
                initial_pos = initial_mocap_positions[leg]
                
                leg_phase = (phase + leg_phase_offsets[leg]) % (2 * np.pi)
                offset = compute_gait_offset(leg, leg_phase, walk_params)
                data.mocap_pos[mocap_id] = initial_pos + offset

        mujoco.mj_forward(model, data)

        # Solve IK for each leg
        for leg in LEGS:
            foot_site_id = leg_info[leg]['foot_site_id']
            target_site_id = leg_info[leg]['target_site_id']
            dof_idxs = leg_info[leg]['dof_idxs']
            qpos_idxs = leg_info[leg]['qpos_idxs']

            p_foot = data.site_xpos[foot_site_id].copy()
            p_target = data.site_xpos[target_site_id].copy()
            err = p_target - p_foot
            err_norm = np.linalg.norm(err)

            if err_norm > tol:
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, None, foot_site_id)

                J = jacp[:, dof_idxs]
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)

                error_scale = min(1.0, err_norm / 10.0)
                dq_scaled = alpha * error_scale * dq

                q = data.qpos[qpos_idxs].copy()
                q_new = q + dq_scaled
                q_new = clamp_to_limits(leg, q_new)
                data.qpos[qpos_idxs] = q_new

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(dt)
        t += dt

