import os
import time
import numpy as np
import mujoco
import mujoco.viewer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "quadruped.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"Model loaded: {model.nmocap} mocap bodies, {model.nu} actuators")

LEGS = ["FL", "FR", "BL", "BR"]  # Front Left, Front Right, Back Left, Back Right

# Store joint information for each leg
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
    """Clamp joint angles for a specific leg to their limits."""
    q_clamped = q.copy()
    for i, jid in enumerate(leg_info[leg]['jids']):
        lo, hi = model.jnt_range[jid]
        q_clamped[i] = np.clip(q_clamped[i], lo, hi)
    return q_clamped

alpha = 0.3      # Step size for gradient descent
damping = 1e-2   # Damping for regularization
tol = 0.5        # Tolerance for convergence
dt = 0.01        # Time step

# Motion mode: circle, static, or trot
motion_mode = "trot" 

circle_freq = 1.5
circle_phase_offset = {"FL": 0.0, "FR": np.pi, "BL": np.pi, "BR": 1.5*np.pi}

trot_params = {
    'step_length': 30.0,      # How far forward/back the foot moves
    'step_height': 15.0,     # How high the foot lifts during swing
    'step_frequency': 0.8,    # Steps per second
    'stance_width': 0.0,      # Lateral offset (can be 0 for straight walking)
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_forward(model, data)
    t = 0.0
    
    # Store initial mocap positions at startup
    initial_mocap_positions = {}
    for leg in LEGS:
        mocap_id = leg_info[leg]['mocap_id']
        initial_mocap_positions[leg] = data.mocap_pos[mocap_id].copy()

    while viewer.is_running():
        if motion_mode == "circle":
            for leg in LEGS:
                mocap_id = leg_info[leg]['mocap_id']
                initial_pos = initial_mocap_positions[leg]
                theta = 2 * np.pi * circle_freq * t + circle_phase_offset[leg]
                offset = np.array([
                    circle_radius * np.cos(theta),
                    circle_radius * np.sin(theta),
                    0.0
                ])
                data.mocap_pos[mocap_id] = initial_pos + offset
                
        elif motion_mode == "trot":
            # Trot gait: diagonal pairs (FL+BR) and (FR+BL) move together
            
            phase = 2 * np.pi * trot_params['step_frequency'] * t
            
            for leg in LEGS:
                mocap_id = leg_info[leg]['mocap_id']
                initial_pos = initial_mocap_positions[leg]
                
                # Determine if this leg is in swing or stance phase
                if leg in ["FL", "BR"]:
                    # FL and BR are in phase
                    leg_phase = phase
                else:  # FR and BL
                    # FR and BL are 180˚ out of phase
                    leg_phase = phase + np.pi
                
                # Normalize phase to [0, 2π]
                leg_phase = leg_phase % (2 * np.pi)
                
                # Determine direction along x=y diagonal for each leg
                # Each leg moves along its natural diagonal axis
                if leg == "FL":
                    # Front Left: moves along +x, +y diagonal
                    diag_x_sign, diag_y_sign = 1.0, 1.0
                elif leg == "FR":
                    # Front Right: moves along +x, -y diagonal
                    diag_x_sign, diag_y_sign = 1.0, -1.0
                elif leg == "BL":
                    # Back Left: moves along -x, +y diagonal
                    diag_x_sign, diag_y_sign = -1.0, 1.0
                else:  # BR
                    # Back Right: moves along -x, -y diagonal
                    diag_x_sign, diag_y_sign = -1.0, -1.0
                
                diag_component = trot_params['step_length'] / np.sqrt(2.0)
                
                if leg_phase < np.pi:
                    # SWING PHASE: foot lifts and moves forward along diagonal
                    swing_progress = leg_phase / np.pi  # 0 to 1
                    
                    # Forward motion along diagonal: starts at -step_length/2, ends at +step_length/2
                    diag_offset = diag_component * (swing_progress - 0.5)
                    x_offset = diag_offset * diag_x_sign
                    y_offset = diag_offset * diag_y_sign
                    
                    # Lift motion: smooth curve (0 -> 1 -> 0)
                    z_offset = trot_params['step_height'] * np.sin(swing_progress * np.pi)
                else:
                    # STANCE PHASE: foot on ground, moves backward along diagonal
                    stance_progress = (leg_phase - np.pi) / np.pi  # 0 to 1
                    
                    # Backward motion along diagonal: starts at +step_length/2, ends at -step_length/2
                    diag_offset = diag_component * (0.5 - stance_progress)
                    x_offset = diag_offset * diag_x_sign
                    y_offset = diag_offset * diag_y_sign
                    
                    # Foot stays on ground
                    z_offset = 0.0
                
                # Apply offset along x=y diagonal
                offset = np.array([x_offset, y_offset, z_offset])
                data.mocap_pos[mocap_id] = initial_pos + offset

        mujoco.mj_forward(model, data)

        for leg in LEGS:
            foot_site_id = leg_info[leg]['foot_site_id']
            target_site_id = leg_info[leg]['target_site_id']
            dof_idxs = leg_info[leg]['dof_idxs']
            qpos_idxs = leg_info[leg]['qpos_idxs']

            # Get current foot and target positions
            p_foot = data.site_xpos[foot_site_id].copy()
            p_target = data.site_xpos[target_site_id].copy()
            err = p_target - p_foot

            # Solve IK if error is large enough
            err_norm = np.linalg.norm(err)
            if err_norm > tol:
                # Compute Jacobian
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, foot_site_id)

                # Extract columns for this leg's joints
                J = jacp[:, dof_idxs]

                # Solve using Damped Least Squares
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err)

                # Scale step size based on error magnitude for smoother convergence
                # Smaller steps when close to target
                error_scale = min(1.0, err_norm / 10.0)  # Scale down when error < 10
                dq_scaled = alpha * error_scale * dq

                # Update joint angles
                q = data.qpos[qpos_idxs].copy()
                q_new = q + dq_scaled
                q_new = clamp_to_limits(leg, q_new)
                data.qpos[qpos_idxs] = q_new

        # Recompute forward kinematics with updated joint angles
        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(dt)
        t += dt

