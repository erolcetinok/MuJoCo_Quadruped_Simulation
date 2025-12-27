import mujoco
import mujoco.viewer

# Load the model (path is relative to this script)
model = mujoco.MjModel.from_xml_path("../meshes/single_leg.xml")
data = mujoco.MjData(model)

# Initialize forward kinematics
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # For position actuators with kp=0, ctrl directly controls qpos
        # Copy control values to joint positions
        for i in range(model.nu):
            jid = model.actuator(i).trnid[0]
            data.qpos[jid] = data.ctrl[i]
        
        # Update kinematics
        mujoco.mj_forward(model, data)
        viewer.sync()
