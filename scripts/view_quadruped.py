import os
import time
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "quadruped.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

foot_site_FL_id = model.site("foot_site_FL").id
foot_site_FR_id = model.site("foot_site_FR").id
foot_site_BL_id = model.site("foot_site_BL").id
foot_site_BR_id = model.site("foot_site_BR").id

mujoco.mj_forward(model, data)

last_print_time = time.time()
print_interval = 0.5

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for i in range(model.nu):
            jid = model.actuator(i).trnid[0]
            data.qpos[jid] = data.ctrl[i]
        
        # Update kinematics
        mujoco.mj_forward(model, data)
        
        viewer.sync()

