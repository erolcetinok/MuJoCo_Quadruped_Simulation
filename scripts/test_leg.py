import os
import time
import mujoco
import mujoco.viewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "meshes", "single_leg.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

foot_site_id = model.site("foot_site").id

data.ctrl[:] = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_print = 0.0

    while viewer.is_running():
        mujoco.mj_step(model, data)

        if data.time - last_print > 0.1:
            x, y, z = data.site_xpos[foot_site_id]
            print(f"ctrl={data.ctrl[:]}   foot_site xyz: {x:.2f}, {y:.2f}, {z:.2f}")
            last_print = data.time

        viewer.sync()
        time.sleep(0.01)