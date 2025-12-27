import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("../meshes/single_leg.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        for i in range(model.nu):
            jid = model.actuator(i).trnid[0]
            data.qpos[jid] = data.ctrl[i]
        

        mujoco.mj_forward(model, data)
        viewer.sync()
