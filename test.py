import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/home/auvc2/mujoco_AUVC/model/sub/sub.xml')
d = mujoco.MjData(m)

def rover_physics_update_plug(model, data):
    water_gain = 470.0
    volume_displaced = 0.4572 * 0.33782 * 0.254
    max_buoyancy_force = 9.806 * volume_displaced * 1000
    start_height = 2.0
    force = (start_height - data.qpos[2]) * water_gain

    # Capture Orientation
    # [Px, Py, Pz, Ow, Ox, Oy, Oz]
    # Extract quaternion: [w, x, y, z]
    qorn = np.array([
        data.sensordata[0],
        data.sensordata[1],
        data.sensordata[2],
        data.sensordata[3]
    ])

    orn = quaternion_to_euler(qorn[1], qorn[2], qorn[3], qorn[0])  # [roll, pitch, yaw]

    if data.qpos[2] <= start_height - 0.1:
        fnew = min(force, max_buoyancy_force)
        for i in range(6, 10):
            data.ctrl[i] = fnew / 4.0
    elif data.qpos[2] >= start_height + 0.1:
        downward_force = ((1 - data.qpos[2]) * water_gain) / 4.0
        for i in range(6, 10):
            data.ctrl[i] = downward_force

    return True


def quaternion_to_euler(x, y, z, w):
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return [roll, pitch, yaw]

class Controller():
  def __init__(self,m,d):
    self._d = d
    self._m = m
    pass

  def forward(self):
    d.ctrl[0] = 10
    pass

control = Controller(m,d)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    control.forward()
    # print(d.sensordata)
    print(quaternion_to_euler(d.sensordata[0],d.sensordata[1],d.sensordata[2],d.sensordata[3]))
    rover_physics_update_plug(m,d)
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
  
