import mujoco
import mujoco.viewer
import numpy as np
import threading
import itertools
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from sirius_deploy.interface.constants import JOINT_NAMES_ISAAC, DEFAULT_JOINT_POS
from sirius_deploy.interface.lcm_interface import Data
from sirius_deploy.timerfd import Timer

MODEL_PATH = Path(__file__).parent / "sirius_wheel" / "sirius_wheel.xml"

class MujocoInterface:
    def __init__(self, model_path: str = MODEL_PATH):
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        self.joint_names_mujoco = []
        for u in range(self.mj_model.nu):
            actuator = self.mj_model.actuator(u)
            joint = self.mj_model.joint(actuator.trnid[0])
            assert joint.name == actuator.name
            self.joint_names_mujoco.append(joint.name)
        print(self.joint_names_mujoco)

        self.Mode = "RL"
        self.send = True

        self.data = Data()

        self.def_jpos = np.array(DEFAULT_JOINT_POS)
        self.rot = R.identity()
        self.gyro = np.zeros(3)
        self.jvel = np.zeros(16)

        self.cmd_lin_vel = np.zeros(2)
        self.cmd_ang_vel = np.zeros(3)
        self.cmd_roll = np.zeros(1)
        self.cmd_pitch = np.zeros(1)
        self.cmd_stand_hei = np.zeros(1)
        self.cmd_phase = np.zeros(1)
        self.cmd_mode = np.array([1, 0, 0, 0])
        self.task_command = np.zeros(13)

        self.isaac2mujoco = [JOINT_NAMES_ISAAC.index(name) for name in self.joint_names_mujoco]
        self.mujoco2isaac = [self.joint_names_mujoco.index(name) for name in JOINT_NAMES_ISAAC]
        
        self.set_command(*self.parse_action(np.zeros(16)))

    @property
    def initialized(self):
        return True

    def start(self):
        print("MujocoInterface start")
        thread_sim = threading.Thread(target=self.thread_sim)
        thread_sim.start()
    
    def thread_sim(self):
        timer = Timer(self.mj_model.opt.timestep)
        viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        for i in itertools.count():
            self.mj_data.ctrl[:] = (
                self.kp_joint * (self.q_des - self.mj_data.qpos[7:]) +
                self.kd_joint * (self.qd_des - self.mj_data.qvel[6:])
            )
            mujoco.mj_step(self.mj_model, self.mj_data)
            timer.sleep()
            if not viewer.is_running():
                break
            elif i % 2 == 0:
                viewer.sync()
        viewer.close()
    
    def set_command(
        self,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray
    ):
        self.q_des = q_des
        self.qd_des = qd_des
        self.kp_joint = kp_joint
        self.kd_joint = kd_joint
    
    def publish_command(self):
        pass

    def update(self):
        self.jpos = self.mj_data.qpos[7:][self.mujoco2isaac]
        self.jvel = self.mj_data.qvel[6:][self.mujoco2isaac]
        self.quat_wxyz = self.mj_data.qpos[3:7]
        self.rot = R.from_quat(self.quat_wxyz, scalar_first=True)
        self.gyro = self.mj_data.qvel[3:6]
        self.projected_gravity = self.rot.inv().apply(np.array([0, 0, -1.]))

    def compute_observation(self):
        self.data.buf_jpos = np.roll(self.data.buf_jpos, 1, axis=0)
        self.data.buf_jpos[0] = self.jpos[:12] # ignore wheel jpos
        self.data.buf_jvel = np.roll(self.data.buf_jvel, 1, axis=0)
        self.data.buf_jvel[0] = self.jvel

        observation = np.concatenate([
            self.gyro,
            self.projected_gravity,
            self.data.buf_jpos.flatten(),
            self.data.buf_jvel.flatten(),
            self.data.buf_act.flatten(),
        ], dtype=np.float32)
        return observation[None, :]
    
    def compute_command(self):
        # self.cmd_lin_vel[0] = self.gamepad_command["left_stick"][1]
        # self.cmd_lin_vel[1] = self.gamepad_command["right_stick"][0]
        # self.cmd_ang_vel[2] = self.gamepad_command["left_stick"][0]

        command = np.concatenate([
            self.cmd_lin_vel[:2],
            self.cmd_ang_vel,
            self.cmd_roll,
            self.cmd_pitch,
            self.cmd_phase,
            1 - self.cmd_phase,
            self.cmd_mode
        ], dtype=np.float32)
        self.task_command = command
        return command[None, :]
    
    def parse_action(self, action: np.ndarray):
        assert len(action) == 16
        self.data.buf_act = np.roll(self.data.buf_act, 1, axis=0)
        self.data.buf_act[0] = action
        self.data.applied_action = self.data.applied_action * 0.2 + action * 0.8
        leg_action, wheel_action = np.split(self.data.applied_action, [12])

        # leg actions
        q_des = self.def_jpos.copy()
        q_des[:12] += leg_action * self.data.leg_scaling
        # wheel actions
        qd_des = np.zeros(16)
        qd_des[12:] = wheel_action * self.data.wheel_scaling
        return (
            q_des[self.isaac2mujoco],
            qd_des[self.isaac2mujoco],
            self.data.jkp_isaac[self.isaac2mujoco],
            self.data.jkd_isaac[self.isaac2mujoco]
        )


if __name__ == "__main__":
    mjc = MujocoInterface()
    mjc.start()

