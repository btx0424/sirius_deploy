import lcm
import time
import itertools
import numpy as np
import argparse
import threading

from scipy.spatial.transform import Rotation as R
from lcm_types import (
    leg_control_command_lcmt, 
    leg_control_data_lcmt, 
    gamepad_lcmt
)
from onnx_module import ONNXModule
from timerfd import Timer


joint_names_isaac = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
    "LF_WHEEL", "LH_WHEEL", "RF_WHEEL", "RH_WHEEL"
]

joint_names_real = [
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_WHEEL", "LF_WHEEL", "RH_WHEEL", "LH_WHEEL"
]

default_joint_pos = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.4000000059604645,
    -0.4000000059604645,
    0.4000000059604645,
    -0.4000000059604645,
    -1.2000000476837158,
    1.2000000476837158,
    -1.2000000476837158,
    1.2000000476837158,
    0.0,
    0.0,
    0.0,
    0.0
]


class LCMControl:
    def __init__(self):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.lc.subscribe("leg_control_data", self.handle_data)
        self.lc.subscribe("gamepad", self.handle_gamepad)
        
        self.buf_jpos = np.zeros((4, 12)) # ignore wheel jpos
        self.buf_jvel = np.zeros((4, 16))
        self.buf_act = np.zeros((2, 16))
        self.jkp = np.zeros(16); self.jkp[:12] = 40.0
        self.jkd = np.zeros(16); self.jkd[:12] = 1.0
        self.act_scaling = np.zeros(16)
        self.act_scaling[:12] = 0.5; self.act_scaling[12:] = 2.0
        self.def_jpos = np.array(default_joint_pos)
        
        self.cmd_lin_vel = np.zeros(2)
        self.cmd_ang_vel = np.zeros(3)
        self.cmd_roll = np.zeros(1)
        self.cmd_pitch = np.zeros(1)
        self.cmd_stand_hei = np.zeros(1)
        self.cmd_phase = np.zeros(1)
        self.cmd_mode = np.array([1, 0, 0, 0])

        self.isaac2real = [joint_names_isaac.index(name) for name in joint_names_real]
        self.real2isaac = [joint_names_real.index(name) for name in joint_names_isaac]
        
        self.Mode = "Passive"
        self.RL_Mode = "RL_Dog"
        self.send = False
        self.gamepad_command = {
            "left_stick": [0.0, 0.0],  # [x, y] values for the left stick
            "right_stick": [0.0, 0.0],  # [x, y] values for the right stick
            "a_button": 0,              # A button state (0 or 1)
            "b_button": 0,              # B button state (0 or 1)
            "y_button": 0,              # B button state (0 or 1)
            "x_button": 0,              # B button state (0 or 1)
            "send": 0,
            "start": 0,
            "back": 0,
            "rightBumper": 0,
            "leftBumper": 0,
            "rightTriggerButton": 0,
            "leftTriggerButton": 0,
            "rightTriggerAnalog": 0.0,
            "leftTriggerAnalog": 0.0,
            "rightStickAnalog": [0.0, 0.0],
            "leftStickAnalog": [0.0, 0.0]
        }
        self.command_msg = leg_control_command_lcmt()

    def start(self):
        self.thread_lcm_receive = threading.Thread(target=self.run)
        self.thread_lcm_receive.start()
    
    def run(self):
        while True:
            self.lc.handle()
    
    def set_command(
        self,
        q_des: np.ndarray,
        qd_des: np.ndarray,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray
    ):
        self.command_msg.q_des = q_des.tolist()
        self.command_msg.qd_des = qd_des.tolist()
        self.command_msg.kp_joint = kp_joint.tolist()
        self.command_msg.kd_joint = kd_joint.tolist()
    
    def publish_command(self):
        self.lc.publish("controller2robot", self.command_msg.encode())
    
    def to_damping_mode(self):
        self.command_msg.q_des = [0.0] * 16
        self.command_msg.qd_des = [0.0] * 16
        self.command_msg.kp_joint = [0.0] * 16
        self.command_msg.kd_joint = [4.0] * 16

    def handle_data(self, channel: str, data: leg_control_data_lcmt):
        self.state_msg = leg_control_data_lcmt.decode(data)

        jpos = np.asarray(self.state_msg.q)[self.real2isaac]
        jvel = np.asarray(self.state_msg.qd)[self.real2isaac]
        self.buf_jpos = np.roll(self.buf_jpos, 1, axis=0)
        self.buf_jpos[0] = jpos[:12] # ignore wheel jpos
        self.buf_jvel = np.roll(self.buf_jvel, 1, axis=0)
        self.buf_jvel[0] = jvel

        self.rot = R.from_quat(data.quat, scalar_first=True)
        self.gyro = data.gyro
    
    def handle_gamepad(self, channel: str, data: gamepad_lcmt):
        msg = gamepad_lcmt.decode(data)
        # Update gamepad command variables
        self.gamepad_command["left_stick"] = msg.leftStickAnalog
        self.gamepad_command["right_stick"] = msg.rightStickAnalog
        self.gamepad_command["a_button"] = msg.a
        self.gamepad_command["b_button"] = msg.b
        self.gamepad_command["y_button"] = msg.y
        self.gamepad_command["x_button"] = msg.x
        self.gamepad_command["rightBumper"] = msg.rightBumper
        self.gamepad_command["leftBumper"] = msg.leftBumper
        self.gamepad_command["start"] = msg.start
        self.gamepad_command["back"] = msg.back
        if self.gamepad_command["start"] == 1 & self.gamepad_command["back"] == 1:
            self.send = True
        # print(Mode)
        if (self.gamepad_command["a_button"] == 1) & self.send:
            self.Mode = "RL"
        elif (self.gamepad_command["b_button"] == 1) & self.send:
            self.Mode = "Stand"
        elif (self.gamepad_command["leftBumper"] == 1) & (self.gamepad_command["rightBumper"] == 1) & self.send:
            self.Mode = "Passive"
        elif (self.gamepad_command["x_button"] == 1) & self.send:
            self.Mode = "Sitdown"
        elif (self.gamepad_command["rightBumper"] == 1) & self.send:
            self.Mode = "Damping"
        if self.gamepad_command["start"]  == 1 & self.gamepad_command["y_button"] == 1:
                self.RL_Mode = "RL_Dog"
        elif self.gamepad_command["back"] == 1 & self.gamepad_command["y_button"] == 1:
                self.RL_Mode = "RL_Stand"
    
    def compute_observation(self):
        projected_gravity = self.rot.inv().apply(np.array([0, 0, -1.]))
        observation = np.concatenate([
            self.gyro,
            projected_gravity,
            self.buf_jpos.flatten(),
            self.buf_jvel.flatten(),
            self.buf_act.flatten(),
        ])
        return observation
    
    def compute_command(self):
        self.cmd_lin_vel[0] = self.gamepad_command["left_stick"][1]
        self.cmd_lin_vel[1] = self.gamepad_command["right_stick"][0]
        self.cmd_ang_vel[2] = self.gamepad_command["left_stick"][0]

        command = np.concatenate([
            self.cmd_lin_vel[:2],
            self.cmd_ang_vel,
            self.cmd_roll,
            self.cmd_pitch,
            self.cmd_phase,
            1 - self.cmd_phase,
            self.cmd_mode
        ])
        return command
    
    def parse_action(self, action: np.ndarray):
        assert len(action) == 16
        action = action * self.act_scaling
        q_des = self.def_jpos.copy(); q_des[:12] += action[:12] # leg actions
        qd_des = np.zeros(16); qd_des[12:] = action[12:] # wheel actions
        return q_des[self.isaac2real], qd_des[self.isaac2real], self.jkp, self.jkd
