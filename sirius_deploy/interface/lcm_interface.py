import lcm
import time
import itertools
import numpy as np
import argparse
import threading
import h5py
import os

from scipy.spatial.transform import Rotation as R
from sirius_deploy.interface.lcm_types import (
    leg_control_command_lcmt, 
    leg_control_data_lcmt, 
    gamepad_lcmt
)
from sirius_deploy.interface.constants import JOINT_NAMES_ISAAC, DEFAULT_JOINT_POS
from sirius_deploy.timerfd import Timer


joint_names_real = [
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RF_WHEEL", "LF_WHEEL", "RH_WHEEL", "LH_WHEEL"
]

class Data:
    def __init__(self):
        self.buf_jpos = np.zeros((4, 12)) # ignore wheel jpos
        self.buf_jvel = np.zeros((4, 16))
        self.buf_act = np.zeros((2, 16))

        self.jkp = np.zeros(16)
        self.jkp[:12] = 40.0
        self.jkd = np.zeros(16)
        self.jkd[:12] = 1.0; self.jkd[12:] = 5.0

        self.wheel_scaling = 10.0
        self.leg_scaling = 0.5
        self.applied_action = np.zeros(16)


class LCMControl:
    def __init__(self, use_gamepad: bool):
        self.lc_state = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.lc_state.subscribe("robot2controller", self.handle_data)
        
        self.use_gamepad = use_gamepad
        if self.use_gamepad:
            self.lc_gamepad = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
            self.lc_gamepad.subscribe("gamepad2controller", self.handle_gamepad)
        
        self.data = Data()

        self.def_jpos = np.array(DEFAULT_JOINT_POS)
        self.rot = R.identity()
        self.gyro = np.zeros(3)
        
        self.cmd_lin_vel = np.zeros(2)
        self.cmd_ang_vel = np.zeros(3)
        self.cmd_roll = np.zeros(1)
        self.cmd_pitch = np.zeros(1)
        self.cmd_stand_hei = np.zeros(1)
        self.cmd_phase = np.zeros(1)
        self.cmd_mode = np.array([1, 0, 0, 0])
        self.task_command = np.zeros(13)

        self.isaac2real = [JOINT_NAMES_ISAAC.index(name) for name in joint_names_real]
        self.real2isaac = [joint_names_real.index(name) for name in JOINT_NAMES_ISAAC]
        
        self.Mode = "RL" # "Passive"
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
        self.state_initialized = False
        self.gamepad_initialized = False
        self.step_count = 0

        # logging
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{timestr}"
        os.makedirs(self.log_dir, exist_ok=True)
        command_dim = 13
        init_length = 1000
        self.log_file = h5py.File(f"{self.log_dir}/log.h5", "w")
        self.log_file.attrs["ptr"] = 0 # pointer to the current index of the log
        self.log_file.attrs["max_ptr"] = init_length
        self.log_file.create_dataset("gyro", data=np.zeros((init_length, 3)), maxshape=(None, 3))
        self.log_file.create_dataset("projected_gravity", data=np.zeros((init_length, 3)), maxshape=(None, 3))
        self.log_file.create_dataset("jpos", data=np.zeros((init_length, 16)), maxshape=(None, 16))
        self.log_file.create_dataset("jvel", data=np.zeros((init_length, 16)), maxshape=(None, 16))
        self.log_file.create_dataset("action", data=np.zeros((init_length, 16)), maxshape=(None, 16))
        self.log_file.create_dataset("quat", data=np.zeros((init_length, 4)), maxshape=(None, 4))
        self.log_file.create_dataset("command", data=np.zeros((init_length, command_dim)), maxshape=(None, command_dim))

    @property
    def initialized(self):
        if self.use_gamepad:
            return self.state_initialized and self.gamepad_initialized
        else:
            return self.state_initialized
        
    def start(self):
        print("LCMControl start")
        thread_lc_state = threading.Thread(target=self.thread_lc_state)
        thread_lc_state.start()
        if self.use_gamepad:
            thread_lc_gamepad = threading.Thread(target=self.thread_lc_gamepad)
            thread_lc_gamepad.start()
    
    def update(self):
        """
        Update states and log data. This method is called at a frequency higher than 
        the control frequency but lower than the lcm handle frequency.
        """
        if not self.initialized:
            return
        self.projected_gravity = self.rot.inv().apply(np.array([0, 0, -1.]))

        max_ptr = self.log_file.attrs["max_ptr"]
        self.log_file["gyro"][self.step_count] = self.gyro
        self.log_file["projected_gravity"][self.step_count] = self.projected_gravity
        self.log_file["jpos"][self.step_count] = self.jpos
        self.log_file["jvel"][self.step_count] = self.jvel
        self.log_file["quat"][self.step_count] = self.quat_xyzw
        self.log_file["action"][self.step_count] = self.applied_action
        self.log_file["command"][self.step_count] = self.task_command
        if self.step_count == max_ptr-1: # expand the log file
            max_ptr += 1000
            # resize all datasets
            for key in self.log_file.keys():
                self.log_file[key].resize(max_ptr, axis=0)
            self.log_file.flush()
            print(f"Expanded log file to {max_ptr} steps")
        self.step_count += 1
        self.log_file.attrs["ptr"] = self.step_count
        self.log_file.attrs["max_ptr"] = max_ptr
    
    def thread_lc_state(self):
        while True:
            self.lc_state.handle()
            time.sleep(0.001)
    
    def thread_lc_gamepad(self):
        while True:
            self.lc_gamepad.handle()
            time.sleep(0.001)
    
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
        self.lc_state.publish("controller2robot", self.command_msg.encode())
    
    def to_damping_mode(self):
        self.command_msg.q_des = [0.0] * 16
        self.command_msg.qd_des = [0.0] * 16
        self.command_msg.kp_joint = [0.0] * 16
        self.command_msg.kd_joint = [4.0] * 16

    def handle_data(self, channel: str, data: leg_control_data_lcmt):
        self.state_msg = leg_control_data_lcmt.decode(data)

        self.jpos = np.asarray(self.state_msg.q)[self.real2isaac]
        self.jvel = np.asarray(self.state_msg.qd)[self.real2isaac]

        self.quat_xyzw = np.asarray(self.state_msg.quat)[[1, 2, 3, 0]]
        self.rot = R.from_quat(self.quat_xyzw)
        self.gyro = np.asarray(self.state_msg.gyro)
        if not self.state_initialized:
            print("State Initialized.")
            self.state_initialized = True
    
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
        if not self.gamepad_initialized:
            print("Gamepad Initialized.")
            self.gamepad_initialized = True
    
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
            q_des[self.isaac2real],
            qd_des[self.isaac2real],
            self.data.jkp[self.isaac2real],
            self.data.jkd[self.isaac2real]
        )
