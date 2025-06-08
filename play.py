import lcm
import time
import itertools
import numpy as np
import argparse
import threading

from scipy.spatial.transform import Rotation as R
from onnx_module import ONNXModule
from timerfd import Timer
from lcm_interface import LCMControl
from tkinter import ttk
import tkinter as tk


class RobotControlUI:
    def __init__(self, lcm_interface: LCMControl):
        self.lcm = lcm_interface
        self.root = tk.Tk()
        self.root.title("Robot Control Interface")
        self.lcm.gamepad_command = {"left_stick": [0.0, 0.0], "right_stick": [0.0, 0.0]
                                    }
        # Create buttons
        ttk.Button(self.root, text="Stand", command=self.stand).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.root, text="Sit", command=self.sit).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.root, text="RL Mode", command=self.rl_mode).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.root, text="Damping", command=self.damping).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.root, text="Exit", command=self.exit).pack(fill=tk.X, padx=5, pady=5)
        
        # Create sliders
        self.forward_slider = ttk.Scale(self.root, from_=-1.0, to=1.0, orient="horizontal", command=self.update_forward)
        self.forward_slider.set(0.0)
        self.forward_slider.pack(fill=tk.X, padx=5, pady=5)

        self.lateral_slider = ttk.Scale(self.root, from_=-1.0, to=1.0, orient="horizontal", command=self.update_lateral)
        self.lateral_slider.set(0.0)
        self.lateral_slider.pack(fill=tk.X, padx=5, pady=5)

        self.rotation_slider = ttk.Scale(self.root, from_=-1.0, to=1.0, orient="horizontal", command=self.update_rotation)
        self.rotation_slider.set(0.0)
        self.rotation_slider.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.root, text="STOP", command=self.stop).pack(fill=tk.X, padx=5, pady=5)
        # Status label
        self.status = tk.StringVar()
        self.status.set("Ready")
        ttk.Label(self.root, textvariable=self.status).pack()
        
    def update_forward(self, value):
        self.lcm.gamepad_command["left_stick"][1] = float(value)

    def update_lateral(self, value):
        self.lcm.gamepad_command["left_stick"][0] = float(value)

    def update_rotation(self, value):
        self.lcm.gamepad_command["right_stick"][0] = float(value)
        
    def stand(self):
        self.lcm.Mode = "Stand"
        self.status.set("Stand command sent")
    
    def sit(self):
        self.lcm.Mode = "Sitdown"
        self.status.set("Sit command sent")
    
    def rl_mode(self):
        self.lcm.Mode = "RL"
        self.status.set("RL mode activated")
    
    def damping(self):
        self.lcm.Mode = "Damping"
        self.status.set("Damping mode activated")

    def stop(self):
        self.forward_slider.set(0.0)
        self.lateral_slider.set(0.0)
        self.rotation_slider.set(0.0)
        self.lcm.gamepad_command["left_stick"] = [0.0, 0.0]
        self.lcm.gamepad_command["right_stick"] = [0.0, 0.0]    
        self.status.set("STOPPED")
    
    def exit(self):
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


class RobotControl:
    def __init__(self):
        self.lcm_interface = LCMControl(use_gamepad=False)
        self.lcm_interface.start()
        self.lcm_interface.send = True

        self.stand_flag = True
        self.sitdown_flag = True

        self.jpos_sitdown = np.array([
            0.0, 2.7, -2.6,
            0.0, 2.7, -2.6,
            0.0, -2.7, 2.6,
            0.0, -2.7, 2.6,
            0.0, 0.0, 0.0, 0.0
        ])
        self.jpos_stand = np.array([
            0.0, 0.7, -1.6,
            0.0, 0.7, -1.6,
            0.0, -0.7, 1.6,
            0.0, -0.7, 1.6,
            0.0, 0.0, 0.0, 0.0
        ])
        self.jpos_pre_stand = np.array([
            0.0, 1.4, -2.6,
            0.0, 1.4, -2.6,
            0.0, -1.4, 2.6,
            0.0, -1.4, 2.6,
            0.0, 0.0, 0.0, 0.0
        ])
        self.stand_kp, self.stand_kd = 120.0, 2.0
        self.sitdown_kp, self.sitdown_kd = 100.0, 2.0
        
        self.rl_action = np.zeros(16, dtype=np.float32)
    
    @property
    def mode(self):
        return self.lcm_interface.Mode
    
    @property
    def send(self):
        return self.lcm_interface.send
    
    def command_handler(self):
        if self.lcm_interface.Mode == "Stand":
            if self.stand_flag == True:
                self.stand_time_start = time.time()
                self.q_current = np.asarray(self.lcm_interface.state_msg.q)
                self.stand_flag = False
            stand_time = time.time() - self.stand_time_start
            q_desired = cubicBezier(self.q_current, self.jpos_stand, 0.5 * stand_time)
            q_desired[12:] = 0.0
            jkp = np.zeros(16); jkp[:12] = self.stand_kp
            jkd = np.zeros(16); jkd[:12] = self.stand_kd
            self.lcm_interface.set_command(q_desired, np.zeros(16), jkp, jkd)
            self.sitdown_flag = True

        elif self.lcm_interface.Mode == "Sitdown":
            if self.sitdown_flag == True:
                self.sitdown_time_start = time.time()
                self.q_current = np.asarray(self.lcm_interface.state_msg.q)
                self.sitdown_flag = False
            sitdown_time = time.time() - self.sitdown_time_start
            if sitdown_time <= 2.0:
                q_desired = cubicBezier(self.q_current, self.jpos_sitdown, 0.5*sitdown_time)
            else:
                q_desired = cubicBezier(self.jpos_sitdown, self.jpos_pre_stand, 0.5*(sitdown_time-2.0))
            q_desired[12:] = 0.0
            jkp = np.zeros(16); jkp[:12] = self.sitdown_kp
            jkd = np.zeros(16); jkd[:12] = self.sitdown_kd
            self.lcm_interface.set_command(q_desired, np.zeros(16), jkp, jkd)
            self.stand_flag = True

        elif self.lcm_interface.Mode == "RL":
            q_des, qd_des, jkp, jkd = self.lcm_interface.parse_action(self.rl_action)
            self.lcm_interface.set_command(q_des, qd_des, jkp, jkd)
            self.stand_flag = True
            self.sitdown_flag = True

        elif self.lcm_interface.Mode == "Damping":
            self.lcm_interface.to_damping_mode()
        
        if self.lcm_interface.send:
            self.lcm_interface.publish_command()

    def run(self, policy):
        inp = {
            "is_init": np.array([True]),
            "hx": np.zeros((1, 128), dtype=np.float32),
        }

        dt = 0.01
        timer = Timer(dt); log_interval = 2 // dt

        def should_run_policy(i):
            if policy is None or not self.lcm_interface.initialized:
                return False
            return i % 2 == 0

        t = time.perf_counter()
        for i in itertools.count():
            self.lcm_interface.update()
            
            if should_run_policy(i):
                inp["command"] = self.lcm_interface.compute_command()
                inp["policy"] = self.lcm_interface.compute_observation()
                inp["is_init"]  = np.array([False], dtype=bool)
                self.rl_action, carry = policy(inp)
                inp = carry
            
            self.command_handler()
            timer.sleep()

            if i % log_interval == 0:
                main_loop_freq = log_interval / (time.perf_counter() - t)
                t = time.perf_counter()
                print(f"Main Loop Frequency: {main_loop_freq:.2f} Hz")
                print(f"Mode: {self.mode}, send: {self.send}, initialized: {self.lcm_interface.initialized}.")
                print(f"State initialized: {self.lcm_interface.state_initialized}, Gamepad initialized: {self.lcm_interface.gamepad_initialized}.")



def cubicBezier(y0, yf, x):
    if x <= 0:
        x = 0
    elif x >= 1:
        x = 1
    yDiff = yf - y0
    bezier = x * x * x + (x * x * (1 - x))
    # print(y0)
    return y0 + bezier * yDiff

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None)
    args = parser.parse_args()

    if args.path is not None:
        policy_module = ONNXModule(args.path)
        def policy(inp):
            out = policy_module(inp)
            action = out["action"].reshape(-1)
            carry = {k[1]: v for k, v in out.items() if k[0] == "next"}
            return action, carry
    else:
        policy = None
    
    robot_control = RobotControl()
    robot_control.run(policy)


if __name__ == "__main__":
    main()