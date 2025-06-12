import lcm
import time
import itertools
import numpy as np
import argparse
import threading

from scipy.spatial.transform import Rotation as R
from sirius_deploy.interface import LCMControl, MujocoInterface
from sirius_deploy.timerfd import Timer
from sirius_deploy.onnx_module import ONNXModule


class RobotControl:
    def __init__(self, interface: MujocoInterface | LCMControl):
        self.interface = interface
        self.interface.start()

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
        return self.interface.Mode
    
    @property
    def send(self):
        return self.interface.send
    
    def command_handler(self):
        if self.interface.Mode == "Stand":
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

        elif self.interface.Mode == "Sitdown":
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
            self.interface.set_command(q_desired, np.zeros(16), jkp, jkd)
            self.stand_flag = True

        elif self.interface.Mode == "RL":
            q_des, qd_des, jkp, jkd = self.interface.parse_action(self.rl_action)
            self.interface.set_command(q_des, qd_des, jkp, jkd)
            self.stand_flag = True
            self.sitdown_flag = True

        elif self.interface.Mode == "Damping":
            self.interface.to_damping_mode()
        
        if self.interface.send:
            self.interface.publish_command()

    def run(self, policy):
        inp = {
            "is_init": np.array([True]),
            "hx": np.zeros((1, 128), dtype=np.float32),
        }

        dt = 0.01
        timer = Timer(dt); log_interval = 2 // dt

        def should_run_policy(i):
            if policy is None or not self.interface.initialized:
                return False
            return i % 2 == 0

        t = time.perf_counter()
        for i in itertools.count():
            self.interface.update()
            
            if should_run_policy(i):
                inp["command"] = self.interface.compute_command()
                inp["policy"] = self.interface.compute_observation()
                inp["is_init"]  = np.array([False], dtype=bool)
                self.rl_action, carry = policy(inp)
                inp = carry
            
            self.command_handler()
            timer.sleep()

            if i % log_interval == 0:
                main_loop_freq = log_interval / (time.perf_counter() - t)
                t = time.perf_counter()
                print(f"Main Loop Frequency: {main_loop_freq:.2f} Hz")
                print(f"Mode: {self.mode}, send: {self.send}, initialized: {self.interface.initialized}.")



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
    parser.add_argument("-i", "--interface", type=str, default="lcm")
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
    
    if args.interface == "mjc":
        interface = MujocoInterface()
    elif args.interface == "lcm":
        interface = LCMControl(use_gamepad=False)
    else:
        raise ValueError(f"Invalid interface: {args.interface}")
    
    robot_control = RobotControl(interface)
    robot_control.run(policy)


if __name__ == "__main__":
    main()