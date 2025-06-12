from pynput import keyboard
import numpy as np

class KeyboardController:
    def __init__(self) -> None:
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        self.cmd_time = 0.
        self.jump = False
        self.des_contact = np.array([0, 0, 0, 0])
    
    def update(self):
        self.x = 0.
        self.y = 0.
        self.yaw_rate = 0.
        self.handle_keys()
        
        if self.jump:
            self.cmd_time += 0.02
            if self.cmd_time > 0.3:
                self.des_contact = np.array([-1, -1, -1, -1])
            if self.cmd_time > 0.7:
                self.des_contact = np.array([0, 0, 0, 0])
            if self.cmd_time > 1.:
                self.jump = False
                self.cmd_time = 0.
                self.des_contact = np.array([0, 0, 0, 0])
    
    def handle_keys(self):
        if 'w' in self.keys_pressed:
            self.x = 1.0
        if 'a' in self.keys_pressed:
            self.y = 0.8
        if 's' in self.keys_pressed:
            self.x = -1.0
        if 'd' in self.keys_pressed:
            self.y = -0.8
        
        if 'Key.left' in self.keys_pressed:
            self.yaw_rate += 1.0
        if 'Key.right' in self.keys_pressed:
            self.yaw_rate -= 1.0
        if 'Key.down' in self.keys_pressed:
            self.des_contact = np.array([-1, 0, -1, 0])
            self.stand = True
        elif 'Key.up' in self.keys_pressed:
            self.des_contact = np.array([0, -1, 0, 1])
            self.stand = True
        
        if "Key.space" in self.keys_pressed:
            if not self.jump:
                self.cmd_time = 0.
            self.jump = True

    def on_press(self, key):
        try:
            # For regular keys
            self.keys_pressed.add(key.char)
        except AttributeError:
            # For special keys like arrows
            self.keys_pressed.add(str(key))
        
    def on_release(self, key):
        try:
            # For regular keys
            self.keys_pressed.discard(key.char)
        except AttributeError:
            # For special keys like arrows
            self.keys_pressed.discard(str(key))
            
    def __del__(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        
        