from pynput import keyboard


class KeyboardController:
    def __init__(self) -> None:
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def update(self):
        self.x = 0.
        self.y = 0.
        self.yaw_rate = 0.
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
        
        