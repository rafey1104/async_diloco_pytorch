import torch
import copy
import threading
from queue import Queue

class AsyncMomentumServer:
    def __init__(self, model, lr=0.7, momentum=0.9):
        self.model = model
        self.lr = lr
        self.gamma = momentum

        self.update_queue = Queue()
        self.lock = threading.Lock()
        self.momentum_buffers = {
            k: torch.zeros_like(p) for k, p in model.state_dict().items()
        }

    def apply_update(self, delta_state_dict):
        with self.lock:
            current_state = self.model.state_dict()
            new_state = {}

            for name in current_state:
                delta = delta_state_dict[name]

                # Momentum update
                m_prev = self.momentum_buffers[name]
                m_new = self.gamma * m_prev + (1 - self.gamma) * delta
                self.momentum_buffers[name] = m_new

                # Final param update
                updated_param = current_state[name] - self.lr * (self.gamma * m_new + delta)
                new_state[name] = updated_param

            # Commit updated weights
            self.model.load_state_dict(new_state)

    def run(self):
        print("[Server] Running...")
        while True:
            update = self.update_queue.get()
            if update == "STOP":
                print("[Server] Shutting down.")
                break
            self.apply_update(update)
