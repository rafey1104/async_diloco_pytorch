import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random

class AsyncWorker(threading.Thread):
    def __init__(self, worker_id, model_fn, dataloader, server, device, inner_steps=5, lr=0.01, delay=(0.1, 0.3)):
        super().__init__()
        self.worker_id = worker_id
        self.dataloader = dataloader
        self.device = device
        self.server = server
        self.inner_steps = inner_steps
        self.lr = lr
        self.delay = delay
        self.model_fn = model_fn
        self.running = True

    def run(self):
        print(f"[Worker {self.worker_id}] Starting.")
        while self.running:
            # 1. Sync model from server
            with self.server.lock:
                global_state = copy.deepcopy(self.server.model.state_dict())

            local_model = self.model_fn().to(self.device)
            local_model.load_state_dict(global_state)

            # Save old weights for pseudogradient
            old_state = copy.deepcopy(local_model.state_dict())

            # 2. Local training
            optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            data_iter = iter(self.dataloader)
            for _ in range(self.inner_steps):
                try:
                    inputs, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    inputs, labels = next(data_iter)

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 3. Compute pseudogradient (delta = old - new)
            new_state = local_model.state_dict()
            delta = {k: old_state[k] - new_state[k] for k in old_state}

            # 4. Send update to server
            self.server.update_queue.put(delta)

            # 5. Simulate worker delay (heterogeneity)
            time.sleep(random.uniform(*self.delay))

    def stop(self):
        self.running = False
