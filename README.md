# 🚀 Momentum Look-Ahead for Asynchronous DiLoCo – PyTorch Implementation

This project is a research-oriented PyTorch implementation of the paper:

> **"Momentum Look-Ahead for Asynchronous Distributed Low-Communication Training"**  
> *Published at ICLR 2025*  
> [arXiv Link](https://arxiv.org/abs/2401.09135)

We simulate asynchronous training with 4 local workers and a central parameter server using LeNet on the CIFAR-10 dataset.

---

## 📌 Features

- ✅ Asynchronous worker training (no synchronization)
- ✅ Central server applies **Momentum Look-Ahead** updates (per paper Eq. 7)
- ✅ Modular architecture with real-time threading
- ✅ Clean, research-style code (easy to extend to multi-machine)

---

## 🧠 Paper Idea in Short

Instead of waiting for all workers to sync, each one sends updates (∆θ) asynchronously.  
The server uses:

\\[
\\begin{aligned}
m_{t+1} &= \\gamma m_t + (1 - \\gamma)\\Delta\\theta_t \\\\
\\theta_{t+1} &= \\theta_t - \\eta (\\gamma m_{t+1} + \\Delta\\theta_t)
\\end{aligned}
\\]

This approach improves convergence in heterogeneous hardware setups.

---

## 📁 Project Structure

async_diloco_pytorch/
├── main.py # Launches async training and evaluation
├── models/
│ └── lenet.py # LeNet-5 CNN model for CIFAR-10
├── datasets/
│ └── cifar_loader.py # CIFAR-10 DataLoader
├── server/
│ └── async_server.py # Momentum Look-Ahead server logic
├── workers/
│ └── async_worker.py # Async local training worker (threaded)
└── README.md

---

## ⚙️ Installation

\`\`\`bash
conda create -n async-diloco-env python=3.10 -y
conda activate async-diloco-env
pip install torch torchvision matplotlib
\`\`\`

---

## 🚀 Usage

\`bash
python main.py
\`

> The default configuration runs for **30 seconds** with 4 workers.

---

## 🔍 Example Output

\`\`\`
[Server] Running...
[Worker 0] Starting.
[Worker 1] Starting.
[Worker 2] Starting.
[Worker 3] Starting.

[Main] Running training for 30 seconds...
[Server] Shutting down.
[Eval] Accuracy: 31.59%
\`\`\`

---

## 🛠 Configuration (in `main.py`)

| Param         | Meaning                            | Example Value |
|---------------|------------------------------------|---------------|
| `NUM_WORKERS` | Number of async threads            | 4             |
| `INNER_STEPS` | Local SGD steps before sending     | 5             |
| `SERVER_LR`   | Learning rate at server            | 0.7           |
| `MOMENTUM`    | γ for momentum buffer              | 0.9           |
| `TRAIN_SECONDS`| How long to train asynchronously | 30            |

---

## 📈 TODOs and Extensions

- [ ] Add dynamic local update (DyLU)
- [ ] Compare with Async-SGD and Sync-DiLoCo
- [ ] Add TensorBoard logging
- [ ] Extend to multiprocessing or RPC-based distributed training

---

## 📜 License

MIT License – open for research and educational use.

---

## 🙏 Acknowledgments

Based on the paper by Ajanthan et al., published at ICLR 2025. Inspired by the official [DeepMind async DiLoCo repo](https://github.com/google-deepmind/asyncdiloco).
"""
