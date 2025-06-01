# import torch
# import threading
# import time
# import multiprocessing

# def main():
#     from models.leNet import LeNet
#     from server.async_server import AsyncMomentumServer
#     from workers.async_workers import AsyncWorker
#     from data.generate_data import get_cifar10_loaders
#     from torch.utils.data import DataLoader, TensorDataset
#     import torch.nn.functional as F
#     from torchvision.datasets import CIFAR10
#     import torchvision.transforms as transforms

#     # ========== CONFIG ==========
#     NUM_WORKERS = 4
#     INNER_STEPS = 5
#     WORKER_LR = 0.01
#     SERVER_LR = 0.7
#     MOMENTUM = 0.9
#     TRAIN_SECONDS = 30
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # ============================

#     # 1. Load full dataset
#     trainloader, _ = get_cifar10_loaders(batch_size=128, num_workers=0)  # Use num_workers=0 for safety on Windows

#     # 2. Split into shards
#     data_shards = [[] for _ in range(NUM_WORKERS)]
#     for i, batch in enumerate(trainloader):
#         data_shards[i % NUM_WORKERS].append(batch)

#     shard_loaders = []
#     for shard in data_shards:
#         inputs = torch.cat([x[0] for x in shard])
#         labels = torch.cat([x[1] for x in shard])
#         dataset = TensorDataset(inputs, labels)
#         shard_loaders.append(DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0))

#     # 3. Model and server
#     global_model = LeNet().to(DEVICE)
#     server = AsyncMomentumServer(global_model, lr=SERVER_LR, momentum=MOMENTUM)
#     server_thread = threading.Thread(target=server.run)
#     server_thread.start()

#     # 4. Launch workers
#     workers = []
#     for i in range(NUM_WORKERS):
#         worker = AsyncWorker(
#             worker_id=i,
#             model_fn=LeNet,
#             dataloader=shard_loaders[i],
#             server=server,
#             device=DEVICE,
#             inner_steps=INNER_STEPS,
#             lr=WORKER_LR,
#             delay=(0.1, 0.4)
#         )
#         worker.start()
#         workers.append(worker)

#     print(f"\n[Main] Running training for {TRAIN_SECONDS} seconds...\n")
#     time.sleep(TRAIN_SECONDS)

#     # 5. Shutdown
#     for w in workers:
#         w.stop()
#     for w in workers:
#         w.join()

#     server.update_queue.put("STOP")
#     server_thread.join()

#     # 6. Evaluation
#     def evaluate(model, device):
#         model.eval()
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
#         testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
#         testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for images, labels in testloader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 preds = torch.argmax(outputs, dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)

#         print(f"[Eval] Accuracy: {100.0 * correct / total:.2f}%")

#     evaluate(global_model, DEVICE)

# if __name__ == "__main__":
#     multiprocessing.freeze_support()  # Windows fix for spawn
#     main()

# Enhance `main.py` output for better visual clarity and showcase purpose


import torch
import threading
import time
import multiprocessing

def main():
    from models.leNet import LeNet
    from server.async_server import AsyncMomentumServer
    from workers.async_workers import AsyncWorker
    from data.generate_data import get_cifar10_loaders
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    print("\\n🚀 Starting: Momentum Look-Ahead for Asynchronous DiLoCo (LeNet + CIFAR-10)")
    print("🔧 Backend: PyTorch + Threads + Custom Async Training Loop")
    print("=============================================")

    NUM_WORKERS = 4
    INNER_STEPS = 5
    WORKER_LR = 0.01
    SERVER_LR = 0.7
    MOMENTUM = 0.9
    TRAIN_SECONDS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📦 Usig device: {DEVICE}")
    print(f"🧠 Numer of Workers: {NUM_WORKERS}")
    print(f"🔁 Inner teps per Worker: {INNER_STEPS}")
    print("=============================================")

    trainloader, _ = get_cifar10_loaders(batch_size=128, num_workers=0)
    data_shards = [[] for _ in range(NUM_WORKERS)]
    for i, batch in enumerate(trainloader):
        data_shards[i % NUM_WORKERS].append(batch)

    shard_loaders = []
    for shard in data_shards:
        inputs = torch.cat([x[0] for x in shard])
        labels = torch.cat([x[1] for x in shard])
        dataset = TensorDataset(inputs, labels)
        shard_loaders.append(DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0))

    global_model = LeNet().to(DEVICE)
    server = AsyncMomentumServer(global_model, lr=SERVER_LR, momentum=MOMENTUM)
    server_thread = threading.Thread(target=server.run)
    server_thread.start()

    workers = []
    for i in range(NUM_WORKERS):
        worker = AsyncWorker(
            worker_id=i,
            model_fn=LeNet,
            dataloader=shard_loaders[i],
            server=server,
            device=DEVICE,
            inner_steps=INNER_STEPS,
            lr=WORKER_LR,
            delay=(0.1, 0.4)
        )
        worker.start()
        workers.append(worker)

    print(f"🕒 Training asynchronously for {TRAIN_SECONDS} seconds...\n")
    time.sleep(TRAIN_SECONDS)

    for w in workers:
        w.stop()
    for w in workers:
        w.join()

    server.update_queue.put("STOP")
    server_thread.join()

    print("\\n✅ Training Complete!")
    print("📊 Evaluating Final Global Model on CIFAR-10 Test Set...\n")

    def evaluate(model, device):
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"🎯 Final Test Accuracy: {100.0 * correct / total:.2f}%")

    evaluate(global_model, DEVICE)
    print("🏁 All done. Results powered by Asynchronous Momentum Look-Ahead 🚀")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()