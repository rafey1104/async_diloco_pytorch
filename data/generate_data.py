import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),                         # Convert to [0,1] tensor
        transforms.Normalize((0.5, 0.5, 0.5),           # Mean normalization
                             (0.5, 0.5, 0.5))           # Std normalization
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

# from datasets.cifar_loader import get_cifar10_loaders

trainloader, testloader = get_cifar10_loaders(batch_size=64)

images, labels = next(iter(trainloader))
print(images.shape)  # Expected: [64, 3, 32, 32]