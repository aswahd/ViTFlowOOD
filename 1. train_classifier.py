import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from models import mixer_b16_224_in21k

IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

parser = argparse.ArgumentParser("Train MLP Mixer on in-distribution data")
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--clip_grad_norm', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--momentum', type=float, default=0.)
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)


train_transforms = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop((224, 224)),
    transforms.Resize((224, 224)),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# Dataset
if args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(args.data_path, train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    test_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)
    num_classes = 10
elif args.dataset == "cifar100":
    train_dataset = torchvision.datasets.CIFAR100(args.data_path, train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    test_dataset = torchvision.datasets.CIFAR100(args.data_path, train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)
    num_classes = 100
else:
    raise Exception("Dataset not implemented")

# Model
model = mixer_b16_224_in21k(pretrained=False)
state_dict = torch.load('jx_mixer_b16_224_in21k-617b3de2.pth')
model.load_state_dict(state_dict)
model.head = nn.Linear(model.head.in_features, num_classes)
model.head.weight.data.normal_(mean=0.0, std=0.01)
model.head.bias.data.zero_()
model.to(device)


if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    raise Exception("Optimizer not implemented. Choice (sgd, adam).")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 1e-3)
criterion = nn.CrossEntropyLoss()  # label_smoothing=args.label_smoothing


def test(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = net(x.cuda())
            correct += (pred.argmax(-1) == y.cuda()).sum()
            total += len(x)

    return correct / total


best_acc = 0.
for epoch in range(args.epochs):
    model.train()
    for X, y in tqdm(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
    # scheduler.step()

    acc = test(model, test_loader)
    print(f"Epoch: {epoch} Test acc: {acc}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f'{args.checkpoint_dir}/{args.dataset}_mixer_b16.pth')

torch.save(model.state_dict(), f'{args.checkpoint_dir}/{args.dataset}_final_mixer_b16.pth')
