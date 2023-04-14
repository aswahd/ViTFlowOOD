import os
from functools import partial
import argparse
from tqdm import tqdm
from models import mixer_b16_224_in21k
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from dataset import LSUN

IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

parser = argparse.ArgumentParser("Generate features from the penultimate layer of the network.")
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--root', type=str, default='data')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_checkpoint', type=str)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to extracted features
os.makedirs('features/cifar10_model', exist_ok=True)
os.makedirs('features/cifar100_model', exist_ok=True)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

num_classes = 10 if args.dataset == "cifar10" else 100
# Model
model = mixer_b16_224_in21k()
model.head = nn.Linear(model.head.in_features, num_classes)
state_dict = torch.load(args.model_checkpoint)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def test(dl):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dl):
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / total * 100


# Evaluate test accuracy
if args.dataset == "cifar10":
    ds = torchvision.datasets.CIFAR10
elif args.dataset == "cifar100":
    ds = torchvision.datasets.CIFAR100
else:
    raise NotImplementedError("Invalid dataset")

data = ds(args.root, train=False, download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
acc = test(test_loader)
print(f"{args.dataset} test accuracy: {acc:.2f}%")


def get_features(net, loader):
    """ Extract features from the penultimate layer of the network. """

    def hook(mod, inp, out, _features):
        _features.append(inp[0])

    features = []
    hook = partial(hook, _features=features)
    h = net.head.register_forward_hook(hook)  # Attach a hook to the penultimate layer
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            net(x.to(device))
    h.remove()
    return torch.cat(features).cpu()


# CIFAR10
for phase in ['train', 'test']:
    print(f"Saving CIFAR10_{phase} features")
    data = torchvision.datasets.CIFAR10(args.root,
                                        train=True if phase == "train" else False,
                                        download=True, transform=test_transforms)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
    feats = get_features(model, data_loader)
    path = f'features/{args.dataset}_model/cifar10_{phase}_mixer_b16_features.pth'
    torch.save(feats, path)

# CIFAR100
for phase in ['train', 'test']:
    print(f"Saving CIFAR100_{phase} features")
    data = torchvision.datasets.CIFAR100(args.root,
                                         train=True if phase == "train" else False,
                                         download=True, transform=test_transforms)
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
    feats = get_features(model, data_loader)
    path = f'features/{args.dataset}_model/cifar100_{phase}_mixer_b16_features.pth'
    torch.save(feats, path)

# SVHN
print("Saving SVHN features")
data = torchvision.datasets.SVHN(args.root, split='test', download=True, transform=test_transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
feats = get_features(model, data_loader)
path = f'features/{args.dataset}_model/svhn_mixer_b16_features.pth'
torch.save(feats, path)

# LSUN
print("Saving LSUN features")
data = LSUN(args.root+'/LSUN', transform=test_transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
feats = get_features(model, data_loader)
path = f'features/{args.dataset}_model/lsun_mixer_b16_features.pth'
torch.save(feats, path)

# Places365
print("Saving Places365 features")
data = torchvision.datasets.ImageFolder(args.root+'/places365', transform=test_transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
feats = get_features(model, data_loader)
path = f'features/{args.dataset}_model/places365_mixer_b16_features.pth'
torch.save(feats, path)

# Texture
print("Saving Texture features")
data = torchvision.datasets.ImageFolder(args.root+'/texture', transform=test_transforms)
data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size)
feats = get_features(model, data_loader)
path = f'features/{args.dataset}_model/texture_mixer_b16_features.pth'
torch.save(feats, path)
