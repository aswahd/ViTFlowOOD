import os
import numpy as np
import itertools
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from models import mixer_b16_224_in21k
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from nflib.flows import (
    ActNorm, AffineHalfFlow, NormalizingFlowModel
)
from utils import get_measures

from dataset import SVHN, LSUN
from ood_evaluation_methods import msp, ebo, odin, ours
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_data", type=str, default="cifar10")
parser.add_argument('--root', type=str, default='data')
parser.add_argument("--results_path", type=str, default="results.txt", help="path to save results")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model_checkpoint", nargs='+', type=str, required=True)
args = parser.parse_args()
f = open(args.results_path, 'w')
f.write(f"------ In-distribution data: {args.in_data} ------\n")

IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10 if args.in_data == "cifar10" else 100

# Classification Model
model_cls = mixer_b16_224_in21k()
model_cls.head = nn.Linear(model_cls.head.in_features, num_classes)
state_dict = torch.load(args.model_checkpoint[0])
model_cls.load_state_dict(state_dict)
model_cls.to(device)
model_cls.eval()

# Normalizing flow model
dim = 768
prior = TransformedDistribution(Uniform(torch.zeros(dim).cuda(), torch.ones(dim).cuda()), SigmoidTransform().inv)
flows = [AffineHalfFlow(dim=dim, parity=i % 2, nh=256) for i in range(2)] + \
        [AffineHalfFlow(dim=dim, parity=i % 2, nh=128) for i in range(3)]
norms = [ActNorm(dim=dim, data_dep_init_done=True) for _ in flows]
flows = list(itertools.chain(*zip(norms, flows)))
model_nf = NormalizingFlowModel(prior, flows).cuda()
ckpt = torch.load(args.model_checkpoint[1])
model_nf.load_state_dict(ckpt)
model_nf.eval()
model_nf.cuda()

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

to_replot_dict = {}


def eval_ood(in_scores, loader, method, recall_level=0.95, ood_ds_name="", make_plot=True):
    score_fn = {
        'msp': msp,
        'odin': odin,
        'ebo': ebo,
        'ours': ours
    }[method]
    if method == 'ours':
        out_scores = score_fn(loader, model_cls, model_nf)
    else:
        out_scores = score_fn(loader, model_cls)

    in_scores, out_scores = -in_scores, -out_scores  # the higher should be OOD
    auroc, aupr, fpr = get_measures(out_scores, in_scores, recall_level)
    print(f"{ood_ds_name + ':':<20} auroc: {auroc:.4f}, aupr: {aupr:.4f}, fpr@{int(recall_level * 100):d}: {fpr:.4f}")
    # f.write(
    #     f"{ood_ds_name + ':':<20} auroc: {auroc:.4f}, aupr: {aupr:.4f}, fpr@{int(recall_level * 100):d}: {fpr:.4f}\n")
    f.write(
        f"{ood_ds_name + ':':<20} auroc: {auroc:.4f}/{aupr:.4f}/{fpr:.4f}\n")
    in_scores, out_scores = -in_scores, -out_scores
    # Plot Histogram
    if make_plot:
        plt.figure(figsize=(5.5, 3), dpi=100)

        plt.title(f"MLP-Mixer-B16 on {args.in_data} vs. {ood_ds_name} \n"
                  f" AUROC= {str(float(auroc * 100))[:6]}%", fontsize=14)

    vals, bins = np.histogram(out_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0
    to_replot_dict[f"{ood_ds_name}"] = [bin_centers, vals]
    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="crimson", marker="", label="out test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="crimson", alpha=0.3)

    vals, bins = np.histogram(in_scores, bins=100)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0
    to_replot_dict[f"{args.in_data}"] = [bin_centers, vals]

    if make_plot:
        plt.plot(bin_centers, vals, linewidth=4, color="navy", marker="", label="in test")
        plt.fill_between(bin_centers, vals, [0] * len(vals), color="navy", alpha=0.3)

        plt.xlabel("Log Prob.", fontsize=14)
        plt.ylabel("Count", fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0, None])

        plt.legend(fontsize=14)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{method}_{args.in_data} vs. {ood_ds_name}_auroc.png")
        plt.close()


# In-distribution data
if args.in_data == "cifar10":
    in_data = torchvision.datasets.CIFAR10(args.root, train=False, download=True, transform=test_transforms)
    in_loader = torch.utils.data.DataLoader(in_data, batch_size=args.batch_size, shuffle=False)
    num_classes = 10
elif args.in_data == "cifar100":
    in_data = torchvision.datasets.CIFAR100(args.root, train=False, download=True, transform=test_transforms)
    in_loader = torch.utils.data.DataLoader(in_data, batch_size=args.batch_size, shuffle=False)
    num_classes = 100
else:
    raise Exception("Dataset not implemented")


for method in ['msp', 'ebo', 'odin', 'ours']:
    print(f"Method: {method}")
    f.write(f"Method: {method}\n")
    score_fn = {
        'msp': msp,
        'odin': odin,
        'ebo': ebo,
        'ours': ours
    }[method]
    if method == 'ours':
        in_score = score_fn(in_loader, model_cls, model_nf)
    else:
        in_score = score_fn(in_loader, model_cls)

    # CIFAR10/100
    ood_name = "CIFAF100" if args.in_data == "cifar10" else "CIFAR10"
    print(f"Evaluating on {ood_name}")
    ood_data = torchvision.datasets.CIFAR100 if args.in_data == "cifar10" else torchvision.datasets.CIFAR10
    ood_data = ood_data(args.root, train=False, download=True, transform=test_transforms)
    ood_loader = DataLoader(ood_data, batch_size=args.batch_size)
    eval_ood(in_score, ood_loader, method=method, ood_ds_name=ood_name)

    # SVHN
    print("Evaluating on SVHN...")
    ood_data = torchvision.datasets.SVHN(args.root, split='test', download=True, transform=test_transforms)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size)
    eval_ood(in_score, ood_loader, method=method, ood_ds_name="SVHN")

    # LSUN
    print("Evaluating on LSUN...")
    ood_data = LSUN(args.root+'/LSUN', transform=test_transforms)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size)
    eval_ood(in_score, ood_loader, method=method, ood_ds_name="LSUN")

    # Places365
    print("Evaluating on Places365...")
    ood_data = torchvision.datasets.ImageFolder(args.root + '/places365', transform=test_transforms)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size)
    eval_ood(in_score, ood_loader, method=method, ood_ds_name="Places365")

    # Texture
    print("Evaluating on Texture...")
    ood_data = torchvision.datasets.ImageFolder(args.root + '/texture', transform=test_transforms)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size)
    eval_ood(in_score, ood_loader, method=method, ood_ds_name="Texture")


f.close()
# torch.save(to_replot_dict, f"plots/{method}_{args.in_data}_to_replot_dict.pt")