import numpy as np
import os
import itertools
from utils import get_auroc
import argparse
import torch
import sklearn.metrics as sk
import torch.optim as optim
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform, MultivariateNormal

from torch.utils.data import TensorDataset, DataLoader

from nflib.flows import (
    ActNorm, AffineHalfFlow, NormalizingFlowModel
)

parser = argparse.ArgumentParser("Train normalizing flow model.")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--model_checkpoint", type=str, default='checkpoints')
args = parser.parse_args()
os.makedirs(args.model_checkpoint, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data
assert args.dataset in ['cifar10', 'cifar100'], "Not implemented! Choice (cifar10, cifar100)"
train_data = torch.load(f'features/{args.dataset}_model/{args.dataset}_train_mixer_b16_features.pth')
train_dataset = TensorDataset(train_data, torch.zeros(len(train_data)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
dim = train_data.shape[1]

test_data = torch.load(f'features/{args.dataset}_model/{args.dataset}_test_mixer_b16_features.pth')
test_data = TensorDataset(test_data, torch.zeros(len(test_data)))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# OOD Data for evaluation
ood_data = 'cifar100' if args.dataset == "cifar10" else "cifar10"
ood_data = torch.load(f'features/{args.dataset}_model/{ood_data}_test_mixer_b16_features.pth')
ood_dataset = TensorDataset(ood_data, torch.zeros(len(ood_data)))
ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False)

# Build model
prior = TransformedDistribution(Uniform(torch.zeros(dim).cuda(), torch.ones(dim).cuda()), SigmoidTransform().inv)
# prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))
flows = [AffineHalfFlow(dim=dim, parity=i % 2, nh=128) for i in range(5)]
norms = [ActNorm(dim=dim) for _ in flows]
flows = list(itertools.chain(*zip(norms, flows)))
model = NormalizingFlowModel(prior, flows).to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
print("number of params: ", sum(p.numel() for p in model.parameters()))


def get_lop_prob_scores(loader):
    """ The negative log-likelihood is the OOD score."""
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.cuda()
        with torch.no_grad():
            _, prior_logprob, log_det = model(x)
            logprob = prior_logprob + log_det
            scores.append(logprob.cpu().numpy())
    return np.concatenate(scores)


def get_measures(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((neg, pos)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(neg)] += 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    return auroc, aupr


def eval_ood(id_loader, out_loader, i):
    id_score = get_lop_prob_scores(id_loader)
    ood_score = get_lop_prob_scores(out_loader)
    auroc, aupr = get_measures(ood_score, id_score)

    # Log-prob scores
    y_true = np.concatenate([np.zeros(len(ood_score)), np.ones(len(id_score))])
    scores = np.concatenate([ood_score, id_score])
    get_auroc(y_true, scores, make_plot=True, add_to_title=f"MLP-Mixer-B16: {args.dataset} vs. CIFAR100\n",
              name=f'{args.model_checkpoint}/Epoch_{i}_{args.dataset}_nf_b16.png')
    print(f"Epoch: {i} auroc: {auroc} aupr: {aupr}")


for epoch in range(args.epochs):
    model.train()
    for i, (x, _) in enumerate(train_loader):
        x = x.cuda()

        zs, prior_logprob, log_det = model(x)
        logprob = prior_logprob + log_det
        loss = -torch.mean(logprob)
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()

        if i % 100 == 0:
            print(f"iter: {i} loss: {loss.item():4f}")

    eval_ood(test_loader, ood_loader, epoch)

print("Saving model ...")
torch.save(model.state_dict(), f'{args.model_checkpoint}/{args.dataset}_nf_b16.pth')

