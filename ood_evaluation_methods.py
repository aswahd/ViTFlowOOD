from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm


def msp(loader, net):
    """
        Args:
            net: classification model
            loader: data loader for the dataset
        Returns:
            scores: numpy array of scores (higher for ID data)
    """

    net.eval()
    scores = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            probs = net(x.cuda()).softmax(-1)
            probs, _ = probs.max(-1)
            scores.append(probs.cpu())

    return torch.cat(scores).numpy()


def ebo(loader, net, temperature=1.0):
    """
        Args:
            net: classification model
            loader: data loader for the dataset
            temperature: temperature for the softmax
        Returns:
            scores: numpy array of scores (higher for ID data)
    """

    net.eval()
    scores = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            logits = net(x.cuda())
            score = temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.append(score.cpu())

    return torch.cat(scores).numpy()


def odin(loader, net, temperature=1.0, epsilon=0.001):
    """
        Args:
            net: classification model
            loader: data loader for the dataset
            temperature: temperature for the softmax
            epsilon: noise to be added to the input
        Returns:
            scores: numpy array of scores (higher for ID data)
    """

    net.eval()
    scores = []
    criterion = nn.CrossEntropyLoss()

    for x, _ in tqdm(loader):
        x = x.cuda()
        x.requires_grad = True
        logits = net(x)

        # Temperature scaling
        logits = logits / temperature

        # Add input perturbation
        # i.e., sign of the gradient scaled by epsilon
        labels = logits.argmax(axis=1)
        loss = criterion(logits, labels)
        loss.backward()

        # Adding small perturbations to images
        gradient = torch.sign(x.grad.data)
        x_temp = torch.add(x.data, gradient, alpha=-epsilon)
        # Prediction after perturbation and temperature scaling
        with torch.no_grad():
            logits = net(x_temp)
        logits = logits / temperature
        # Calculating the confidence after adding perturbations
        probs = torch.softmax(logits, dim=1)
        probs, _ = probs.max(-1)
        scores.append(probs.cpu())

    return torch.cat(scores).numpy()


def ours(loader, model_cls, model_nf):

    """ The negative log-likelihood is the OOD score.
        model_cls: classification model
        model_nf: normalizing flow model
    """
    model_nf.eval()
    model_cls.eval()

    # Add a hook to the penultimate layer of the classification model
    def hook(mod, inp, out, _features):
        _features.append(inp[0])

    features = []
    hook = partial(hook, _features=features)
    h = model_cls.head.register_forward_hook(hook)

    scores = []
    for x, _ in tqdm(loader):
        x = x.cuda()
        with torch.no_grad():
            model_cls(x)
            x = features[0]
            features.clear()
            zs, prior_logprob, log_det = model_nf(x)
            logprob = prior_logprob + log_det
            scores.append(logprob.cpu())
    h.remove()
    return torch.cat(scores)

