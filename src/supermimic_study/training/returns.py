"""Equations 2 and 3, kept separately testable for time/survival indexing."""
import torch


def imagined_objective(rewards, terminations, bootstrap, gamma=.99):
    """[H,B] signals, [B] bootstrap. Reward t+1 is weighted by c_t."""
    survival = torch.ones_like(bootstrap)
    total = torch.zeros_like(bootstrap)
    for t in range(len(rewards)):
        total = total + gamma**t*survival*rewards[t]
        survival = survival*(1-terminations[t])
    return (total+gamma**len(rewards)*survival*bootstrap).mean()


@torch.no_grad()
def gae_targets(rewards, terminations, values, gamma=.99, lam=.95):
    """Fixed targets R_t; values contains V_0 through V_H, shape [H+1,B]."""
    if values.shape[0] != rewards.shape[0]+1 or rewards.shape != terminations.shape:
        raise ValueError("inconsistent imagined trajectory shapes")
    future = values[-1]
    result = []
    for t in reversed(range(len(rewards))):
        future = rewards[t]+gamma*(1-terminations[t])*((1-lam)*values[t+1]+lam*future)
        result.append(future)
    return torch.stack(list(reversed(result))).detach()
