import torch
import torch.nn.functional as F
import torch.optim as optim

def pg_loss_vanila(logits, actions_t, returns_t):
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions_t)

    loss = -(log_probs * returns_t).mean()
    return loss


def ppo_loss_value_vanila(values, returns_t):
    return F.mse_loss(values, returns_t)

def ppo_loss_policy_vanila(logits, actions, old_log_probs, advantages, clip_ratio):
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)

    ratio = torch.exp(log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

    loss = -torch.min(unclipped, clipped).mean()
    return loss

def ppo_entropy_loss_vanila(logits):
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().mean()

def loss_behavioral_cloning(logits, actions_t):
    loss = F.cross_entropy(logits, actions_t)
    return loss

