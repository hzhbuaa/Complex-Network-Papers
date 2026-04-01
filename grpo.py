"""
A minimal, educational GRPO (Group Relative Policy Optimization) implementation.

This script shows the core GRPO update loop with:
- Grouped rollouts per prompt
- Relative advantages normalized inside each group
- PPO-style clipped objective + KL penalty against a reference policy

It is intentionally compact and self-contained for learning purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    vocab_size: int = 128
    hidden_size: int = 128
    seq_len: int = 32
    prompt_len: int = 8
    batch_size: int = 16
    group_size: int = 4
    lr: float = 1e-3
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    train_steps: int = 400
    device: str = "cpu"
    seed: int = 42


class TinyPolicy(nn.Module):
    """A tiny autoregressive token policy for demonstration."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T]
        x = self.emb(tokens)
        h, _ = self.rnn(x)
        return self.head(h)  # [B, T, V]


@torch.no_grad()
def sample_grouped_sequences(
    policy: TinyPolicy,
    prompts: torch.Tensor,
    seq_len: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample G completions per prompt.

    Returns:
        sequences: [B*G, T]
        old_logprobs: [B*G, T-1] token log-probs under current policy
    """
    device = prompts.device
    bsz, prompt_len = prompts.shape

    # repeat each prompt group_size times
    seq = prompts.unsqueeze(1).repeat(1, group_size, 1).view(bsz * group_size, prompt_len)

    # autoregressive generation
    while seq.shape[1] < seq_len:
        logits = policy(seq)[:, -1, :]  # [B*G, V]
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_tok], dim=1)

    logits_full = policy(seq[:, :-1])
    logp_full = torch.log_softmax(logits_full, dim=-1)
    tok = seq[:, 1:].unsqueeze(-1)
    old_logprobs = torch.gather(logp_full, -1, tok).squeeze(-1)
    return seq.to(device), old_logprobs.to(device)


@torch.no_grad()
def toy_reward_fn(sequences: torch.Tensor) -> torch.Tensor:
    """
    A toy scalar reward per sequence for demo purposes.

    Heuristic:
    - reward if final token parity matches first token parity
    - small bonus for diversity (unique token count)
    """
    first = sequences[:, 0]
    last = sequences[:, -1]
    parity_reward = ((first % 2) == (last % 2)).float()

    # diversity bonus in [0, 1]
    uniq = []
    for row in sequences:
        uniq.append(torch.unique(row).numel())
    uniq = torch.tensor(uniq, device=sequences.device, dtype=torch.float32)
    diversity = uniq / sequences.shape[1]

    return parity_reward + 0.2 * diversity


def group_relative_advantages(rewards: torch.Tensor, batch_size: int, group_size: int) -> torch.Tensor:
    """Normalize rewards within each prompt group (GRPO core idea)."""
    r = rewards.view(batch_size, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True).clamp_min(1e-6)
    adv = (r - mean) / std
    return adv.view(-1)


def sequence_logprobs(policy: TinyPolicy, seq: torch.Tensor) -> torch.Tensor:
    """Token-wise log probs for seq[:,1:] conditioned on seq[:,:-1]."""
    logits = policy(seq[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    tok = seq[:, 1:].unsqueeze(-1)
    return torch.gather(logp, -1, tok).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    logp = torch.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1)


def grpo_train(cfg: GRPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    policy = TinyPolicy(cfg.vocab_size, cfg.hidden_size).to(device)
    ref_policy = TinyPolicy(cfg.vocab_size, cfg.hidden_size).to(device)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    for step in range(1, cfg.train_steps + 1):
        prompts = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.prompt_len), device=device)

        seq, old_logp = sample_grouped_sequences(policy, prompts, cfg.seq_len, cfg.group_size)
        rewards = toy_reward_fn(seq)
        advantages = group_relative_advantages(rewards, cfg.batch_size, cfg.group_size)

        new_logp = sequence_logprobs(policy, seq)
        with torch.no_grad():
            ref_logp = sequence_logprobs(ref_policy, seq)

        # sequence-level ratio via sum of token log-probs
        old_sum = old_logp.sum(dim=1)
        new_sum = new_logp.sum(dim=1)
        ratio = torch.exp(new_sum - old_sum)

        adv = advantages.detach()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        # approximate KL (token mean)
        kl = torch.mean(new_logp - ref_logp)

        logits = policy(seq[:, :-1])
        entropy = entropy_from_logits(logits).mean()

        loss = policy_loss + cfg.kl_beta * kl - cfg.entropy_coef * entropy

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optim.step()

        if step % 40 == 0 or step == 1:
            print(
                f"step={step:04d} "
                f"loss={loss.item():+.4f} "
                f"reward={rewards.mean().item():+.4f} "
                f"adv_std={advantages.std().item():+.4f} "
                f"kl={kl.item():+.4f}"
            )


if __name__ == "__main__":
    config = GRPOConfig()
    grpo_train(config)
