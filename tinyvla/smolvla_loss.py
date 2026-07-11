"""Repository-owned correction for LeRobot 0.4.4 SmolVLA action loss."""
from __future__ import annotations

from types import MethodType

import torch
from torch import Tensor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def assert_tail_padding(action_is_pad: Tensor) -> None:
    """Require ``action_is_pad`` to be contiguous episode-end (tail) padding.

    Padding invariance — padded action timesteps cannot influence the model's
    predictions at valid timesteps — is guaranteed by SmolVLA's action-expert
    attention being CAUSAL within the action block (token k attends only to action
    tokens 0..k plus the prefix) *combined with* padding always being a tail
    suffix: a valid (earlier) token then never attends to a padded (later) one.
    This holds for chunk padding produced when a chunk overruns the episode end.
    If a dataset ever emitted interleaved padding, that invariant would break, so
    fail loudly here rather than silently leak padded tokens into valid outputs.
    """
    if action_is_pad.ndim != 2:
        raise ValueError(f"action_is_pad must be [B, T]; got {tuple(action_is_pad.shape)}")
    pad = action_is_pad.to(torch.int8)
    # A tail mask is non-decreasing along time (0...0 1...1). Any 1->0 step is
    # interleaved padding.
    if pad.shape[1] > 1 and torch.any(pad[:, 1:] < pad[:, :-1]):
        raise ValueError(
            "action_is_pad must be contiguous tail (episode-end) padding; "
            "interleaved padding would break attention padding-invariance"
        )


def reduce_valid_action_loss(
    losses: Tensor,
    action_is_pad: Tensor | None,
    action_dim: int,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Reduce only real action dimensions and valid, unpadded scalar entries."""

    if reduction not in {"mean", "none"}:
        raise ValueError(f"unsupported reduction: {reduction}")
    if action_dim <= 0 or action_dim > losses.shape[-1]:
        raise ValueError(f"invalid action_dim={action_dim} for loss shape {tuple(losses.shape)}")

    real_losses = losses[..., :action_dim]
    if action_is_pad is None:
        valid = torch.ones(real_losses.shape[:2], dtype=torch.bool, device=real_losses.device)
    else:
        if tuple(action_is_pad.shape) != tuple(real_losses.shape[:2]):
            raise ValueError(
                f"action_is_pad shape {tuple(action_is_pad.shape)} does not match "
                f"loss time axes {tuple(real_losses.shape[:2])}"
            )
        valid = ~action_is_pad.to(device=real_losses.device, dtype=torch.bool)

    masked = real_losses * valid.unsqueeze(-1)
    per_sample_denominator = valid.sum(dim=1) * action_dim
    if torch.any(per_sample_denominator == 0):
        raise ValueError("batch contains a sample with no valid action scalars")
    per_sample = masked.sum(dim=(1, 2)) / per_sample_denominator
    return per_sample if reduction == "none" else masked.sum() / per_sample_denominator.sum()


def _corrected_forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
    if "actions_id_pad" in batch:
        raise KeyError("legacy actions_id_pad is invalid; dataset processors must emit action_is_pad")
    if self.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    action_is_pad = batch.get("action_is_pad")
    actions = self.prepare_action(batch)
    # Padded action tokens still enter the action-expert transformer. Two things
    # keep them from affecting valid-timestep predictions: (1) action-block
    # attention is causal, so a valid (earlier) token never attends to a padded
    # (later) one, and (2) padding is a contiguous tail (asserted here). Zeroing
    # the padded targets is belt-and-suspenders — it makes the padded inputs
    # canonical — but the real guarantee is (1)+(2). See assert_tail_padding.
    if action_is_pad is not None:
        assert_tail_padding(action_is_pad.to(device=actions.device))
        actions = actions.masked_fill(
            action_is_pad.to(device=actions.device, dtype=torch.bool).unsqueeze(-1), 0.0
        )
    raw_losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    action_dim = int(self.config.action_feature.shape[0])
    reduced = reduce_valid_action_loss(raw_losses, action_is_pad, action_dim, reduction=reduction)
    scalar = reduced.mean() if reduced.ndim else reduced
    valid_scalars = (
        raw_losses.shape[0] * raw_losses.shape[1] * action_dim
        if action_is_pad is None
        else int((~action_is_pad.bool()).sum().item() * action_dim)
    )
    return reduced, {
        "loss": float(scalar.detach().item()),
        "action_dim": action_dim,
        "valid_action_scalars": valid_scalars,
        "truth_harness_loss": True,
    }


def install_corrected_smolvla_loss(policy):
    """Install the corrected forward pass on one policy instance."""

    policy.forward = MethodType(_corrected_forward, policy)
    policy._tinyvla_corrected_loss = True
    return policy
