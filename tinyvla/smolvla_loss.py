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
    # Padded action tokens participate in the action-expert transformer. Give
    # them a canonical value so arbitrary dataset padding cannot influence even
    # the predictions at valid timesteps through attention.
    if action_is_pad is not None:
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
