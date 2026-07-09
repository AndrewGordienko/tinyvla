from __future__ import annotations

TRAINABLE_MODES = ("checkpoint", "expert", "brain", "brain_visual", "all")
RECOVERY_TRAINABLE_MODES = tuple(mode for mode in TRAINABLE_MODES if mode != "checkpoint")


def group_for_param(name: str) -> str:
    if ".vision_model." in name:
        return "vision_encoder"
    if ".connector." in name:
        return "vision_connector"
    if ".text_model." in name:
        return "vlm_text"
    if ".lm_expert." in name:
        return "action_expert"
    if ".state_proj." in name:
        return "state_proj"
    if ".action_" in name:
        return "action_projectors"
    return "other"


def should_train_param(name: str, mode: str) -> bool:
    if mode == "checkpoint":
        raise ValueError("checkpoint mode preserves existing requires_grad values")
    if mode == "all":
        return True

    expert = ".lm_expert." in name or ".action_" in name or ".state_proj" in name
    brain = expert or ".text_model." in name
    brain_visual = brain or ".vision_model." in name or ".connector." in name

    if mode == "expert":
        return expert
    if mode == "brain":
        return brain
    if mode == "brain_visual":
        return brain_visual
    raise ValueError(f"unknown trainable mode: {mode}")


def set_trainable(policy, mode: str) -> int:
    if mode == "checkpoint":
        return sum(param.numel() for param in policy.parameters() if param.requires_grad)
    for name, param in policy.named_parameters():
        param.requires_grad = should_train_param(name, mode)
    return sum(param.numel() for param in policy.parameters() if param.requires_grad)

