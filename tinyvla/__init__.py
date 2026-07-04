"""Tiny SO-101/SmolVLA utilities."""

__all__ = ["CompactTokenEmbedding", "SO101Env", "SO101ReachTask", "load_pruned_smolvla"]


def __getattr__(name: str):
    if name == "SO101Env":
        from .env import SO101Env

        return SO101Env
    if name == "SO101ReachTask":
        from .task import SO101ReachTask

        return SO101ReachTask
    if name in {"CompactTokenEmbedding", "load_pruned_smolvla"}:
        from .smolvla_pruned import CompactTokenEmbedding, load_pruned_smolvla

        return {
            "CompactTokenEmbedding": CompactTokenEmbedding,
            "load_pruned_smolvla": load_pruned_smolvla,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
