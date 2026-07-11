"""Deterministic experiment utilities which do not leak evaluation RNG state."""
from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and supported accelerators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Restore all process-global RNGs after inline evaluation or sampling."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = None
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        mps_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
