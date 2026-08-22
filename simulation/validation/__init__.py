"""Validation workflows for estimator performance studies."""

from typing import Any

__all__ = ["AEKFValidationConfig", "run_aekf_validation"]


def __getattr__(name: str) -> Any:
    """Load validation entry points lazily for ``python -m`` compatibility."""

    if name in __all__:
        from simulation.validation import aekf

        return getattr(aekf, name)

    raise AttributeError(name)
