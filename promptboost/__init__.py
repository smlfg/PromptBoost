"""PromptBoost v0.1 local app."""

__all__ = ["__version__"]

__version__ = "0.1.1"


def boost(*args, **kwargs):
    from .core import boost as _boost

    return _boost(*args, **kwargs)


def classify_task(*args, **kwargs):
    from .pipeline import classify_task as _classify_task

    return _classify_task(*args, **kwargs)
