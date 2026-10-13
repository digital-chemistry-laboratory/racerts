"""Optional dependencies, imported when first needed."""

from importlib import import_module


def require(module: str, extra: str):
    """Import module, or raise ImportError naming the pip extra that installs it."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is not installed; install it with `pip install racerts[{extra}]`."
        ) from exc
