from importlib import import_module

__all__ = ["AbsInferenceArguments"]


def __getattr__(name):
    if name == "AbsInferenceArguments":
        module = import_module("Nexus.abc.inference.arguments")
        return getattr(module, name)
    raise AttributeError(f"module 'Nexus.abc' has no attribute {name!r}")
