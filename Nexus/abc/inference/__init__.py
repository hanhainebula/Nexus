from importlib import import_module

__all__ = [
    "AbsEmbedder",
    "AbsReranker",
    "AbsInferenceArguments",
    "InferenceEngine",
]


def __getattr__(name):
    if name == "AbsEmbedder":
        module = import_module("Nexus.abc.inference.embedder")
        return getattr(module, name)
    if name == "AbsReranker":
        module = import_module("Nexus.abc.inference.reranker")
        return getattr(module, name)
    if name == "AbsInferenceArguments":
        module = import_module("Nexus.abc.inference.arguments")
        return getattr(module, name)
    if name == "InferenceEngine":
        module = import_module("Nexus.abc.inference.inference_engine")
        return getattr(module, name)
    raise AttributeError(f"module 'Nexus.abc.inference' has no attribute {name!r}")
