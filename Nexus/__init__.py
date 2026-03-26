from importlib import import_module

__all__ = [
    "AbsInferenceArguments",
    "TextEmbedder",
    "BaseEmbedderInferenceEngine",
    "TextReranker",
    "BaseRerankerInferenceEngine",
    "MultiModalEmbedder",
    "MultimodalEmbedder",
]


def __getattr__(name):
    if name == "AbsInferenceArguments":
        module = import_module("Nexus.abc")
        return getattr(module, name)
    if name in {"TextEmbedder", "BaseEmbedderInferenceEngine", "TextReranker", "BaseRerankerInferenceEngine"}:
        module = import_module("Nexus.inference")
        return getattr(module, name)
    if name in {"MultiModalEmbedder", "MultimodalEmbedder"}:
        module = import_module("Nexus.inference.embedder.multimodal_retrieval")
        return getattr(module, name, getattr(module, "MultimodalEmbedder"))
    raise AttributeError(f"module 'Nexus' has no attribute {name!r}")
