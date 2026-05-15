from importlib import import_module

__all__ = [
    "TextEmbedder",
    "BaseEmbedderInferenceEngine",
    "TextReranker",
    "BaseRerankerInferenceEngine",
    "MultiModalEmbedder",
    "MultimodalEmbedder",
]


def __getattr__(name):
    if name in {"TextEmbedder", "BaseEmbedderInferenceEngine"}:
        module = import_module("Nexus.inference.embedder.text_retrieval")
        return getattr(module, name)
    if name in {"TextReranker", "BaseRerankerInferenceEngine"}:
        module = import_module("Nexus.inference.reranker.text_retrieval")
        return getattr(module, name)
    if name in {"MultiModalEmbedder", "MultimodalEmbedder"}:
        module = import_module("Nexus.inference.embedder.multimodal_retrieval")
        return getattr(module, name, getattr(module, "MultimodalEmbedder"))
    raise AttributeError(f"module 'Nexus.inference' has no attribute {name!r}")
