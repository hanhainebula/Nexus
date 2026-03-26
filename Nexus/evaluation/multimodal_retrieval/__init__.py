from importlib import import_module

__all__ = [
    "MultimodalRetrievalEvalArgs",
    "MultimodalRetrievalEvalModelArgs",
    "MultimodalRetrievalEvalDataLoader",
    "MultimodalRetrievalAbsEvaluator",
    "MultimodalRetrievalEvalRetriever",
    "MultimodalRetrievalEvalDenseRetriever",
    "MultimodalRetrievalEvalRunner",
]


def __getattr__(name):
    if name in {"MultimodalRetrievalEvalArgs", "MultimodalRetrievalEvalModelArgs"}:
        module = import_module("Nexus.evaluation.multimodal_retrieval.arguments")
        return getattr(module, name)
    if name == "MultimodalRetrievalEvalDataLoader":
        module = import_module("Nexus.evaluation.multimodal_retrieval.data_loader")
        return getattr(module, name)
    if name == "MultimodalRetrievalAbsEvaluator":
        module = import_module("Nexus.evaluation.multimodal_retrieval.evaluator")
        return getattr(module, name)
    if name in {"MultimodalRetrievalEvalRetriever", "MultimodalRetrievalEvalDenseRetriever"}:
        module = import_module("Nexus.evaluation.multimodal_retrieval.searcher")
        return getattr(module, name)
    if name == "MultimodalRetrievalEvalRunner":
        module = import_module("Nexus.evaluation.multimodal_retrieval.runner")
        return getattr(module, name)
    raise AttributeError(f"module 'Nexus.evaluation.multimodal_retrieval' has no attribute {name!r}")
