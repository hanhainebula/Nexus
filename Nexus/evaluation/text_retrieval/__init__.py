from importlib import import_module

__all__ = [
    "TextRetrievalEvalArgs",
    "TextRetrievalEvalModelArgs",
    "TextRetrievalAbsEvaluator",
    "TextRetrievalEvalDataLoader",
    "TextRetrievalEvalRetriever",
    "TextRetrievalEvalDenseRetriever",
    "TextRetrievalEvalReranker",
    "TextRetrievalEvalRunner",
]


def __getattr__(name):
    if name in {"TextRetrievalEvalArgs", "TextRetrievalEvalModelArgs"}:
        module = import_module("Nexus.evaluation.text_retrieval.arguments")
        return getattr(module, name)
    if name == "TextRetrievalAbsEvaluator":
        module = import_module("Nexus.evaluation.text_retrieval.evaluator")
        return getattr(module, name)
    if name == "TextRetrievalEvalDataLoader":
        module = import_module("Nexus.evaluation.text_retrieval.data_loader")
        return getattr(module, name)
    if name in {"TextRetrievalEvalRetriever", "TextRetrievalEvalDenseRetriever", "TextRetrievalEvalReranker"}:
        module = import_module("Nexus.evaluation.text_retrieval.searcher")
        return getattr(module, name)
    if name == "TextRetrievalEvalRunner":
        module = import_module("Nexus.evaluation.text_retrieval.runner")
        return getattr(module, name)
    raise AttributeError(f"module 'Nexus.evaluation.text_retrieval' has no attribute {name!r}")
