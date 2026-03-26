import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from Nexus.inference.embedder.multimodal_retrieval import MultimodalEmbedder


class MultimodalRetrievalEvalRetriever(ABC):
    def __init__(self, embedder: MultimodalEmbedder, search_top_k: int = 1000, overwrite: bool = False):
        self.embedder = embedder
        self.search_top_k = search_top_k
        self.overwrite = overwrite

    def __str__(self) -> str:
        return os.path.basename(self.embedder.model_name_or_path)

    def stop_multi_process_pool(self):
        self.embedder.stop_self_pool()

    @abstractmethod
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, Dict[str, Any]],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ):
        pass


class MultimodalRetrievalEvalDenseRetriever(MultimodalRetrievalEvalRetriever):
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, Dict[str, Any]],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ):
        from Nexus.evaluation.text_retrieval.utils import index, search

        corpus_ids = list(corpus.keys())
        query_ids = list(queries.keys())

        if corpus_embd_save_dir is not None:
            os.makedirs(corpus_embd_save_dir, exist_ok=True)
            corpus_embedding_path = os.path.join(corpus_embd_save_dir, "corpus_embeddings.npy")
        else:
            corpus_embedding_path = None

        if corpus_embedding_path is not None and os.path.exists(corpus_embedding_path) and not self.overwrite:
            corpus_embeddings = np.load(corpus_embedding_path)
        else:
            corpus_embeddings = self.embedder.encode_corpus([corpus[cid] for cid in corpus_ids])
            if corpus_embedding_path is not None:
                np.save(corpus_embedding_path, corpus_embeddings)

        query_embeddings = self.embedder.encode_queries([queries[qid] for qid in query_ids])
        faiss_index = index(corpus_embeddings=corpus_embeddings)
        all_scores, all_indices = search(faiss_index=faiss_index, k=self.search_top_k, query_embeddings=query_embeddings)

        results = {qid: {} for qid in query_ids}
        for idx, qid in enumerate(query_ids):
            for score, indice in zip(all_scores[idx], all_indices[idx]):
                if indice == -1:
                    continue
                docid = corpus_ids[indice]
                if ignore_identical_ids and qid == docid:
                    continue
                results[qid][docid] = float(score)
        return results
