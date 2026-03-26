import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus import MultimodalEmbedder

MEDIA_ROOT = REPO_ROOT / "examples" / "multimodal_retrieval" / "data" / "media"


def main():
    model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "Qwen/Qwen2-VL-2B-Instruct")
    processor_name_or_path = os.getenv("PROCESSOR_NAME_OR_PATH", model_name_or_path)
    model_type = os.getenv("MODEL_TYPE", "auto")
    model = MultimodalEmbedder(
        model_name_or_path=model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        model_type=model_type,
        trust_remote_code=True,
        normalize_embeddings=True,
        query_max_length=512,
        passage_max_length=1024,
        batch_size=2,
    )

    queries = [
        {
            "text": "Find the matching color grid",
            "images": [str(MEDIA_ROOT / "query.ppm")],
        },
        {
            "text": "Retrieve the reference grid with a red corner",
        },
    ]
    passages = [
        {
            "text": "The reference grid with a red corner and green center",
            "images": [str(MEDIA_ROOT / "doc.ppm")],
        },
        {
            "text": "An alternate grid dominated by blue pixels",
            "images": [str(MEDIA_ROOT / "neg.ppm")],
        },
    ]

    query_embeddings = model.encode_queries(queries, convert_to_numpy=False)
    passage_embeddings = model.encode_corpus(passages, convert_to_numpy=False)
    scores = query_embeddings @ passage_embeddings.T
    print(scores)


if __name__ == "__main__":
    main()
