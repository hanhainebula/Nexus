import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus import MultimodalEmbedder

MEDIA_ROOT = REPO_ROOT / "examples" / "multimodal_retrieval" / "data" / "media"


def main():
    model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    processor_name_or_path = os.getenv("PROCESSOR_NAME_OR_PATH", model_name_or_path)
    model_type = os.getenv("MODEL_TYPE", "auto")
    model = MultimodalEmbedder(
        model_name_or_path=model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        model_type=model_type,
        pooling_method="last_token",
        normalize_embeddings=True,
        use_chat_template=True,
        use_fp16=False,
    )

    queries = [
        {
            "text": "Find the matching color grid.",
            "image_path": str(MEDIA_ROOT / "query.ppm"),
        }
    ]
    corpus = [
        {"text": "The reference grid with a red corner and green center.", "image_path": str(MEDIA_ROOT / "doc.ppm")},
        {"text": "The negative grid dominated by blue pixels.", "image_path": str(MEDIA_ROOT / "neg.ppm")},
    ]

    q_emb = model.encode_queries(queries)
    d_emb = model.encode_corpus(corpus)
    print("query shape:", q_emb.shape)
    print("doc shape:", d_emb.shape)
    print("scores:", q_emb @ d_emb.T)


if __name__ == "__main__":
    main()
