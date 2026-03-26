import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus.evaluation.multimodal_retrieval.data_loader import MultimodalRetrievalEvalDataLoader


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def test_eval_loader_resolves_image_and_video_roots(tmp_path):
    dataset_dir = tmp_path / "eval" / "toyset"
    dataset_dir.mkdir(parents=True)
    image_root = tmp_path / "shared_images"
    video_root = tmp_path / "shared_videos"
    image_root.mkdir()
    video_root.mkdir()

    write_jsonl(
        dataset_dir / "corpus.jsonl",
        [
            {"_id": "d1", "text": "document", "image_path": "page.png"},
            {"_id": "d2", "text": "video doc", "video_path": "clip.mp4"},
        ],
    )
    write_jsonl(dataset_dir / "test_queries.jsonl", [{"_id": "q1", "text": "query", "video_path": "query.mp4"}])
    write_jsonl(dataset_dir / "test_qrels.jsonl", [{"query-id": "q1", "corpus-id": "d2", "score": 1}])

    loader = MultimodalRetrievalEvalDataLoader(
        eval_name="toy",
        dataset_dir=str(tmp_path / "eval"),
        image_root=str(image_root),
        video_root=str(video_root),
    )

    corpus = loader.load_corpus(dataset_name="toyset")
    queries = loader.load_queries(dataset_name="toyset")

    assert corpus["d1"]["images"][0]["path"] == str(image_root / "page.png")
    assert corpus["d2"]["videos"][0]["path"] == str(video_root / "clip.mp4")
    assert queries["q1"]["videos"][0]["path"] == str(video_root / "query.mp4")
