from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Nexus
from Nexus.modules.multimodal import (
    build_media_base_dir,
    build_prefixed_multimodal_group,
    extract_multimodal_hidden_states,
    _get_registered_conditional_generation_model_class,
    infer_multimodal_output_mode,
    load_multimodal_backbone,
    load_multimodal_processor,
    MultimodalProcessorAdapter,
    normalize_multimodal_item,
)


def test_import_nexus_without_optional_inference_backends():
    assert hasattr(Nexus, "MultimodalEmbedder")


def test_normalize_multimodal_item_supports_separate_media_roots(tmp_path):
    image_root = tmp_path / "images"
    video_root = tmp_path / "videos"
    image_root.mkdir()
    video_root.mkdir()

    base_dir = build_media_base_dir(
        base_dir=str(tmp_path / "metadata"),
        image_root=str(image_root),
        video_root=str(video_root),
    )

    item = normalize_multimodal_item(
        {
            "text": "find the clip",
            "image_path": "page_1.png",
            "video_path": "sample.mp4",
        },
        base_dir=base_dir,
    )

    assert item["text"] == "find the clip"
    assert item["images"][0]["path"] == str(image_root / "page_1.png")
    assert item["videos"][0]["path"] == str(video_root / "sample.mp4")


def test_build_prefixed_multimodal_group_supports_video_fields(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()

    base_dir = build_media_base_dir(base_dir=str(tmp_path), video_root=str(video_root))
    row = {
        "query_text": "retrieve the matching video",
        "query_video_path": "clips/a.mp4",
        "pos_text": "positive target",
        "pos_video_frames": ["frames/0001.jpg", "frames/0002.jpg"],
    }

    query_group = build_prefixed_multimodal_group(row, "query", base_dir=base_dir)
    pos_group = build_prefixed_multimodal_group(row, "pos", base_dir=base_dir)

    assert query_group[0]["videos"][0]["path"] == str(video_root / "clips/a.mp4")
    assert pos_group[0]["videos"][0]["paths"] == [
        str(tmp_path / "frames/0001.jpg"),
        str(tmp_path / "frames/0002.jpg"),
    ]


def test_normalize_multimodal_item_avoids_duplicate_media_root_join(tmp_path):
    media_root = tmp_path / "data" / "media"
    media_root.mkdir(parents=True)
    image_path = media_root / "query.ppm"
    image_path.write_bytes(b"P6\n1 1\n255\n\xff\x00\x00")

    base_dir = build_media_base_dir(
        base_dir=str(tmp_path / "data"),
        media_root=str(media_root),
    )

    item = normalize_multimodal_item(
        {
            "text": "find the matching image",
            "image_path": "media/query.ppm",
        },
        base_dir=base_dir,
    )

    assert item["images"][0]["path"] == str(image_path)


def test_build_media_base_dir_prefers_existing_cwd_relative_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    base_dir = repo_root / "examples" / "multimodal_retrieval" / "data"
    media_root = base_dir / "media"
    media_root.mkdir(parents=True)

    monkeypatch.chdir(repo_root)
    resolved = build_media_base_dir(
        base_dir=str(base_dir),
        media_root="examples/multimodal_retrieval/data/media",
    )

    assert resolved["media"] == str(media_root)
    assert resolved["image"] == str(media_root)
    assert resolved["video"] == str(media_root)


def test_processor_adapter_batches_visual_tensors_without_processor_pad():
    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            for feature in features:
                ids = list(feature["input_ids"])
                mask = list(feature.get("attention_mask", [1] * len(ids)))
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append(mask + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

    adapter = MultimodalProcessorAdapter(FakeProcessor(), model_type="qwen2_vl", use_chat_template=False)
    adapter._encode_single = lambda item, max_length=None: {
        "input_ids": [[11, 12, 13]],
        "attention_mask": [[1, 1, 1]],
        "pixel_values": torch.ones((1, 4, 8)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    batch = adapter.encode_batch([{"text": "a"}, {"text": "b"}], max_length=8)

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["pixel_values"].shape) == (2, 4, 8)
    assert tuple(batch["image_grid_thw"].shape) == (2, 3)


def test_processor_adapter_preserves_mm_token_type_ids_without_processor_pad():
    class FakeTokenizer:
        padding_side = "right"

        def pad(self, features, padding=True, return_tensors="pt"):
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            for feature in features:
                ids = list(feature["input_ids"])
                mask = list(feature.get("attention_mask", [1] * len(ids)))
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append(mask + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

    adapter = MultimodalProcessorAdapter(FakeProcessor(), model_type="qwen3_5", use_chat_template=False)
    encoded_items = [
        {
            "input_ids": [[11, 12, 13]],
            "attention_mask": [[1, 1, 1]],
            "mm_token_type_ids": [[0, 1, 1]],
        },
        {
            "input_ids": [[21, 22]],
            "attention_mask": [[1, 1]],
            "mm_token_type_ids": [[0, 1]],
        },
    ]
    adapter._encode_single = lambda item, max_length=None: encoded_items.pop(0)

    batch = adapter.encode_batch([{"text": "a"}, {"text": "b"}], max_length=8)

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["mm_token_type_ids"].shape) == (2, 3)
    assert batch["mm_token_type_ids"].tolist() == [[0, 1, 1], [0, 1, 0]]


def test_processor_adapter_preserves_llava_next_image_sizes_without_processor_pad():
    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            for feature in features:
                ids = list(feature["input_ids"])
                mask = list(feature.get("attention_mask", [1] * len(ids)))
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append(mask + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

    adapter = MultimodalProcessorAdapter(FakeProcessor(), model_type="llava_next", use_chat_template=False)
    adapter._encode_single = lambda item, max_length=None: {
        "input_ids": [[11, 12]],
        "attention_mask": [[1, 1]],
        "pixel_values": np.ones((1, 3, 3, 16, 16), dtype=np.float32),
        "image_sizes": [(32, 32)],
    }

    batch = adapter.encode_batch([{"text": "a"}, {"text": "b"}], max_length=8)

    assert tuple(batch["input_ids"].shape) == (2, 2)
    assert tuple(batch["pixel_values"].shape) == (2, 3, 3, 16, 16)
    assert tuple(batch["image_sizes"].shape) == (2, 2)


def test_load_multimodal_backbone_supports_peft_adapter_dirs(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    recorded = {}

    class DummyConfig:
        model_type = "qwen2_vl"

    def fake_auto_config(path, **kwargs):
        recorded["config_path"] = path
        return DummyConfig()

    def fake_registered_loader(config, model_name_or_path, load_kwargs):
        recorded["base_model_path"] = model_name_or_path
        return "base-model"

    def fake_peft_from_pretrained(model, adapter_path, is_trainable=False):
        recorded["adapter_model"] = model
        recorded["adapter_path"] = adapter_path
        recorded["is_trainable"] = is_trainable
        return "wrapped-model"

    monkeypatch.setattr(
        "Nexus.modules.multimodal._maybe_load_peft_config",
        lambda *args, **kwargs: SimpleNamespace(base_model_name_or_path="/tmp/base-model"),
    )
    monkeypatch.setattr("Nexus.modules.multimodal.AutoConfig.from_pretrained", fake_auto_config)
    monkeypatch.setattr("Nexus.modules.multimodal._maybe_load_registered_conditional_generation_model", fake_registered_loader)

    import peft

    monkeypatch.setattr(peft.PeftModel, "from_pretrained", fake_peft_from_pretrained)

    model, config = load_multimodal_backbone(
        str(adapter_dir),
        model_type="qwen2_vl",
        peft_is_trainable=True,
    )

    assert model == "wrapped-model"
    assert config.model_type == "qwen2_vl"
    assert recorded["config_path"] == "/tmp/base-model"
    assert recorded["base_model_path"] == "/tmp/base-model"
    assert recorded["adapter_model"] == "base-model"
    assert recorded["adapter_path"] == str(adapter_dir)
    assert recorded["is_trainable"] is True


def test_load_multimodal_processor_falls_back_to_base_model_for_adapter(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    attempted_paths = []

    def fake_auto_processor(path, **kwargs):
        attempted_paths.append(path)
        if path == str(adapter_dir):
            raise ValueError("adapter dir does not contain a processor")
        return {"loaded_from": path}

    monkeypatch.setattr("Nexus.modules.multimodal.AutoProcessor.from_pretrained", fake_auto_processor)
    monkeypatch.setattr(
        "Nexus.modules.multimodal._maybe_load_peft_config",
        lambda *args, **kwargs: SimpleNamespace(base_model_name_or_path="/tmp/base-model"),
    )

    processor = load_multimodal_processor(str(adapter_dir))

    assert processor == {"loaded_from": "/tmp/base-model"}
    assert attempted_paths == [str(adapter_dir), "/tmp/base-model"]


def test_load_multimodal_processor_accepts_processor_kwargs(monkeypatch):
    recorded = {}

    def fake_auto_processor(path, **kwargs):
        recorded["path"] = path
        recorded["kwargs"] = kwargs
        return {"loaded_from": path}

    monkeypatch.setattr("Nexus.modules.multimodal.AutoProcessor.from_pretrained", fake_auto_processor)

    processor = load_multimodal_processor(
        "demo-model",
        processor_kwargs={"max_pixels": 1024, "size": {"shortest_edge": 448}},
    )

    assert processor == {"loaded_from": "demo-model"}
    assert recorded["kwargs"]["max_pixels"] == 1024
    assert recorded["kwargs"]["size"] == {"shortest_edge": 448}


def test_load_multimodal_processor_translates_llava_next_pixel_budgets(monkeypatch):
    recorded = {}

    def fake_auto_processor(path, **kwargs):
        recorded["path"] = path
        recorded["kwargs"] = kwargs
        return {"loaded_from": path}

    monkeypatch.setattr("Nexus.modules.multimodal.AutoProcessor.from_pretrained", fake_auto_processor)

    processor = load_multimodal_processor(
        "demo-model",
        model_type="llava_next",
        processor_kwargs={"max_pixels": 262144},
    )

    assert processor == {"loaded_from": "demo-model"}
    assert "max_pixels" not in recorded["kwargs"]
    assert recorded["kwargs"]["size"] == {"shortest_edge": 512}
    assert recorded["kwargs"]["crop_size"] == {"height": 512, "width": 512}


@pytest.mark.parametrize(
    ("model_type", "version_hint"),
    [
        ("qwen3_vl", "transformers>=4.57.3"),
        ("qwen3_5", "Qwen3.5 support"),
    ],
)
def test_missing_qwen_loader_reports_version_hint(model_type, version_hint):
    with pytest.raises(ImportError, match=version_hint):
        _get_registered_conditional_generation_model_class(
            model_type,
            transformers_module=SimpleNamespace(__version__="4.52.3"),
        )


def test_encode_single_device_casts_bfloat16_before_numpy():
    from Nexus.inference.embedder.multimodal_retrieval.generic import MultimodalEmbedder

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    embedder = MultimodalEmbedder.__new__(MultimodalEmbedder)
    embedder.model = FakeModel()
    embedder.use_fp16 = False
    embedder.pool = None
    embedder._encode_batch = lambda *args, **kwargs: torch.ones((1, 4), dtype=torch.bfloat16)

    embeddings = MultimodalEmbedder.encode_single_device(
        embedder,
        inputs=[{"text": "query"}],
        batch_size=1,
        max_length=8,
        convert_to_numpy=True,
        device="cpu",
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.dtype == np.float32


def test_extract_multimodal_hidden_states_prefers_last_hidden_state():
    outputs = SimpleNamespace(
        last_hidden_state=torch.tensor([[1.0, 2.0]]),
        hidden_states=[torch.tensor([[3.0, 4.0]])],
    )

    hidden_states = extract_multimodal_hidden_states(outputs)

    assert torch.equal(hidden_states, outputs.last_hidden_state)


def test_processor_adapter_retries_without_truncation_on_mm_token_mismatch():
    adapter = MultimodalProcessorAdapter.__new__(MultimodalProcessorAdapter)
    adapter.processor = None
    adapter.model_type = "qwen2_vl"
    adapter.use_chat_template = False
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Yp7cAAAAASUVORK5CYII="

    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("truncation"):
                raise ValueError("Mismatch in `image` token count between text and `input_ids`.")
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    adapter.processor = FakeProcessor()

    result = MultimodalProcessorAdapter._encode_single(
        adapter,
        item={"text": "question", "images": [{"b64": tiny_png_b64}], "videos": []},
        max_length=8,
    )

    assert result["input_ids"] == [[1, 2, 3]]
    assert adapter.processor.calls[0]["truncation"] is True
    assert adapter.processor.calls[0]["max_length"] == 8
    assert adapter.processor.calls[1]["truncation"] is False
    assert "max_length" not in adapter.processor.calls[1]


def test_processor_adapter_merges_processor_call_kwargs():
    recorded = {}

    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([feature["input_ids"][0] for feature in features], dtype=torch.long),
                "attention_mask": torch.tensor([feature["attention_mask"][0] for feature in features], dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, **kwargs):
            recorded["kwargs"] = kwargs
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    adapter = MultimodalProcessorAdapter(
        FakeProcessor(),
        model_type="qwen2_vl",
        use_chat_template=False,
        processor_call_kwargs={"max_pixels": 2048},
    )

    adapter.encode_batch([{"text": "question"}], max_length=16)

    assert recorded["kwargs"]["max_pixels"] == 2048
    assert recorded["kwargs"]["max_length"] == 16


def test_processor_adapter_translates_llava_next_processor_call_pixel_budgets():
    recorded = {}

    class FakeTokenizer:
        def pad(self, features, padding=True, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([feature["input_ids"][0] for feature in features], dtype=torch.long),
                "attention_mask": torch.tensor([feature["attention_mask"][0] for feature in features], dtype=torch.long),
            }

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, **kwargs):
            recorded["kwargs"] = kwargs
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    adapter = MultimodalProcessorAdapter(
        FakeProcessor(),
        model_type="llava_next",
        use_chat_template=False,
        processor_call_kwargs={"max_pixels": 262144},
    )

    adapter.encode_batch([{"text": "question"}], max_length=16)

    assert "max_pixels" not in recorded["kwargs"]
    assert recorded["kwargs"]["size"] == {"shortest_edge": 512}
    assert recorded["kwargs"]["crop_size"] == {"height": 512, "width": 512}
    assert recorded["kwargs"]["max_length"] == 16


def test_load_multimodal_backbone_marks_prefer_base_model_output_mode(monkeypatch):
    class DummyConfig:
        model_type = "qwen3_5"

    class DummyModel:
        pass

    monkeypatch.setattr("Nexus.modules.multimodal._maybe_load_peft_config", lambda *args, **kwargs: None)
    monkeypatch.setattr("Nexus.modules.multimodal.AutoConfig.from_pretrained", lambda *args, **kwargs: DummyConfig())
    monkeypatch.setattr(
        "Nexus.modules.multimodal._maybe_load_base_model_from_conditional_generation_wrapper",
        lambda *args, **kwargs: DummyModel(),
    )

    model, _ = load_multimodal_backbone(
        "demo-model",
        model_type="qwen3_5",
        backbone_load_strategy="prefer_base_model",
    )

    assert infer_multimodal_output_mode(model) == "last_hidden_state"


def test_build_casts_lora_trainable_params_to_fp32(monkeypatch):
    from Nexus.training.embedder.multimodal_retrieval.arguments import (
        MultimodalEmbedderModelArguments,
        WrappedMultimodalEmbedderModelArguments,
    )
    from Nexus.training.embedder.multimodal_retrieval.modeling import BiMultimodalEmbedderModel

    class DummyConfig:
        model_type = "qwen2_vl"

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.ones(2, dtype=torch.float16), requires_grad=True)
            self.frozen = torch.nn.Parameter(torch.ones(2, dtype=torch.float16), requires_grad=False)
            self.peft_config = {"default": object()}

    dummy_model = DummyModel()

    monkeypatch.setattr(
        "Nexus.training.embedder.multimodal_retrieval.modeling.load_multimodal_processor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "Nexus.training.embedder.multimodal_retrieval.modeling.load_multimodal_backbone",
        lambda **kwargs: (dummy_model, DummyConfig()),
    )
    monkeypatch.setattr(
        "Nexus.training.embedder.multimodal_retrieval.modeling.get_peft_model",
        lambda model, config: model,
    )

    model_args = MultimodalEmbedderModelArguments(
        model_name_or_path="dummy",
        processor_name_or_path="dummy",
        model_type="qwen2_vl",
        torch_dtype="float16",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="q_proj",
    )
    wrapped = WrappedMultimodalEmbedderModelArguments(
        negatives_cross_device=False,
        temperature=0.02,
        sub_batch_size=1,
        kd_loss_type="kl_div",
        sentence_pooling_method="last_token",
        normalize_embeddings=True,
        query_max_len=128,
        passage_max_len=128,
        model_type="qwen2_vl",
        use_chat_template=True,
    )

    built = BiMultimodalEmbedderModel.build(model_args, wrapped)

    assert built.model.trainable.dtype == torch.float32
    assert built.model.frozen.dtype == torch.float16
