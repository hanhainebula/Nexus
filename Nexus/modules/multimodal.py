import base64
import io
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor


DEFAULT_MULTIMODAL_MODEL_TYPE = "auto"
CHAT_TEMPLATE_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "llava_next"}
CONDITIONAL_GENERATION_MODEL_TYPES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "llava_next": "LlavaNextForConditionalGeneration",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")
VIDEO_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
DEFAULT_VIDEO_NUM_FRAMES = 8


def resolve_torch_dtype(dtype_name: Optional[str]):
    if dtype_name is None or dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def infer_multimodal_model_type(config) -> str:
    if config is None:
        return DEFAULT_MULTIMODAL_MODEL_TYPE
    return getattr(config, "model_type", DEFAULT_MULTIMODAL_MODEL_TYPE) or DEFAULT_MULTIMODAL_MODEL_TYPE


def _maybe_load_peft_config(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
):
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    if os.path.isdir(model_name_or_path) and not os.path.exists(adapter_config_path):
        return None

    try:
        from peft import PeftConfig

        return PeftConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception:
        return None


def load_multimodal_processor(
    model_name_or_path: str,
    processor_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
):
    processor_path = processor_name_or_path or model_name_or_path
    try:
        return AutoProcessor.from_pretrained(
            processor_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=token,
        )
    except Exception:
        peft_config = _maybe_load_peft_config(
            processor_path,
            cache_dir=cache_dir,
            token=token,
        )
        if peft_config is None:
            raise
        return AutoProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=token,
        )


def _maybe_load_registered_conditional_generation_model(
    config,
    model_name_or_path: str,
    load_kwargs: Dict[str, Any],
):
    model_type = infer_multimodal_model_type(config)
    model_cls = _get_registered_conditional_generation_model_class(model_type)
    if model_cls is None:
        return None
    return model_cls.from_pretrained(model_name_or_path, **load_kwargs)


def _get_registered_conditional_generation_model_class(
    model_type: str,
    transformers_module=None,
):
    class_name = CONDITIONAL_GENERATION_MODEL_TYPES.get(model_type)
    if class_name is None:
        return None

    if transformers_module is None:
        import transformers as transformers_module

    model_cls = getattr(transformers_module, class_name, None)
    if model_cls is not None:
        return model_cls

    transformers_version = getattr(transformers_module, "__version__", "unknown")
    if model_type == "qwen3_vl":
        version_hint = "transformers>=4.57.3"
    else:
        version_hint = "a newer transformers version with the required multimodal classes"
    raise ImportError(
        f"Model type '{model_type}' requires `{class_name}`, but it is unavailable in "
        f"transformers {transformers_version}. Install Nexus with `pip install -e .[multimodal]` "
        f"or upgrade to {version_hint}."
    )


def load_multimodal_backbone(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    model_type: str = DEFAULT_MULTIMODAL_MODEL_TYPE,
    torch_dtype_name: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    peft_is_trainable: bool = False,
):
    peft_config = _maybe_load_peft_config(
        model_name_or_path,
        cache_dir=cache_dir,
        token=token,
    )
    base_model_name_or_path = (
        peft_config.base_model_name_or_path
        if peft_config is not None
        else model_name_or_path
    )

    config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if model_type not in [None, "", DEFAULT_MULTIMODAL_MODEL_TYPE]:
        setattr(config, "model_type", model_type)

    load_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "token": token,
        "config": config,
    }
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = _maybe_load_registered_conditional_generation_model(
        config=config,
        model_name_or_path=base_model_name_or_path,
        load_kwargs=load_kwargs,
    )
    if model is None:
        try:
            model = AutoModel.from_pretrained(base_model_name_or_path, **load_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **load_kwargs)

    if peft_config is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            model_name_or_path,
            is_trainable=peft_is_trainable,
        )
    return model, config


def _clone_image(image):
    return image.copy() if hasattr(image, "copy") else image


def _select_media_base_dir(base_dir: Optional[Any], media_type: str) -> Optional[str]:
    if isinstance(base_dir, dict):
        for key in [media_type, f"{media_type}_root", "media", "media_root", "default", "base_dir"]:
            value = base_dir.get(key)
            if value not in [None, ""]:
                return value
        return None
    return base_dir


def looks_like_remote_reference(path: Optional[str]) -> bool:
    if path in [None, ""]:
        return False
    return "://" in path


def resolve_media_path(path: str, base_dir: Optional[Any] = None, media_type: str = "media") -> str:
    if path is None:
        return None
    selected_base_dir = _select_media_base_dir(base_dir, media_type=media_type)
    if os.path.isabs(path) or selected_base_dir is None or looks_like_remote_reference(path):
        return path

    resolved_path = os.path.normpath(os.path.join(selected_base_dir, path))
    if os.path.exists(resolved_path):
        return resolved_path

    normalized_base_dir = os.path.normpath(selected_base_dir)
    normalized_path = os.path.normpath(path)
    base_name = os.path.basename(normalized_base_dir)
    path_parts = normalized_path.split(os.sep)
    if len(path_parts) > 1 and path_parts[0] == base_name:
        fallback_path = os.path.normpath(os.path.join(os.path.dirname(normalized_base_dir), normalized_path))
        if os.path.exists(fallback_path):
            return fallback_path

    return resolved_path


def resolve_root_override_path(path: Optional[str], default_root: Optional[str], media_type: str) -> Optional[str]:
    if path in [None, ""]:
        return None
    if os.path.isabs(path) or looks_like_remote_reference(path):
        return path

    cwd_candidate = os.path.normpath(path)
    if os.path.exists(cwd_candidate):
        return os.path.abspath(cwd_candidate)

    return resolve_media_path(path, base_dir=default_root, media_type=media_type)


def looks_like_image_reference(path: Optional[str]) -> bool:
    if path in [None, ""]:
        return False
    lowered = path.lower().split("?", 1)[0]
    return lowered.endswith(IMAGE_SUFFIXES)


def looks_like_video_reference(path: Optional[str]) -> bool:
    if path in [None, ""]:
        return False
    lowered = path.lower().split("?", 1)[0]
    return lowered.endswith(VIDEO_SUFFIXES)


def build_media_base_dir(
    base_dir: Optional[Any] = None,
    media_root: Optional[str] = None,
    image_root: Optional[str] = None,
    video_root: Optional[str] = None,
):
    if isinstance(base_dir, dict):
        normalized_base_dir = deepcopy(base_dir)
        default_root = normalized_base_dir.get("default", normalized_base_dir.get("base_dir"))
    else:
        normalized_base_dir = {}
        default_root = base_dir

    if default_root not in [None, ""]:
        normalized_base_dir.setdefault("default", default_root)
        normalized_base_dir.setdefault("base_dir", default_root)

    if media_root not in [None, ""]:
        normalized_base_dir["media"] = resolve_root_override_path(
            media_root,
            default_root=default_root,
            media_type="media",
        )

    if image_root not in [None, ""]:
        normalized_base_dir["image"] = resolve_root_override_path(
            image_root,
            default_root=default_root,
            media_type="image",
        )
    elif normalized_base_dir.get("media") not in [None, ""] and normalized_base_dir.get("image") in [None, ""]:
        normalized_base_dir["image"] = normalized_base_dir["media"]

    if video_root not in [None, ""]:
        normalized_base_dir["video"] = resolve_root_override_path(
            video_root,
            default_root=default_root,
            media_type="video",
        )
    elif normalized_base_dir.get("media") not in [None, ""] and normalized_base_dir.get("video") in [None, ""]:
        normalized_base_dir["video"] = normalized_base_dir["media"]

    if normalized_base_dir.get("image") in [None, ""] and default_root not in [None, ""]:
        normalized_base_dir["image"] = default_root
    if normalized_base_dir.get("video") in [None, ""] and default_root not in [None, ""]:
        normalized_base_dir["video"] = default_root

    return normalized_base_dir or None


def _coerce_image_specs(
    image_value: Any,
    base_dir: Optional[Any] = None,
    media_type: str = "image",
) -> List[Dict[str, Any]]:
    if image_value in [None, "", []]:
        return []

    if not isinstance(image_value, list):
        image_value = [image_value]

    normalized_specs = []
    for image_spec in image_value:
        if image_spec in [None, ""]:
            continue
        if isinstance(image_spec, str):
            normalized_specs.append({"path": resolve_media_path(image_spec, base_dir=base_dir, media_type=media_type)})
        elif isinstance(image_spec, bytes):
            normalized_specs.append({"bytes": image_spec})
        elif isinstance(image_spec, dict):
            spec = deepcopy(image_spec)
            if "path" in spec and spec["path"] is not None:
                spec["path"] = resolve_media_path(spec["path"], base_dir=base_dir, media_type=media_type)
            if "paths" in spec and spec["paths"] is not None:
                paths = []
                for path in spec["paths"]:
                    if path in [None, ""]:
                        paths.append(path)
                    else:
                        paths.append(resolve_media_path(path, base_dir=base_dir, media_type=media_type))
                spec["paths"] = paths
            if "frames" in spec and spec["frames"] not in [None, "", []]:
                spec["frames"] = _coerce_image_specs(spec["frames"], base_dir=base_dir, media_type="image")
            normalized_specs.append(spec)
        else:
            normalized_specs.append({"pil": image_spec})
    return normalized_specs


def _coerce_video_specs(video_value: Any, base_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    if video_value in [None, "", []]:
        return []

    if not isinstance(video_value, list):
        video_value = [video_value]

    normalized_specs = []
    for video_spec in video_value:
        if video_spec in [None, ""]:
            continue
        if isinstance(video_spec, str):
            resolved_path = resolve_media_path(video_spec, base_dir=base_dir, media_type="video")
            if looks_like_image_reference(resolved_path):
                normalized_specs.append({"paths": [resolved_path]})
            else:
                normalized_specs.append({"path": resolved_path})
        elif isinstance(video_spec, dict):
            spec = deepcopy(video_spec)
            if "path" in spec and spec["path"] not in [None, ""]:
                spec["path"] = resolve_media_path(spec["path"], base_dir=base_dir, media_type="video")
            if "video_path" in spec and spec["video_path"] not in [None, ""]:
                spec["path"] = resolve_media_path(spec.pop("video_path"), base_dir=base_dir, media_type="video")
            if "paths" in spec and spec["paths"] is not None:
                spec["paths"] = [
                    resolve_media_path(path, base_dir=base_dir, media_type="image") if path not in [None, ""] else path
                    for path in spec["paths"]
                ]
            if "frame_paths" in spec and spec["frame_paths"] is not None:
                spec["paths"] = [
                    resolve_media_path(path, base_dir=base_dir, media_type="image") if path not in [None, ""] else path
                    for path in spec.pop("frame_paths")
                ]
            if "frames" in spec and spec["frames"] not in [None, "", []]:
                spec["frames"] = _coerce_image_specs(spec["frames"], base_dir=base_dir, media_type="image")
            normalized_specs.append(spec)
        else:
            normalized_specs.append({"frames": _coerce_image_specs(video_spec, base_dir=base_dir, media_type="image")})
    return normalized_specs


def normalize_multimodal_item(item: Any, base_dir: Optional[str] = None) -> Dict[str, Any]:
    if item is None:
        return {"text": "", "images": [], "videos": [], "metadata": {}}
    if isinstance(item, str):
        if looks_like_image_reference(item):
            return {"text": "", "images": _coerce_image_specs(item, base_dir=base_dir), "videos": [], "metadata": {}}
        if looks_like_video_reference(item):
            return {"text": "", "images": [], "videos": _coerce_video_specs(item, base_dir=base_dir), "metadata": {}}
        return {"text": item, "images": [], "videos": [], "metadata": {}}
    if not isinstance(item, dict):
        raise TypeError(f"Unsupported multimodal item type: {type(item)}")

    metadata = deepcopy(item.get("metadata", {}))
    title = item.get("title", "")
    text = item.get("text")
    if text is None:
        text = item.get("query")
    if text is None:
        text = item.get("caption")
    if text is None:
        text = item.get("body", "")
    text = text or ""
    if title:
        text = f"{title} {text}".strip()

    image_value = None
    for key in ["images", "image_paths", "image_path", "image", "pages"]:
        if key in item and item[key] not in [None, "", []]:
            image_value = item[key]
            break

    video_value = None
    if "video_frames" in item and item["video_frames"] not in [None, "", []]:
        video_frames = item["video_frames"]
        if isinstance(video_frames, dict):
            video_value = [video_frames]
        elif isinstance(video_frames, list):
            video_value = [{"paths": video_frames}]
        else:
            video_value = [{"paths": [video_frames]}]
    else:
        for key in ["videos", "video_paths", "video_path", "video"]:
            if key in item and item[key] not in [None, "", []]:
                video_value = item[key]
                break

    return {
        "text": text,
        "images": _coerce_image_specs(image_value, base_dir=base_dir),
        "videos": _coerce_video_specs(video_value, base_dir=base_dir),
        "metadata": metadata,
    }


def normalize_multimodal_group(items: Any, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if items in [None, "", []]:
        return []
    if isinstance(items, list):
        return [normalize_multimodal_item(item, base_dir=base_dir) for item in items if item not in [None, ""]]
    return [normalize_multimodal_item(items, base_dir=base_dir)]


def build_prefixed_multimodal_group(row: Dict[str, Any], prefix: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if prefix in row:
        return normalize_multimodal_group(row[prefix], base_dir=base_dir)

    text_value = row.get(f"{prefix}_text")
    title_value = row.get(f"{prefix}_title")
    multi_image_value = None
    for key in [f"{prefix}_images", f"{prefix}_image_paths", f"{prefix}_pages"]:
        if key in row and row[key] not in [None, "", []]:
            multi_image_value = row[key]
            break
    multi_video_value = None
    multi_video_field = None
    if f"{prefix}_video_frames" in row and row[f"{prefix}_video_frames"] not in [None, "", []]:
        multi_video_value = row[f"{prefix}_video_frames"]
        multi_video_field = f"{prefix}_video_frames"
    else:
        for key in [f"{prefix}_videos", f"{prefix}_video_paths"]:
            if key in row and row[key] not in [None, "", []]:
                multi_video_value = row[key]
                multi_video_field = key
                break

    single_image_value = row.get(f"{prefix}_image_path", row.get(f"{prefix}_image"))
    single_video_value = row.get(f"{prefix}_video_path", row.get(f"{prefix}_video"))
    if multi_image_value is not None:
        return [
            normalize_multimodal_item(
                {"title": title_value, "text": text_value, "images": multi_image_value},
                base_dir=base_dir,
            )
        ]

    if multi_video_value is not None:
        if multi_video_field == f"{prefix}_video_frames":
            return [
                normalize_multimodal_item(
                    {"title": title_value, "text": text_value, "video_frames": multi_video_value},
                    base_dir=base_dir,
                )
            ]
        return [
            normalize_multimodal_item(
                {"title": title_value, "text": text_value, "videos": multi_video_value},
                base_dir=base_dir,
            )
        ]

    if single_video_value not in [None, ""]:
        if isinstance(text_value, list) and isinstance(single_video_value, list) and len(text_value) == len(single_video_value):
            return [
                normalize_multimodal_item(
                    {"title": title_value, "text": text, "video_path": video},
                    base_dir=base_dir,
                )
                for text, video in zip(text_value, single_video_value)
            ]
        if isinstance(single_video_value, list) and all(looks_like_image_reference(path) for path in single_video_value):
            payload = {"title": title_value, "text": text_value, "video_frames": single_video_value}
        else:
            payload = {"title": title_value, "text": text_value, "video_path": single_video_value}
        return [normalize_multimodal_item(payload, base_dir=base_dir)]

    if isinstance(text_value, list) or isinstance(single_image_value, list):
        if isinstance(text_value, list):
            texts = text_value
        else:
            texts = [text_value for _ in range(len(single_image_value))]

        if isinstance(single_image_value, list):
            images = single_image_value
        else:
            images = [single_image_value for _ in range(len(texts))]

        group = []
        for text, image in zip(texts, images):
            group.append(
                normalize_multimodal_item(
                    {"title": title_value, "text": text, "image_path": image},
                    base_dir=base_dir,
                )
            )
        return group

    if text_value not in [None, ""] or title_value not in [None, ""] or single_image_value not in [None, ""]:
        return [
            normalize_multimodal_item(
                {"title": title_value, "text": text_value, "image_path": single_image_value},
                base_dir=base_dir,
            )
        ]
    return []


def apply_instruction(text: str, instruction: Optional[str], instruction_format: str) -> str:
    if instruction is None:
        return text
    return instruction_format.format(instruction, text)


def is_empty_multimodal_item(item: Dict[str, Any]) -> bool:
    return (
        (item.get("text", "").strip() == "")
        and len(item.get("images", [])) == 0
        and len(item.get("videos", [])) == 0
    )


def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError("Pillow is required for multimodal retrieval. Please install `Pillow`.") from exc


def load_image_from_spec(image_spec: Dict[str, Any]):
    _require_pillow()
    from PIL import Image

    if image_spec is None:
        return None
    if "pil" in image_spec and image_spec["pil"] is not None:
        return _clone_image(image_spec["pil"]).convert("RGB")
    if "bytes" in image_spec and image_spec["bytes"] not in [None, b""]:
        return Image.open(io.BytesIO(image_spec["bytes"])).convert("RGB")
    if "b64" in image_spec and image_spec["b64"] not in [None, ""]:
        return Image.open(io.BytesIO(base64.b64decode(image_spec["b64"]))).convert("RGB")
    if "path" in image_spec and image_spec["path"] not in [None, ""]:
        return Image.open(image_spec["path"]).convert("RGB")
    raise ValueError(f"Unsupported image specification: {image_spec}")


def _require_decord():
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "decord is required for raw video decoding fallback. Please install `decord` in the multimodal environment."
        ) from exc
    return VideoReader, cpu


def _resolve_video_num_frames(video_spec: Dict[str, Any], default_num_frames: int = DEFAULT_VIDEO_NUM_FRAMES) -> int:
    for key in ["num_frames", "sample_frames", "max_frames", "n_frames"]:
        value = video_spec.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return default_num_frames


def _sample_video_indices(total_frames: int, target_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= target_frames:
        return list(range(total_frames))
    step = total_frames / float(target_frames)
    return [min(int(step * idx + step / 2.0), total_frames - 1) for idx in range(target_frames)]


def _load_video_frames_from_path(video_path: str, num_frames: int = DEFAULT_VIDEO_NUM_FRAMES) -> List[Any]:
    _require_pillow()
    from PIL import Image

    VideoReader, cpu = _require_decord()
    video_reader = VideoReader(video_path, ctx=cpu(0))
    sampled_indices = _sample_video_indices(len(video_reader), target_frames=num_frames)
    if len(sampled_indices) == 0:
        return []
    frame_batch = video_reader.get_batch(sampled_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frame_batch]


def _build_frame_specs_from_video_spec(video_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if video_spec is None:
        return []
    if "frames" in video_spec and video_spec["frames"] not in [None, "", []]:
        return _coerce_image_specs(video_spec["frames"], media_type="image")

    paths = video_spec.get("paths") or []
    bytes_values = video_spec.get("bytes") or []
    b64_values = video_spec.get("b64s") or []

    if not isinstance(paths, list):
        paths = [paths]
    if not isinstance(bytes_values, list):
        bytes_values = [bytes_values]
    if not isinstance(b64_values, list):
        b64_values = [b64_values]

    num_frames = max(len(paths), len(bytes_values), len(b64_values))
    frame_specs = []
    for index in range(num_frames):
        frame_specs.append(
            {
                "path": paths[index] if index < len(paths) else None,
                "bytes": bytes_values[index] if index < len(bytes_values) else None,
                "b64": b64_values[index] if index < len(b64_values) else None,
            }
        )
    return frame_specs


def materialize_images(item: Dict[str, Any]) -> List[Any]:
    images = []
    for image_spec in item.get("images", []):
        if "paths" in image_spec:
            paths = image_spec.get("paths") or []
            bytes_values = image_spec.get("bytes") or [None] * len(paths)
            for path, bytes_value in zip(paths, bytes_values):
                if path in [None, ""] and bytes_value in [None, b""]:
                    continue
                images.append(load_image_from_spec({"path": path, "bytes": bytes_value}))
            continue
        images.append(load_image_from_spec(image_spec))
    return images


def load_video_from_spec(video_spec: Dict[str, Any], prefer_raw_video: bool = True):
    if video_spec is None:
        return None

    video_path = video_spec.get("path")
    if video_path not in [None, ""] and prefer_raw_video:
        return video_path

    frame_specs = _build_frame_specs_from_video_spec(video_spec)
    if len(frame_specs) > 0:
        materialized_frames = []
        for frame_spec in frame_specs:
            if (
                prefer_raw_video
                and frame_spec.get("path") not in [None, ""]
                and frame_spec.get("bytes") in [None, b""]
                and frame_spec.get("b64") in [None, ""]
            ):
                materialized_frames.append(frame_spec["path"])
            else:
                materialized_frames.append(load_image_from_spec(frame_spec))
        return materialized_frames

    if video_path not in [None, ""]:
        return _load_video_frames_from_path(video_path, num_frames=_resolve_video_num_frames(video_spec))

    raise ValueError(f"Unsupported video specification: {video_spec}")


def materialize_videos(item: Dict[str, Any], prefer_raw_video: bool = True) -> List[Any]:
    videos = []
    for video_spec in item.get("videos", []):
        videos.append(load_video_from_spec(video_spec, prefer_raw_video=prefer_raw_video))
    return videos


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class MultimodalProcessorAdapter:
    def __init__(
        self,
        processor,
        model_type: str = DEFAULT_MULTIMODAL_MODEL_TYPE,
        use_chat_template: bool = True,
    ):
        self.processor = processor
        self.model_type = model_type
        self.use_chat_template = use_chat_template

    def _supports_video_inputs(self) -> bool:
        return self.model_type in VIDEO_MODEL_TYPES or hasattr(self.processor, "video_processor")

    def _flatten_video_frames_to_images(self, videos: Sequence[Any]) -> List[Any]:
        flattened_frames = []
        for video in videos:
            if isinstance(video, (list, tuple)):
                flattened_frames.extend(video)
            else:
                flattened_frames.append(video)
        return flattened_frames

    def _build_chat_text(self, item: Dict[str, Any], images: Sequence[Any], videos: Sequence[Any]) -> str:
        contents = [{"type": "image"} for _ in images]
        contents.extend({"type": "video"} for _ in videos)
        if item.get("text", "").strip() or len(contents) == 0:
            contents.append({"type": "text", "text": item.get("text", "") or " "})
        messages = [{"role": "user", "content": contents}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _encode_single(
        self,
        item: Dict[str, Any],
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        processor_supports_videos = self._supports_video_inputs()
        images = materialize_images(item)
        videos = materialize_videos(item, prefer_raw_video=processor_supports_videos)
        processor_images = list(images)
        processor_videos = list(videos) if processor_supports_videos else []
        if not processor_supports_videos and len(videos) > 0:
            processor_images.extend(self._flatten_video_frames_to_images(videos))

        use_chat_template = (
            self.use_chat_template
            and self.model_type in CHAT_TEMPLATE_MODEL_TYPES
            and hasattr(self.processor, "apply_chat_template")
        )

        if use_chat_template:
            text = self._build_chat_text(item=item, images=processor_images, videos=processor_videos)
        else:
            text = item.get("text", "") or " "

        def _build_processor_kwargs(current_text, current_images, current_videos):
            call_kwargs = {
                "text": current_text,
                "padding": False,
                "truncation": max_length is not None,
                "return_tensors": None,
            }
            if max_length is not None:
                call_kwargs["max_length"] = max_length
            if len(current_images) > 0:
                call_kwargs["images"] = current_images
            if len(current_videos) > 0:
                call_kwargs["videos"] = current_videos
            return call_kwargs

        try:
            return self.processor(**_build_processor_kwargs(text, processor_images, processor_videos))
        except TypeError:
            if len(processor_videos) == 0:
                raise
            fallback_videos = materialize_videos(item, prefer_raw_video=False)
            fallback_images = list(images)
            fallback_images.extend(self._flatten_video_frames_to_images(fallback_videos))
            fallback_text = text
            if use_chat_template:
                fallback_text = self._build_chat_text(item=item, images=fallback_images, videos=[])
            return self.processor(**_build_processor_kwargs(fallback_text, fallback_images, []))
        except ValueError as exc:
            mismatch_error = "Mismatch in `image` token count" in str(exc) or "Mismatch in `video` token count" in str(exc)
            has_multimodal_inputs = len(processor_images) > 0 or len(processor_videos) > 0
            if not mismatch_error or not has_multimodal_inputs or max_length is None:
                raise

            retry_kwargs = _build_processor_kwargs(text, processor_images, processor_videos)
            retry_kwargs["truncation"] = False
            retry_kwargs.pop("max_length", None)
            return self.processor(**retry_kwargs)

    @staticmethod
    def _strip_singleton_batch(value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            return value[0]
        return value

    def _batch_with_tokenizer_pad(self, encoded_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "Current processor does not provide a `pad` method or a tokenizer. "
                "Please use a processor with padding support."
            )

        tokenizer_features = []
        for item in encoded_items:
            feature = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in item:
                    feature[key] = self._strip_singleton_batch(item[key])
            tokenizer_features.append(feature)

        batch = tokenizer.pad(tokenizer_features, padding=True, return_tensors="pt")
        for key in [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "image_sizes",
        ]:
            values = [item[key] for item in encoded_items if item.get(key) is not None]
            if len(values) == 0:
                continue
            batch[key] = torch.cat([torch.as_tensor(np.asarray(value)) for value in values], dim=0)
        return batch

    def encode_batch(
        self,
        items: Iterable[Dict[str, Any]],
        max_length: Optional[int] = None,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
    ) -> Dict[str, Any]:
        normalized_items = []
        for item in items:
            normalized_item = normalize_multimodal_item(item)
            normalized_item["text"] = apply_instruction(
                normalized_item.get("text", ""),
                instruction=instruction,
                instruction_format=instruction_format,
            )
            normalized_items.append(normalized_item)

        encoded_items = [self._encode_single(item, max_length=max_length) for item in normalized_items]

        pad_fn = getattr(self.processor, "pad", None)
        if callable(pad_fn):
            return pad_fn(encoded_items, padding=True, return_tensors="pt")
        return self._batch_with_tokenizer_pad(encoded_items)
