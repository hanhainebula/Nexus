import os
import unicodedata
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from Nexus.inference.embedder.multimodal_retrieval.generic import MultimodalEmbedder
from Nexus.modules.multimodal import (
    build_multimodal_forward_kwargs,
    extract_multimodal_hidden_states,
    move_batch_to_device,
)


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
DEFAULT_IMAGE_TARGET_INSTRUCTION = "Understand the content of the provided image."
DEFAULT_VIDEO_TARGET_INSTRUCTION = "Understand the content of the provided video."
QWEN3_DEFAULT_INSTRUCTION = "Represent the user's input."
QWEN3_MIN_PIXELS = 4 * 32 * 32
QWEN3_MAX_PIXELS = 1800 * 32 * 32
QWEN3_MAX_TOTAL_PIXELS = 10 * 768 * 32 * 32
QWEN_OFFICIAL_STYLE_MODEL_TYPES = {"qwen3_vl", "qwen3_5"}


def _ensure_instruction_punctuation(instruction: str) -> str:
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def _merge_instruction_into_text(text: Any, instruction: Any) -> str:
    text = "" if text is None else str(text).strip()
    instruction = "" if instruction is None else _ensure_instruction_punctuation(str(instruction))
    if not instruction:
        return text
    if not text:
        return instruction
    if text.startswith(instruction):
        return text
    return f"{instruction} {text}".strip()


def _looks_like_frame_path(value: Any) -> bool:
    if not isinstance(value, str) or value in [None, ""]:
        return False
    return value.lower().split("?", 1)[0].endswith(IMAGE_SUFFIXES)


def _load_frame_as_pil(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _as_positive_int(value: Any) -> Optional[int]:
    if value in [None, "", "None"]:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_positive_float(value: Any) -> Optional[float]:
    if value in [None, "", "None"]:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_frame_limit(item_max_frames: Any, default_max_frames: Any) -> Optional[int]:
    item_limit = _as_positive_int(item_max_frames)
    default_limit = _as_positive_int(default_max_frames)
    if item_limit is not None and default_limit is not None:
        return min(item_limit, default_limit)
    return item_limit if item_limit is not None else default_limit


def _sample_sequence_evenly(values: Sequence[Any], max_items: Optional[int]) -> list:
    values = list(values)
    if max_items is None or len(values) <= max_items:
        return values
    if max_items == 1:
        return [values[len(values) // 2]]
    step = (len(values) - 1) / float(max_items - 1)
    return [values[min(int(round(index * step)), len(values) - 1)] for index in range(max_items)]


def _qwen3_media_reference(value: Any) -> Any:
    if isinstance(value, str) and not value.startswith(("http://", "https://", "file://")):
        return "file://" + value
    return value


def _qwen3_normalize_list(value: Any) -> list:
    if value in [None, "", []]:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _qwen3_collect_values(item: Dict[str, Any], keys) -> list:
    collected = []
    for key in keys:
        value = item.get(key)
        if value in [None, "", []]:
            continue
        if isinstance(value, (list, tuple)):
            collected.extend(list(value))
        else:
            collected.append(value)
    return collected


def _qwen3_is_single_video(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)) and len(value) > 0:
        first_value = value[0]
        if isinstance(first_value, Image.Image):
            return True
        if isinstance(first_value, str):
            return _looks_like_frame_path(first_value)
    return False


def _qwen3_collect_videos(item: Dict[str, Any]) -> list:
    videos = []
    for key in ["video", "video_path"]:
        value = item.get(key)
        if value in [None, "", []]:
            continue
        videos.append(value)
    for key in ["videos", "video_paths"]:
        value = item.get(key)
        if value in [None, "", []]:
            continue
        if _qwen3_is_single_video(value):
            videos.append(value)
        elif isinstance(value, (list, tuple)):
            videos.extend(list(value))
        else:
            videos.append(value)
    return videos


def _qwen3_resolve_default_instruction(item: Dict[str, Any], default_instruction: Optional[str]) -> str:
    instruction = item.get("instruction")
    if instruction not in [None, ""]:
        return _ensure_instruction_punctuation(str(instruction))
    return default_instruction or QWEN3_DEFAULT_INSTRUCTION


def _qwen3_format_video_content(
    video_value: Any,
    *,
    fps: Any,
    max_frames: Any,
    total_pixels: int,
) -> Dict[str, Any]:
    normalized_fps = _as_positive_float(fps)
    frame_limit = _as_positive_int(max_frames)

    if isinstance(video_value, dict):
        for frame_key in ["frames", "paths", "frame_paths"]:
            frame_values = video_value.get(frame_key)
            if isinstance(frame_values, (list, tuple)) and len(frame_values) > 0:
                sampled = _sample_sequence_evenly(frame_values, frame_limit)
                return {
                    "type": "video",
                    "video": [_qwen3_media_reference(frame) for frame in sampled],
                    "total_pixels": total_pixels,
                }

        path_value = video_value.get("path") or video_value.get("video") or video_value.get("video_path")
        if path_value not in [None, ""]:
            content = {
                "type": "video",
                "video": _qwen3_media_reference(path_value),
                "fps": _as_positive_float(video_value.get("fps")) or normalized_fps,
                "max_frames": _as_positive_int(video_value.get("max_frames")) or frame_limit,
            }
            return {key: value for key, value in content.items() if value is not None}

    if isinstance(video_value, (list, tuple)):
        sampled = _sample_sequence_evenly(video_value, frame_limit)
        return {
            "type": "video",
            "video": [_qwen3_media_reference(frame) for frame in sampled],
            "total_pixels": total_pixels,
        }

    if isinstance(video_value, str):
        content = {
            "type": "video",
            "video": _qwen3_media_reference(video_value),
            "fps": normalized_fps,
            "max_frames": frame_limit,
        }
        return {key: value for key, value in content.items() if value is not None}

    raise TypeError(f"Unrecognized video type for Qwen3-VL input: {type(video_value)}")


def _qwen3_format_conversation(
    item: Any,
    *,
    default_instruction: Optional[str],
    default_fps: Any,
    default_max_frames: Any,
    min_pixels: int,
    max_pixels: int,
    total_pixels: int,
) -> list:
    if not isinstance(item, dict):
        item = {"text": "" if item is None else str(item)}

    instruction = _qwen3_resolve_default_instruction(item, default_instruction)
    fps = item.get("fps", default_fps)
    max_frames = _resolve_frame_limit(item.get("max_frames"), default_max_frames)

    content = []
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]

    for video_value in _qwen3_collect_videos(item):
        content.append(
            _qwen3_format_video_content(
                video_value,
                fps=fps,
                max_frames=max_frames,
                total_pixels=total_pixels,
            )
        )

    for image_value in _qwen3_collect_values(item, ["image", "images", "image_path", "image_paths"]):
        content.append(
            {
                "type": "image",
                "image": _qwen3_media_reference(image_value),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    for text_value in _qwen3_normalize_list(item.get("text")):
        content.append({"type": "text", "text": "" if text_value is None else str(text_value)})

    if len(content) == 0:
        content.append({"type": "text", "text": "NULL"})
    return conversation


def _coerce_frame_sequence_to_pil(frame_values: Sequence[Any], max_frames: Optional[int]) -> list:
    sampled_frames = _sample_sequence_evenly(frame_values, max_frames)
    return [_load_frame_as_pil(frame) if _looks_like_frame_path(frame) else frame for frame in sampled_frames]


def _materialize_frame_list(video_value: Any, max_frames: Any = None, fps: Any = None) -> Any:
    """Convert frame-only path lists to PIL frames before Nexus sees them.

    Older Nexus video handling may pass a list of frame path strings to a video
    processor as if they were raw video inputs. Supplying PIL frames avoids that
    ambiguity while keeping real raw video paths unchanged.
    """
    frame_limit = _as_positive_int(max_frames)
    normalized_fps = _as_positive_float(fps)

    if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
        if all(_looks_like_frame_path(path) for path in video_value):
            return {"frames": _coerce_frame_sequence_to_pil(video_value, frame_limit)}
        if frame_limit is not None:
            return _sample_sequence_evenly(video_value, frame_limit)
        return video_value

    if isinstance(video_value, dict):
        prepared_video = deepcopy(video_value)
        video_frame_limit = _resolve_frame_limit(prepared_video.get("max_frames"), frame_limit)
        if video_frame_limit is not None:
            prepared_video["max_frames"] = video_frame_limit
        if normalized_fps is not None and prepared_video.get("fps") in [None, "", "None"]:
            prepared_video["fps"] = normalized_fps

        for frame_key in ["frames", "paths", "frame_paths"]:
            frame_values = prepared_video.get(frame_key)
            if isinstance(frame_values, (list, tuple)) and len(frame_values) > 0:
                prepared_video.pop("paths", None)
                prepared_video.pop("frame_paths", None)
                prepared_video["frames"] = _coerce_frame_sequence_to_pil(frame_values, video_frame_limit)
                break
        return prepared_video

    if isinstance(video_value, str) and not _looks_like_frame_path(video_value):
        prepared_video = {"path": video_value}
        if frame_limit is not None:
            prepared_video["max_frames"] = frame_limit
        if normalized_fps is not None:
            prepared_video["fps"] = normalized_fps
        return prepared_video

    return video_value


def _has_media_value(item: Dict[str, Any], keys) -> bool:
    for key in keys:
        value = item.get(key)
        if value not in [None, "", []]:
            return True
    return False


def _prepare_single_input(item: Any, default_fps: Any = None, default_max_frames: Any = None) -> Any:
    if not isinstance(item, dict):
        return item

    prepared = deepcopy(item)
    instruction = prepared.pop("instruction", None)
    fps = prepared.pop("fps", default_fps)
    max_frames = _resolve_frame_limit(prepared.pop("max_frames", None), default_max_frames)
    text = prepared.get("text", "")

    if instruction in [None, ""] and str(text or "").strip() == "":
        if _has_media_value(prepared, ["video", "videos", "video_path", "video_paths"]):
            instruction = DEFAULT_VIDEO_TARGET_INSTRUCTION
        elif _has_media_value(prepared, ["image", "images", "image_path", "image_paths"]):
            instruction = DEFAULT_IMAGE_TARGET_INSTRUCTION

    if instruction not in [None, ""]:
        prepared["text"] = _merge_instruction_into_text(text, instruction)

    if "video" in prepared:
        prepared["video"] = _materialize_frame_list(prepared["video"], max_frames=max_frames, fps=fps)
    if "videos" in prepared and isinstance(prepared["videos"], list):
        prepared["videos"] = [
            _materialize_frame_list(video, max_frames=max_frames, fps=fps) for video in prepared["videos"]
        ]

    return prepared


def prepare_nexus_inputs(inputs: Any, default_fps: Any = None, default_max_frames: Any = None) -> Any:
    if isinstance(inputs, list):
        return [
            _prepare_single_input(item, default_fps=default_fps, default_max_frames=default_max_frames)
            for item in inputs
        ]
    return _prepare_single_input(inputs, default_fps=default_fps, default_max_frames=default_max_frames)


class NexusMultimodalCompatEmbedder(MultimodalEmbedder):
    @torch.no_grad()
    def forward(self, model_inputs):
        device = next(self.model.parameters()).device
        model_inputs = move_batch_to_device(model_inputs, device)

        outputs = self.model(
            **model_inputs,
            **build_multimodal_forward_kwargs(self.model),
        )
        last_hidden_state = extract_multimodal_hidden_states(outputs)

        return {
            "last_hidden_state": last_hidden_state,
            "attention_mask": model_inputs["attention_mask"],
        }

    def _qwen3_official_conversations(self, inputs):
        default_instruction = getattr(self, "default_instruction", None)
        if default_instruction in [None, ""]:
            default_instruction = getattr(self, "query_instruction_for_retrieval", None)
        min_pixels = _as_positive_int(getattr(self, "min_pixels", None)) or QWEN3_MIN_PIXELS
        max_pixels = _as_positive_int(getattr(self, "max_pixels", None)) or QWEN3_MAX_PIXELS
        total_pixels = _as_positive_int(getattr(self, "total_pixels", None)) or QWEN3_MAX_TOTAL_PIXELS

        if not isinstance(inputs, list):
            inputs = [inputs]

        return [
            _qwen3_format_conversation(
                item,
                default_instruction=default_instruction,
                default_fps=getattr(self, "default_fps", None),
                default_max_frames=getattr(self, "default_max_frames", None),
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                total_pixels=total_pixels,
            )
            for item in inputs
        ]

    def _qwen3_preprocess_inputs(self, conversations):
        try:
            from qwen_vl_utils.vision_process import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL official-style MMEB evaluation requires `qwen-vl-utils`. "
                "Install it in the active Nexus environment before running evaluation."
            ) from exc

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception:
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=max(self.query_max_length, self.passage_max_length),
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @torch.no_grad()
    def _process_qwen3_official_style(self, inputs, normalize=True):
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = None
        if target_device is None or str(target_device) == "cpu":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            target_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(target_device)
        if self.use_fp16 and target_device != "cpu":
            self.model.half()

        conversations = self._qwen3_official_conversations(inputs)
        model_inputs = self._qwen3_preprocess_inputs(conversations)
        outputs = self.forward(model_inputs)
        embeddings = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @torch.no_grad()
    def process(self, inputs, normalize=True):
        if getattr(self, "model_type", None) in QWEN_OFFICIAL_STYLE_MODEL_TYPES and hasattr(self.processor, "apply_chat_template"):
            return self._process_qwen3_official_style(inputs, normalize=normalize)

        inputs = prepare_nexus_inputs(
            inputs,
            default_fps=getattr(self, "default_fps", None),
            default_max_frames=getattr(self, "default_max_frames", None),
        )
        embeddings = self.encode(
            inputs,
            convert_to_numpy=False,
        )

        # MultimodalEmbedder.encode() returns CPU tensors by default.
        # Move back to model device so NCCL all_gather can work.
        device = next(self.model.parameters()).device
        embeddings = embeddings.to(device, non_blocking=True)

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings
