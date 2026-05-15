import os

from datasets import Dataset, load_dataset

from .video_classification_utils import DATASET_INSTRUCTION
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import save_frames, process_video_frames


def _normalize_instruction(instruction, fallback):
    instruction = (instruction or fallback or "Understand the action in the video.").strip()
    if instruction.endswith(":"):
        instruction = instruction[:-1] + "."
    return instruction


def _resolve_local_jsonl_path(data_args, kwargs):
    data_path = kwargs.get("data_path")
    if data_path and os.path.exists(data_path):
        return data_path

    data_basedir = getattr(data_args, "data_basedir", None)
    if data_basedir:
        candidate = os.path.join(data_basedir, "video-tasks", "data", "ssv2.jsonl")
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_video_path(video_root, video_id, original_video_path=None):
    candidates = []
    if original_video_path:
        candidates.append(original_video_path)
    candidates.extend(
        [
            os.path.join(video_root, str(video_id)),
            os.path.join(video_root, str(video_id) + ".mp4"),
            os.path.join(video_root, str(video_id) + ".avi"),
            os.path.join(video_root, str(video_id) + ".webm"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[1]


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    num_frames = kwargs['num_frames']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    dataset_name = kwargs['dataset_name']

    default_instruction = DATASET_INSTRUCTION.get(dataset_name, "Understand the action in the video.")

    query_inputs, cand_inputs, dataset_infos = [], [], []
    original_video_paths = batch_dict.get('video_path') or [None] * len(batch_dict['video_id'])
    instruction = _normalize_instruction(default_instruction, default_instruction)
    
    for video_id, pos_text, cand_text, original_video_path in zip(
        batch_dict['video_id'], 
        batch_dict['pos_text'], 
        batch_dict['neg_text'],
        original_video_paths,
    ):
        # Process video path and extract frames
        video_path = _resolve_video_path(video_root, video_id, original_video_path)
        frame_dir = os.path.join(frame_root, str(video_id))
        if not os.path.exists(frame_dir):
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        if not video_frame_paths:
            raise FileNotFoundError(f"No usable frames found for video_id={video_id} under {frame_dir}")

        # Query input: video as primary input with instruction describing the task
        query_inputs.append({
            "video": video_frame_paths,
            "instruction": instruction,
        })

        # Candidate input: plain text list (all candidate action descriptions including positive and negative samples)
        # In SSv2-MC (Multiple Choice) mode, cand_text typically contains multiple alternative actions
        cand_inputs.append([{"text": t} for t in cand_text])
        
        dataset_infos.append({
            "cand_names": cand_text,
            "label_name": pos_text,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "ssv2"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ssv2_dataset(model_args, data_args, **kwargs):
    """
    SSv2-MC setup for zero-shot evaluation.
    """
    dataset_name = kwargs['dataset_name']
    local_jsonl_path = _resolve_local_jsonl_path(data_args, kwargs)
    if local_jsonl_path is not None:
        dataset = Dataset.from_json(local_jsonl_path)
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
    dataset = sample_dataset(dataset, **kwargs)

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs), 
        batched=True,
        batch_size=256, 
        num_proc=1,
        drop_last_batch=False, 
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])
    corpus = None

    return dataset, corpus
