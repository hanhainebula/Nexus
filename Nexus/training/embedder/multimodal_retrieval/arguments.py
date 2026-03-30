import os
from dataclasses import dataclass, field
from typing import Optional

from Nexus.abc.training.arguments import AbsModelArguments
from Nexus.abc.training.embedder import (
    AbsEmbedderDataArguments,
    AbsEmbedderModelArguments,
    AbsEmbedderTrainingArguments,
)


@dataclass
class MultimodalEmbedderModelArguments(AbsEmbedderModelArguments):
    processor_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional processor path. Defaults to model_name_or_path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for model and processor."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading Hugging Face objects."},
    )
    token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN", None),
        metadata={"help": "HF token used for gated models."},
    )
    model_type: str = field(
        default="auto",
        metadata={"help": "Explicit model type override. Default uses the HF config."},
    )
    backbone_load_strategy: str = field(
        default="auto",
        metadata={
            "help": "Backbone loading strategy. `auto` preserves the current path, while "
            "`prefer_base_model` tries base Model/AutoModel first for direct `last_hidden_state` pooling.",
            "choices": ["auto", "prefer_base_model"],
        },
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype string for loading the backbone. Use `auto` to defer."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation passed to the backbone if supported."},
    )
    use_chat_template: bool = field(
        default=True,
        metadata={"help": "Use processor chat template for VLM backbones that support it."},
    )
    processor_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON object passed to AutoProcessor.from_pretrained. "
            "Config-file mode may provide either a JSON string or a JSON object.",
        },
    )
    processor_call_kwargs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON object merged into each processor(...) call. "
            "Useful for `size`, `min_pixels`, or `max_pixels` exploration.",
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Apply LoRA adapters on top of the backbone."},
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target modules for LoRA."},
    )


@dataclass
class MultimodalEmbedderDataArguments(AbsEmbedderDataArguments):
    media_root: Optional[str] = field(
        default=None,
        metadata={"help": "Base directory used to resolve relative media paths."},
    )
    train_group_size: int = field(default=8)
    query_max_len: int = field(default=512)
    passage_max_len: int = field(default=1024)
    max_example_num_per_dataset: int = field(
        default=100000000,
        metadata={"help": "Maximum number of examples loaded from each dataset file."},
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None,
        metadata={"help": "Instruction prepended to query text."},
    )
    query_instruction_format: str = field(
        default="{}{}",
        metadata={"help": "Format string used to inject the query instruction."},
    )
    passage_instruction_for_retrieval: Optional[str] = field(
        default=None,
        metadata={"help": "Instruction prepended to target text."},
    )
    passage_instruction_format: str = field(
        default="{}{}",
        metadata={"help": "Format string used to inject the passage instruction."},
    )
    knowledge_distillation: bool = field(
        default=False,
        metadata={"help": "Use `pos_scores` and `neg_scores` when present."},
    )
    shuffle_ratio: float = field(
        default=0.0,
        metadata={"help": "Randomly shuffle long text spans to regularize the text field."},
    )


@dataclass
class MultimodalEmbedderTrainingArguments(AbsEmbedderTrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "Share negatives across devices."})
    temperature: Optional[float] = field(default=0.02, metadata={"help": "Softmax temperature."})
    sentence_pooling_method: str = field(
        default="last_token",
        metadata={"help": "Pooling method.", "choices": ["cls", "mean", "last_token"]},
    )
    normalize_embeddings: bool = field(default=True, metadata={"help": "L2-normalize embeddings."})
    sub_batch_size: Optional[int] = field(default=None, metadata={"help": "Optional embedding sub-batch size."})
    kd_loss_type: str = field(
        default="kl_div",
        metadata={"help": "Knowledge distillation loss.", "choices": ["kl_div", "m3_kd_loss"]},
    )


@dataclass
class WrappedMultimodalEmbedderModelArguments(AbsModelArguments):
    negatives_cross_device: bool = field(default=False)
    temperature: float = field(default=1.0)
    sub_batch_size: int = field(default=-1)
    kd_loss_type: str = field(default="kl_div")
    sentence_pooling_method: str = field(default="last_token")
    normalize_embeddings: bool = field(default=False)
    query_max_len: int = field(default=512)
    passage_max_len: int = field(default=1024)
    model_type: str = field(default="auto")
    use_chat_template: bool = field(default=True)
    processor_call_kwargs: Optional[str] = field(default=None)
