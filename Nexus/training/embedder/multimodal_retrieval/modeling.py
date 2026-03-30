import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor

from Nexus.abc.training.embedder import AbsEmbedderModel, EmbedderOutput
from Nexus.modules.loss import CrossEntropyLoss, KLDivLoss, M3KDLoss
from Nexus.modules.multimodal import (
    annotate_multimodal_backbone,
    build_multimodal_forward_kwargs,
    extract_multimodal_hidden_states,
    infer_multimodal_model_type,
    infer_multimodal_output_mode,
    MultimodalProcessorAdapter,
    load_multimodal_backbone,
    load_multimodal_processor,
    move_batch_to_device,
)
from Nexus.modules.score import IP_text_retrieval

from .arguments import MultimodalEmbedderModelArguments, WrappedMultimodalEmbedderModelArguments

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEmbedderOutput(EmbedderOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


def _cast_trainable_params_to_fp32(model):
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype in {torch.float16, torch.bfloat16}:
            parameter.data = parameter.data.float()
    return model


class BiMultimodalEmbedderModel(AbsEmbedderModel):
    def __init__(
        self,
        model,
        processor,
        model_args: WrappedMultimodalEmbedderModelArguments,
        loss_function=None,
        score_function=None,
        *args,
        **kwargs,
    ):
        self.kd_loss_type = model_args.kd_loss_type
        super().__init__(*args, **kwargs)

        if loss_function is not None:
            self.loss_function = loss_function
        if score_function is not None:
            self.score_function = score_function

        self.model = model
        self.processor = processor
        self.processor_adapter = MultimodalProcessorAdapter(
            processor=processor,
            model_type=model_args.model_type,
            use_chat_template=model_args.use_chat_template,
            processor_call_kwargs=model_args.processor_call_kwargs,
        )
        self.temperature = model_args.temperature
        self.negatives_cross_device = model_args.negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Distributed training has not been initialized for cross-device negatives.")
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.sub_batch_size = model_args.sub_batch_size
        self.sentence_pooling_method = model_args.sentence_pooling_method
        self.normalize_embeddings = model_args.normalize_embeddings
        self.query_max_len = model_args.query_max_len
        self.passage_max_len = model_args.passage_max_len

    @classmethod
    def build(
        cls,
        model_args: MultimodalEmbedderModelArguments,
        wrapped_model_args: WrappedMultimodalEmbedderModelArguments,
        *args,
        **kwargs,
    ):
        processor = load_multimodal_processor(
            model_name_or_path=model_args.model_name_or_path,
            processor_name_or_path=model_args.processor_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            token=model_args.token,
            processor_kwargs=model_args.processor_kwargs,
            model_type=model_args.model_type,
        )
        model, config = load_multimodal_backbone(
            model_name_or_path=model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            token=model_args.token,
            model_type=model_args.model_type,
            torch_dtype_name=model_args.torch_dtype,
            attn_implementation=model_args.attn_implementation,
            peft_is_trainable=True,
            backbone_load_strategy=model_args.backbone_load_strategy,
        )
        effective_model_type = (
            infer_multimodal_model_type(config)
            if model_args.model_type in [None, "", "auto"]
            else model_args.model_type
        )
        wrapped_model_args.model_type = effective_model_type

        if model_args.use_lora and hasattr(model, "peft_config"):
            logger.warning(
                "The provided model path already resolves to a PEFT adapter. "
                "Skipping an extra get_peft_model() wrap."
            )
        elif model_args.use_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules.split(","),
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            model = annotate_multimodal_backbone(
                model,
                output_mode=infer_multimodal_output_mode(model),
            )

        if hasattr(model, "peft_config"):
            # Keep trainable adapter weights in fp32 for stable LoRA optimization.
            model = _cast_trainable_params_to_fp32(model)

        return cls(model=model, processor=processor, model_args=wrapped_model_args, *args, **kwargs)

    def init_modules(self):
        super().init_modules()
        self.distill_loss = self.get_distill_loss(self.kd_loss_type)

    def get_distill_loss(self, kd_loss_type):
        if kd_loss_type == "kl_div":
            return KLDivLoss()
        if kd_loss_type == "m3_kd_loss":
            return M3KDLoss()
        raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def get_loss_function(self):
        return CrossEntropyLoss()

    def get_score_function(self):
        return IP_text_retrieval()

    def _pool_hidden_states(self, hidden_states, attention_mask):
        if self.sentence_pooling_method == "cls":
            reps = hidden_states[:, 0]
        elif self.sentence_pooling_method == "mean":
            numerator = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
            denominator = attention_mask.sum(dim=1, keepdim=True).float()
            reps = numerator / denominator
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                reps = hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                reps = hidden_states[
                    torch.arange(hidden_states.shape[0], device=hidden_states.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"Unsupported pooling method: {self.sentence_pooling_method}")

        if self.normalize_embeddings:
            reps = torch.nn.functional.normalize(reps, dim=-1)
        return reps.contiguous()

    def _encode_batch(self, features, max_length: int):
        model_inputs = self.processor_adapter.encode_batch(features, max_length=max_length)
        model_device = next(self.model.parameters()).device
        model_inputs = move_batch_to_device(model_inputs, model_device)
        outputs = self.model(**model_inputs, **build_multimodal_forward_kwargs(self.model))
        hidden_states = extract_multimodal_hidden_states(outputs)

        return self._pool_hidden_states(hidden_states, model_inputs["attention_mask"])

    def encode(self, features, max_length: int):
        if features is None:
            return None

        if not isinstance(features, list):
            features = [features]

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            return self._encode_batch(features, max_length=max_length)

        all_reps = []
        for start in range(0, len(features), self.sub_batch_size):
            end = min(start + self.sub_batch_size, len(features))
            all_reps.append(self._encode_batch(features[start:end], max_length=max_length))
        return torch.cat(all_reps, dim=0).contiguous()

    def encode_query(self, features):
        return self.encode(features, max_length=self.query_max_len)

    def encode_info(self, features):
        return self.encode(features, max_length=self.passage_max_len)

    def compute_score(self, q_reps, p_reps):
        scores = self.score_function(q_reps, p_reps) / self.temperature
        return scores.view(q_reps.size(0), -1)

    def get_local_score(self, q_reps, p_reps, all_scores):
        group_size = p_reps.size(0) // q_reps.size(0)
        indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
        specific_scores = []
        for i in range(group_size):
            specific_scores.append(all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i])
        return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)

    def compute_local_score(self, q_reps, p_reps, compute_score_func=None, **kwargs):
        all_scores = self.compute_score(q_reps, p_reps) if compute_score_func is None else compute_score_func(q_reps, p_reps, **kwargs)
        return self.get_local_score(q_reps, p_reps, all_scores)

    def _compute_no_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        group_size = p_reps.size(0) // q_reps.size(0)
        local_scores = self.compute_local_score(q_reps, p_reps, compute_score_func, **kwargs)

        if teacher_targets is not None:
            loss = self.distill_loss(teacher_targets, local_scores, group_size=group_size)
            if self.kd_loss_type == "kl_div":
                local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
                loss += self.loss_function(local_scores, local_targets)
        else:
            local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
            loss = self.loss_function(local_scores, local_targets)
        return local_scores, loss

    def _compute_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        group_size = p_reps.size(0) // q_reps.size(0)
        scores = self.compute_score(q_reps, p_reps) if compute_score_func is None else compute_score_func(q_reps, p_reps, **kwargs)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student_scores = self.get_local_score(q_reps, p_reps, scores)
                loss = self.distill_loss(teacher_targets, student_scores, group_size=group_size)
                targets = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * group_size
                loss += self.loss_function(scores, targets)
            elif self.kd_loss_type == "m3_kd_loss":
                loss = self.distill_loss(teacher_targets, scores, group_size=group_size)
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            targets = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * group_size
            loss = self.loss_function(scores, targets)
        return scores, loss

    def _gather_tensor(self, tensor: Tensor):
        if tensor is None:
            return None
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        gathered[self.process_rank] = tensor
        return torch.cat(gathered, dim=0)

    def _compute_cross_device_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        group_size = p_reps.size(0) // q_reps.size(0)
        cross_q_reps = self._gather_tensor(q_reps)
        cross_p_reps = self._gather_tensor(p_reps)
        scores = self.compute_score(cross_q_reps, cross_p_reps) if compute_score_func is None else compute_score_func(cross_q_reps, cross_p_reps, **kwargs)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student_scores = self.get_local_score(cross_q_reps, cross_p_reps, scores)
                student_scores = student_scores[
                    q_reps.size(0) * self.process_rank : q_reps.size(0) * (self.process_rank + 1)
                ]
                loss = self.distill_loss(teacher_targets, student_scores, group_size=group_size)
                targets = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * group_size
                loss += self.loss_function(scores, targets)
            elif self.kd_loss_type == "m3_kd_loss":
                all_teacher_targets = self._gather_tensor(teacher_targets)
                loss = self.distill_loss(all_teacher_targets, scores, group_size=group_size)
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            targets = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) * group_size
            loss = self.loss_function(scores, targets)
        return scores, loss

    def compute_loss(self, batch, *args, **kwargs):
        queries, passages, teacher_scores, no_in_batch_neg_flag = batch
        q_reps = self.encode_query(queries)
        p_reps = self.encode_info(passages)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()
                teacher_targets = F.softmax(teacher_scores, dim=-1)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            elif self.negatives_cross_device:
                compute_loss_func = self._compute_cross_device_neg_loss
            else:
                compute_loss_func = self._compute_in_batch_neg_loss

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
        else:
            scores = self.compute_score(q_reps, p_reps)
            loss = None

        return MultimodalEmbedderOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            scores=scores,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({key: value.clone().cpu() for key, value in state_dict.items()})
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            raise NotImplementedError(f"Model {self.model.__class__.__name__} does not support save_pretrained.")
        if self.processor is not None and hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(output_dir)
