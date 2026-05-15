from typing import Dict, Optional
import json
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoConfig

from .nexus_multimodal_compat import NexusMultimodalCompatEmbedder


class MMEBEmbeddingModel(nn.Module):
    """Simplified MMEBModel for Qwen3VL embeddings."""

    def __init__(self,
                 encoder: NexusMultimodalCompatEmbedder,
                 normalize: bool = True,
                 temperature: float = 0.02):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        # DDP setup
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        return self.encoder.model.device

    @property
    def config(self):
        return self.encoder.model.config

    @classmethod
    def load(cls,
            model_name_or_path: str,
            normalize: bool = True,
            temperature: float = 0.02,
            instruction: Optional[str] = None,
            **kwargs) -> "MMEBEmbeddingModel":
            
        max_frames = kwargs.pop("max_frames", 32)
        fps = kwargs.pop("fps", 1.0)
        default_instruction = kwargs.pop("default_instruction", instruction)
        config_name_or_path = model_name_or_path
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        is_peft_adapter = os.path.isdir(model_name_or_path) and os.path.exists(adapter_config_path)
        if is_peft_adapter:
            try:
                from peft import PeftConfig

                peft_config = PeftConfig.from_pretrained(model_name_or_path)
                config_name_or_path = peft_config.base_model_name_or_path or config_name_or_path
            except Exception:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                config_name_or_path = adapter_config.get("base_model_name_or_path") or config_name_or_path

        config = AutoConfig.from_pretrained(config_name_or_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "qwen3_vl") or "qwen3_vl"
        backbone_load_strategy = "auto" if is_peft_adapter else "prefer_base_model"
        
        encoder = NexusMultimodalCompatEmbedder(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            query_instruction_for_retrieval=default_instruction,
            passage_instruction_for_retrieval=default_instruction,
            use_chat_template=True,
            pooling_method="last_token",
            backbone_load_strategy=backbone_load_strategy,
            query_max_length=8192,
            passage_max_length=8192,
            # 还原最干净的 processor 配置
            processor_kwargs={
                "padding_side": "right",
                "min_pixels": 4096,
                "max_pixels": 1843200,
            },
            **kwargs,
        )

        if is_peft_adapter:
            # PEFT adapters trained on conditional-generation wrappers have keys such as
            # `base_model.model.model.language_model...`.  Load them on the full wrapper
            # first so the keys match, then evaluate on the wrapped backbone to avoid the
            # lm_head logits allocation.
            peft_wrapper = getattr(encoder, "model", None)
            lora_base = getattr(peft_wrapper, "base_model", None)
            conditional_model = getattr(lora_base, "model", None)
            backbone_model = getattr(conditional_model, "model", None)
            if backbone_model is not None and backbone_model is not conditional_model:
                from Nexus.modules.multimodal import (
                    OUTPUT_MODE_LAST_HIDDEN_STATE,
                    annotate_multimodal_backbone,
                )

                encoder._peft_wrapper_for_backbone = peft_wrapper
                encoder.model = annotate_multimodal_backbone(
                    backbone_model,
                    output_mode=OUTPUT_MODE_LAST_HIDDEN_STATE,
                    backbone_load_strategy="prefer_base_model",
                    loader_kind="conditional_wrapper_base_model_with_peft",
                )
        
        # 将默认参数保存在实例上，供后续调用时参考
        model = cls(encoder=encoder, normalize=normalize, temperature=temperature)
        model.default_fps = fps
        model.default_max_frames = max_frames
        model.encoder.default_fps = fps
        model.encoder.default_max_frames = max_frames
        
        return model

    def save(self, output_dir: str):
        self.encoder.model.save_pretrained(output_dir)
        self.encoder.processor.save_pretrained(output_dir)

    def encode_input(self, inputs: Dict) -> Tensor:
        """Encode inputs using the Qwen3VL embedder.
        
        Args:
            inputs: Dict containing 'text', 'image', 'video', 'instruction' etc.
                    Can be a single dict or a list of dicts.
        """
        # 如果是预处理过的 tensor 输入，直接 forward
        if 'input_ids' in inputs:
            outputs = self.encoder.forward(inputs)
            hidden_state = outputs['last_hidden_state']
            attention_mask = outputs['attention_mask']
            pooled = self._pooling_last(hidden_state, attention_mask)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled
        
        # 否则使用 embedder 的 process 方法
        if isinstance(inputs, dict):
            inputs = [inputs]
        return self.encoder.process(inputs, normalize=self.normalize)

    def _pooling_last(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last non-padded token."""
        last_pos = attention_mask.flip(dims=[1]).argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def forward(self,
                qry: Dict[str, Tensor] = None,
                tgt: Dict[str, Tensor] = None) -> Dict:
        """Forward pass for contrastive learning / evaluation."""
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry = self._dist_gather(qry_reps)
            all_tgt = self._dist_gather(tgt_reps)
        else:
            all_qry, all_tgt = qry_reps, tgt_reps

        scores = torch.matmul(all_qry, all_tgt.T) / self.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        target = target * (all_qry.size(0) // all_tgt.size(0))
        loss = self.cross_entropy(scores, target)

        if self.is_ddp:
            loss = loss * self.world_size
        return {"loss": loss, "qry_reps": qry_reps, "tgt_reps": tgt_reps}

    def _dist_gather(self, t: Tensor) -> Tensor:
        """Gather tensors across distributed processes."""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute similarity matrix between query and passage representations."""
        return torch.matmul(q_reps, p_reps.T)


if __name__ == '__main__':
    model = MMEBEmbeddingModel.load(
        model_name_or_path=r'Your model path',
        attn_implementation='flash_attention_2',
        torch_dtype="bfloat16", device_map='cuda'
    )

    inputs = {
        'inputs': [
            {
                'text': "a woman breaks an egg",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'text': "a woman breaks two eggs in a bowl",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'image': r'/path/to/example_0.jpeg',
            },
            {
                'image': r'/path/to/example_1.jpg',
            },
        ],
    }

    embeddings = model.encode_input(inputs["inputs"])

    print(
        f'Embeddings:\n{embeddings[:, :10].tolist()}\n{embeddings[:, -10:].tolist()}\n'
        f'Score:\n{model.compute_similarity(embeddings, embeddings).tolist()}\n'
    )
