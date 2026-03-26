import logging
from typing import Any, List, Optional, Union

import torch
from tqdm import tqdm

from Nexus.abc.inference import AbsEmbedder
from Nexus.modules.multimodal import (
    MultimodalProcessorAdapter,
    apply_instruction,
    infer_multimodal_model_type,
    load_multimodal_backbone,
    load_multimodal_processor,
    move_batch_to_device,
    normalize_multimodal_item,
)

logger = logging.getLogger(__name__)


class MultimodalEmbedder(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "last_token"

    def __init__(
        self,
        model_name_or_path: str,
        processor_name_or_path: Optional[str] = None,
        model_type: str = "auto",
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}",
        passage_instruction_for_retrieval: Optional[str] = None,
        passage_instruction_format: str = "{}{}",
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: str = "last_token",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        batch_size: int = 8,
        query_max_length: int = 512,
        passage_max_length: int = 1024,
        convert_to_numpy: bool = True,
        use_chat_template: bool = True,
        torch_dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval
        self.passage_instruction_format = passage_instruction_format
        self.target_devices = self.get_target_devices(devices)
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.convert_to_numpy = convert_to_numpy
        self.pooling_method = pooling_method
        self.pool = None

        self.processor = load_multimodal_processor(
            model_name_or_path=model_name_or_path,
            processor_name_or_path=processor_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        self.model, config = load_multimodal_backbone(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            token=token,
            model_type=model_type,
            torch_dtype_name=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self.model_type = infer_multimodal_model_type(config) if model_type in [None, "", "auto"] else model_type
        self.processor_adapter = MultimodalProcessorAdapter(
            processor=self.processor,
            model_type=self.model_type,
            use_chat_template=use_chat_template,
        )

    def _get_model_device(self):
        return next(self.model.parameters()).device

    def _pool_hidden_states(self, hidden_states, attention_mask):
        if self.pooling_method == "cls":
            reps = hidden_states[:, 0]
        elif self.pooling_method == "mean":
            numerator = torch.sum(hidden_states * attention_mask.unsqueeze(-1).float(), dim=1)
            denominator = attention_mask.sum(dim=1, keepdim=True).float()
            reps = numerator / denominator
        elif self.pooling_method == "last_token":
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
            raise NotImplementedError(f"Unsupported pooling method: {self.pooling_method}")

        if self.normalize_embeddings:
            reps = torch.nn.functional.normalize(reps, dim=-1)
        return reps

    def _encode_batch(
        self,
        items,
        max_length: int,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
    ):
        normalized_items = []
        for item in items:
            normalized_item = normalize_multimodal_item(item)
            normalized_item["text"] = apply_instruction(
                normalized_item.get("text", ""),
                instruction=instruction,
                instruction_format=instruction_format,
            )
            normalized_items.append(normalized_item)

        model_inputs = self.processor_adapter.encode_batch(
            normalized_items,
            max_length=max_length,
            instruction=None,
            instruction_format=instruction_format,
        )
        model_inputs = move_batch_to_device(model_inputs, self._get_model_device())

        outputs = self.model(**model_inputs, return_dict=True, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = hidden_states[-1]
        return self._pool_hidden_states(hidden_states, model_inputs["attention_mask"])

    def encode_queries(
        self,
        queries,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any,
    ):
        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length or self.query_max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=self.query_instruction_for_retrieval,
            instruction_format=self.query_instruction_format,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any,
    ):
        return self.encode(
            corpus,
            batch_size=batch_size,
            max_length=max_length or self.passage_max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=self.passage_instruction_for_retrieval,
            instruction_format=self.passage_instruction_format,
            **kwargs,
        )

    def encode_query(self, *args, **kwargs):
        return self.encode_queries(*args, **kwargs)

    def encode_info(self, *args, **kwargs):
        return self.encode_corpus(*args, **kwargs)

    def encode(
        self,
        inputs,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
        **kwargs: Any,
    ):
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.passage_max_length
        if convert_to_numpy is None:
            convert_to_numpy = self.convert_to_numpy

        single_input = not isinstance(inputs, list)
        if single_input:
            inputs = [inputs]

        if single_input or len(self.target_devices) == 1:
            embeddings = self.encode_single_device(
                inputs,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
                instruction=instruction,
                instruction_format=instruction_format,
                **kwargs,
            )
        else:
            if self.pool is None:
                self.pool = self.start_multi_process_pool(AbsEmbedder._encode_multi_process_worker)
            embeddings = self.encode_multi_process(
                inputs,
                self.pool,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                instruction=instruction,
                instruction_format=instruction_format,
                **kwargs,
            )
        return embeddings[0] if single_input else embeddings

    def compute_score(self, queries, passages, **kwargs: Any):
        q_embeddings = self.encode_queries(queries, convert_to_numpy=False, **kwargs)
        p_embeddings = self.encode_corpus(passages, convert_to_numpy=False, **kwargs)
        return q_embeddings @ p_embeddings.T

    @torch.no_grad()
    def encode_single_device(
        self,
        inputs,
        batch_size: int = 8,
        max_length: int = 1024,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
        **kwargs: Any,
    ):
        self.model.eval()
        self.model.to(device)
        if self.use_fp16 and device != "cpu":
            self.model.half()

        encoded_batches = []
        for start in tqdm(range(0, len(inputs), batch_size), desc="Encoding", disable=len(inputs) <= batch_size):
            end = min(start + batch_size, len(inputs))
            reps = self._encode_batch(
                inputs[start:end],
                max_length=max_length,
                instruction=instruction,
                instruction_format=instruction_format,
            )
            encoded_batches.append(reps.cpu())

        embeddings = torch.cat(encoded_batches, dim=0)
        if convert_to_numpy:
            if embeddings.dtype in {torch.bfloat16, torch.float16}:
                embeddings = embeddings.float()
            return embeddings.numpy()
        return embeddings
