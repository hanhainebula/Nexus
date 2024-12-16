import logging

import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
import torch.nn.functional as F
from UniRetrieval.training.abc.embedder import AbsEmbedderModel, EmbedderOutput
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
logger = logging.getLogger(__name__)


@dataclass
class TextEmbedderOutput(EmbedderOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class BiEncoderOnlyEmbedderModel(AbsEmbedderModel):
    """Embedder class for encoder only model.

    Args:
        base_model (AutoModel): The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``"kl_div"``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'cls'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
    """
    TRANSFORMER_CLS = AutoModel
    
    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = False,
    ):
        super.__init__()
        self.model = base_model
        self.tokenizer = tokenizer

        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.sub_batch_size = sub_batch_size
        self.kd_loss_type = kd_loss_type

        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')

    def encode(self, features):
        """Encode and get the embedding.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_p_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]
                    last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                    p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
                all_p_reps = self._sentence_embedding(last_hidden_state, features['attention_mask'])
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        """Use the pooling method to get the sentence embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The sentence embeddings.
        """
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

    def compute_score(self, q_reps, p_reps):
        """Computes the scores between query and passage representations.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed scores, adjusted by temperature.
        """
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _compute_similarity(self, q_reps, p_reps):
        """Computes the similarity between query and passage representations using inner product.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed similarity matrix.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_loss(self, scores, target):
        """Compute the loss using cross entropy.

        Args:
            scores (torch.Tensor): Computed score.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        """Save the model to the directory.

        Args:
            output_dir (str): Directory for saving the model.
        """
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


    def get_local_score(self, q_reps, p_reps, all_scores):
        """Get the local score of queries and passages.

        Args:
            q_reps (torch.Tensor): Queries representations.
            p_reps (torch.Tensor): Passages rerpresentations.
            all_scores (torch.Tensor): All the query-passage scores computed.

        Returns:
            torch.Tensor: Local scores to compute loss.
        """
        group_size = p_reps.size(0) // q_reps.size(0)
        indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
        specific_scores = []
        for i in range(group_size):
            specific_scores.append(
                all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
            )
        return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)

    def compute_local_score(self, q_reps, p_reps, compute_score_func=None, **kwargs):
        """Compute the local score of queries and passages.

        Args:
            q_reps (torch.Tensor): Queries representations.
            p_reps (torch.Tensor): Passages rerpresentations.
            compute_score_func (function, optional): Function to compute score. Defaults to ``None``, which will use the
                :meth:`self.compute_score`.

        Returns:
            torch.Tensor: Local scores to compute loss.
        """
        if compute_score_func is None:
            all_scores = self.compute_score(q_reps, p_reps)
        else:
            all_scores = compute_score_func(q_reps, p_reps, **kwargs)
        loacl_scores = self.get_local_score(q_reps, p_reps, all_scores)
        return loacl_scores

    def _compute_no_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when using no in-batch negatives and no cross-device negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        local_scores = self.compute_local_score(q_reps, p_reps, compute_score_func, **kwargs)   # (batch_size, group_size)

        if teacher_targets is not None:
            # compute kd loss
            loss = self.distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=group_size)

            # add normal loss if needed
            if self.kd_loss_type == "kl_div":
                local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long) # (batch_size)
                loss += self.compute_loss(local_scores, local_targets)
        else:
            local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long) # (batch_size)
            loss = self.compute_loss(local_scores, local_targets)

        return local_scores, loss

    def _compute_in_batch_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when only using in-batch negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        if compute_score_func is None:
            scores = self.compute_score(q_reps, p_reps) # (batch_size, batch_size * group_size)
        else:
            scores = compute_score_func(q_reps, p_reps, **kwargs)   # (batch_size, batch_size * group_size)

        if teacher_targets is not None:
            # compute kd loss
            if self.kd_loss_type == "kl_div":
                student_scores = self.get_local_score(q_reps, p_reps, scores) # (batch_size, group_size)

                loss = self.distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)

                idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                targets = idxs * (p_reps.size(0) // q_reps.size(0)) # (batch_size)
                loss += self.compute_loss(scores, targets)
            elif self.kd_loss_type == "m3_kd_loss":
                loss = self.distill_loss(self.kd_loss_type, teacher_targets, scores, group_size)
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
            targets = idxs * group_size # (batch_size)
            loss = self.compute_loss(scores, targets)

        return scores, loss

    def _compute_cross_device_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when using both in-batch negatives and cross-device negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        cross_q_reps = self._dist_gather_tensor(q_reps) # (world_size * batch_size, dim)
        cross_p_reps = self._dist_gather_tensor(p_reps) # (world_size * batch_size * group_size, dim)

        if compute_score_func is None:
            cross_scores = self.compute_score(cross_q_reps, cross_p_reps)   # (world_size * batch_size, world_size * batch_size * group_size)
        else:
            cross_scores = compute_score_func(cross_q_reps, cross_p_reps, **kwargs) # (world_size * batch_size, world_size * batch_size * group_size)

        if teacher_targets is not None:
            # compute kd loss
            if self.kd_loss_type == "kl_div":
                student_scores = self.get_local_score(cross_q_reps, cross_p_reps, cross_scores) # (world_size * batch_size, group_size)
                student_scores = student_scores[
                    q_reps.size(0)*self.process_rank : q_reps.size(0)*(self.process_rank+1)
                ]   # (batch_size, group_size)

                loss = self.distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)

                cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
                cross_targets = cross_idxs * group_size # (world_size * batch_size)
                loss += self.compute_loss(cross_scores, cross_targets)
            elif self.kd_loss_type == "m3_kd_loss":
                cross_teacher_targets = self._dist_gather_tensor(teacher_targets)   # (world_size * batch_size, group_size)

                loss = self.distill_loss(self.kd_loss_type, cross_teacher_targets, cross_scores, group_size)
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
            cross_targets = cross_idxs * group_size # (world_size * batch_size)
            loss = self.compute_loss(cross_scores, cross_targets)

        return cross_scores, loss


    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            TextEmbedderOutput: Output of the forward call of model.
        """
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()   # (batch_size, group_size)
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
        else:
            loss = None

        return TextEmbedderOutput(
            loss=loss,
        )

    @staticmethod
    def distill_loss(kd_loss_type, teacher_targets, student_scores, group_size=None):
        """Compute the distillation loss.

        Args:
            kd_loss_type (str): Type of knowledge distillation loss, supports "kl_div" and "m3_kd_loss".
            teacher_targets (torch.Tensor): Targets from the teacher model.
            student_scores (torch.Tensor): Score of student model.
            group_size (int, optional): Number of groups for . Defaults to ``None``.

        Raises:
            ValueError: Invalid kd_loss_type

        Returns:
            torch.Tensor: A scalar of computed distillation loss.
        """
        if kd_loss_type == 'kl_div':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, group_size) / (world_size * batch_size, group_size)
            return - torch.mean(
                torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1)
            )
        elif kd_loss_type == 'm3_kd_loss':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, batch_size * group_size) / (world_size * batch_size, world_size * batch_size * group_size)
            labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
            labels = labels * group_size

            loss = 0
            mask = torch.zeros_like(student_scores)
            for i in range(group_size):
                temp_target = labels + i
                temp_scores = student_scores + mask
                temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")  # B
                loss += torch.mean(teacher_targets[:, i] * temp_loss)
                mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                    value=torch.finfo(student_scores.dtype).min)
            return loss
        else:
            raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        """Gather a tensor from all processes in a distributed setting.

        Args:
            t (Optional[torch.Tensor]): The input tensor to be gathered. If `None`, no gathering is performed.

        Returns:
            Union[torch.Tensor, None]: A concatenated tensor from all processes if ``t`` is not ``None``, 
                otherwise returns ``None``.
        """
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    # TODO 此处考虑是否把encode名字改了
    def encode_query(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
    
    def encode_info(self, *args, **kwargs):
        return self.encode(*args, **kwargs)