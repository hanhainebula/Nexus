import os
from typing import Optional

import torch

from Nexus.abc.training.embedder import AbsEmbedderTrainer


class MultimodalEmbedderTrainer(AbsEmbedderTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not hasattr(self.model, "save"):
            raise NotImplementedError(f"MODEL {self.model.__class__.__name__} does not support save interface")

        self.model.save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(batch=inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

