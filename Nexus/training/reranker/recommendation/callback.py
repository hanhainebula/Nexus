import os
from Nexus.abc.training.embedder import CallbackOutput
import json
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from .arguments import TrainingArguments


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_epoch_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_batch_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_eval_end(self, epoch, step, logs={}, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()

    def on_train_end(self, *args, **kwargs) -> CallbackOutput:
        return CallbackOutput()
    
    
class LoggerCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs = None,
        **kwargs
    ):
        _ = logs.pop("total_flos", None)
        _ = logs.pop("epoch", None)
        if state.is_local_process_zero:
            print(f"Step {state.global_step}: {logs}")
            
            
class StopCallback(TrainerCallback):
    """
    Callback class to inspect the state of the training loop and take decision.
    """

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of an epoch.
        """
        current_epoch = state.epoch
        
        if current_epoch >= args.num_train_epochs - 1:
            control.should_training_stop = True
            control.should_save = True
            
        return control
    
class CheckpointCallback(Callback):
    def __init__(self, step_interval: int, checkpoint_dir: str, is_main_process, **kwargs):
        """ CheckpointCallback, saves model checkpoints at a given step interval.

        Args:
            step_interval (int): Interval at which to save checkpoints.
            checkpoint_dir (str): Directory to save checkpoints in.
            is_main_process (bool): Whether the current process is the main process or not.
        """
        super().__init__(**kwargs)
        self.step_interval = step_interval
        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_step = 0
        self.is_main_process = is_main_process

    
    def on_batch_end(self, epoch, step, logs=..., *args, **kwargs) -> CallbackOutput:
        output = CallbackOutput()
        if step > 0 and self.step_interval is not None:
            if (step - self.last_checkpoint_step) % self.step_interval == 0:
                # self.save_checkpoint(step, item_loader=kwargs.get('item_loader', None))
                self.last_checkpoint_step = step
                output.save_checkpoint = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        return output

    def on_epoch_end(self, epoch, step, item_loader=None, *args, **kwargs) -> CallbackOutput:
        output = CallbackOutput()
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint-{step}-epoch-{epoch}")
        output.save_checkpoint = checkpoint_dir
        return output
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        # self.model.save(checkpoint_dir, item_loader=item_loader)
        # print(f"Save checkpoint at epoch {epoch} into directory {checkpoint_dir}")

        
    def save_checkpoint(self, step: int, item_loader=None):
        checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model.save(checkpoint_dir, item_loader=item_loader)
        print(f"Save checkpoint at step {step} into directory {checkpoint_dir}")
        

class EarlyStopCallback(Callback):
    def __init__(
            self,
            monitor_metric: str,
            strategy: str="epoch",
            patience: int=10,
            maximum: bool=True,
            save: bool=False,
            checkpoint_dir: str=None,
            logger=None,
            is_main_process: bool=False,
            **kwargs
        ):
        """ EarlyStopping callback.
        Args:
            monitor_metric: Metric to be monitored during training.
            strategy: Strategy to use for early stopping. Can be "epoch" or "step".
            patience: Number of epochs/steps without improvement after which the training is stopped.
            maximum: Whether to stop when the metric increases or decreases. 
                If True, the metric is considered to be increasing.
            save: Whether to save the best model.
            logger: Logger object used for logging messages.
        """
        super().__init__(**kwargs)
        assert strategy in ["epoch", "step"], "Strategy must be either 'epoch' or 'step'."
        self.monitor_metric = monitor_metric
        self.strategy = strategy
        self.patience = patience
        self.best_val_metric = 0 if maximum else float("inf")
        self.waiting = 0
        self.maximum = maximum
        self.logger = logger
        self.save = save
        if save:
            # assert model is not None, "Model must be provided if save is True."
            assert checkpoint_dir is not None, "Checkpoint directory must be provided if save is True."
        self.checkpoint_dir = checkpoint_dir
        self._last_epoch = 0
        self._last_step = 0
        self.is_main_process = is_main_process

    @property
    def state(self):
        state_d = {
            "best_epoch": self._last_epoch,
            "best_global_step": self._last_step,
            "best_metric": {self.monitor_metric: self.best_val_metric},
        }
        return state_d

    def on_eval_end(self, epoch, global_step, logs, *args, **kwargs) -> dict:
        """ Callback method called at the end of each evaluation step.
        Args:
            epoch: Current epoch number.
            global_step: Current step number within the current epoch.
            logs: Dictionary containing the metrics logged so far.
        Returns:
            dict: A dictionary containing the following keys:
                - "save_checkpoint": The path where the best model should be saved. If None, no model is saved.
                - "stop_training": A boolean indicating whether to stop training.
        """
        val_metric = logs[self.monitor_metric]
        output = CallbackOutput()
        if self.maximum:
            if val_metric < self.best_val_metric:
                self.waiting += (epoch - self._last_epoch) if self.strategy == "epoch" else (global_step-self._last_step)
            else:
                self.best_val_metric = val_metric
                self.waiting = 0
                self._last_epoch = epoch
                self._last_step = global_step
                self.save_state()
                if self.save:
                    output.save_checkpoint = os.path.join(self.checkpoint_dir, "best_ckpt")
        else:
            if val_metric > self.best_val_metric:
                self.waiting += (epoch - self._last_epoch) if self.strategy == "epoch" else (global_step-self._last_step)
            else:
                self.best_val_metric = val_metric
                self.waiting = 0
                self._last_epoch = epoch
                self._last_step = global_step
                self.save_state()
                if self.save:
                    output.save_checkpoint = os.path.join(self.checkpoint_dir, "best_ckpt")

        if self.waiting >= self.patience:
            if self.logger is not None:
                self.logger.info("Early stopping at epoch {}, global step {}".format(epoch, global_step))
            output.stop_training = True
        else:
            if self.logger is not None:
                self.logger.info("Waiting for {} more {}s".format(self.patience - self.waiting, self.strategy))
            output.stop_training = False
        
        return output


    def save_state(self, *args, **kwargs):
        """ Save the best model. """
        if self.save and self.is_main_process:
            checkpoint_dir = self.checkpoint_dir
            best_ckpt_dir = os.path.join(checkpoint_dir, "best_ckpt")
            if not os.path.exists(best_ckpt_dir):
                os.makedirs(best_ckpt_dir)
            state = self.state

            with open(os.path.join(best_ckpt_dir, "state.json"), "w") as f:
                json.dump(state, f)
            
            # self.model.save(best_ckpt_dir)
            
            print(f"Best model saved in {best_ckpt_dir}.")