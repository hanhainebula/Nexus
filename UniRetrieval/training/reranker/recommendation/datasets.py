from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
from dataclasses import dataclass
from typing import List, Union
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
from multiprocessing import Process, Queue, Pool
from accelerate import Accelerator
import pandas as pd
from torch.utils.data import IterableDataset, Dataset
from copy import deepcopy
import re
import fsspec

from UniRetrieval.abc.training.embedder import AbsCallback, CallbackOutput
from UniRetrieval.training.embedder.recommendation.arguments import REQUIRED_DATA_CONFIG, DEFAULT_CONFIG


# callback添加在了datasets里面
class Callback(AbsCallback):
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
    
    
import os


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
            

def read_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_yaml(filepath: str):
    try:
        import yaml
    except ImportError:
        raise ImportError("Please install PyYAML first.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def timeformat_to_regex(timeformat: str):
    return re.compile(re.escape(timeformat).replace("%Y", r"\d{4}")\
                                 .replace("%m", r"\d{2}")\
                                 .replace("%d", r"\d{2}"))

def extract_timestamp(filename: str, format: str="%Y-%m-%d") -> datetime:
    """Extract timestamp from filename using regex according to the given format"""
    # get regex according to the given format
    regex = timeformat_to_regex(format)
    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"No date found in {filename}")
    return datetime.strptime(match.group(), format)

def time2datestr(dt: datetime, format: str="%Y-%m-%d") -> str:
    return dt.strftime(format)


def nested_dict_update(dict_1: dict, dict_2: dict):
    """Update a nested dictionary (dict_1) with another one (dict_2)
    
    Args:
        dict_1 (dict): The target dictionary
        dict_2 (dict): The source dictionary

    Returns:
        dict: The updated dictionary
    """
    res = deepcopy(dict_1)
    for key, value in dict_2.items():
        if key not in res:
            res[key] = {}
        if isinstance(value, dict) and (res[key] is not None):
            res[key] = nested_dict_update(res[key], value)
        else:
            res[key] = value
    return res


def df_to_tensor_dict(df):
    """Convert pandas dataframe into tensor dictionary"""
    return {col: torch.tensor(df[col].values) for col in df.columns}


def process_conditions(conditions: List[str]):
    """
    Process conditions list. From str such as "==3" to a lambda function.
    Supported operators: [==, !=, >, <, >=, <=]
    Args:
        conditions (List[str]): A list of conditions
    Returns:
        function: A lambda function covering all conditions
    """
    def parse_condition(condition_str):
        # Define a regular expression to match the operator and value in the condition string
        match = re.match(r'(==|!=|>=|<=|>|<)(\d+)', condition_str.strip().replace(" ", ""))
        if not match:
            raise ValueError(f"Unsupported condition format: {condition_str}")
        
        operator, value = match.groups()
        value = int(value)  # 将数值转换为整数
        
        # Return a lambda function based on the operator
        if operator == '==':
            return lambda x: x == value
        elif operator == '!=':
            return lambda x: x != value
        elif operator == '>=':
            return lambda x: x >= value
        elif operator == '<=':
            return lambda x: x <= value
        elif operator == '>':
            return lambda x: x > value
        elif operator == '<':
            return lambda x: x < value
        else:
            raise ValueError(f"Unsupported operator in condition: {operator}")

    # Convert each condition string to a lambda function and combine them with logical AND
    lambda_functions = [parse_condition(cond) for cond in conditions]
    return lambda x: all(func(x) for func in lambda_functions)


def detect_file_type(path: str) -> str:
    """Detect the file type from its extension"""
    file_extension = path.split('.')[-1]
    if file_extension == "parquet":
        return "parquet"
    elif file_extension == "feather":
        return "feather"
    elif file_extension in ["csv", "txt"]:
        return "csv"
    elif file_extension in ["pkl", "pickle"]:
        return "pkl"
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


class BaseClient(object):
    def __init__(self, url: str):
        self.url = url

    def load_file(self, path=None, **kwargs):
        if not path:
            path = self.url
        else:
            path = os.path.join(self.url, path)
        filetype = detect_file_type(path)
        if filetype == "parquet":
            df = pd.read_parquet(path, **kwargs)
        elif filetype == "csv":
            df = pd.read_csv(path, **kwargs)
        elif filetype == "feather":
            df = pd.read_feather(path, **kwargs)
        elif filetype == "pkl":
            df = pd.read_pickle(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")
        return df
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        return os.listdir(self.url)
    
class HDFSClient(BaseClient):
    def __init__(self, url: str):
        self._check_hdfs_connection(url)
        super(HDFSClient, self).__init__(url)
        
    @staticmethod
    def _check_hdfs_connection(hdfs_url: str) -> bool:
        """Check that we can connect to the HDFS filesystem at url"""
        if not isinstance(hdfs_url, str):
            raise TypeError(f"Expected `url` to be a string, got {type(hdfs_url)}")
        if not hdfs_url.startswith("hdfs://"):
            raise ValueError(f"Expected `url` to start with 'hdfs://', got {hdfs_url}")
        try:
            fs = fsspec.filesystem('hdfs', fs_kwargs={'hdfs_connect': hdfs_url})
        except ImportError:
            raise ImportError("`fsspec` is not installed")
        except Exception as e:
            print(e)
            raise ValueError(f"Could not connect to {hdfs_url}")
        return True
    
    def list_dir(self) -> list[str]:
        """List all files and directories in the given directory."""
        fs = fsspec.filesystem('hdfs')
        return [os.path.basename(file) for file in fs.ls(self.url)]
    
CLIENT_MAP = {
    'file': BaseClient,
    'hdfs': HDFSClient
}

def get_client(client_type: str, url: str):
    if client_type in CLIENT_MAP.keys():
        return CLIENT_MAP[client_type](url=url)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

class Statistics(object):
    @staticmethod
    def from_dict(d: dict):
        stat = Statistics()
        for k, v in d.items():
            setattr(stat, k.strip(), v)
        return stat

    def add_argument(self, name, value):
        setattr(self, name, value)

@dataclass
class DataAttr4Model:
    """
    Data attributes for a dataset. Serve for models
    """
    fiid: str
    flabels: List[str]
    features: List[str]
    context_features: List[str]
    item_features: List[str]
    seq_features: List[str]
    num_items: int  # number of candidate items instead of maximum id of items
    stats: Statistics

    @staticmethod
    def from_dict(d: dict):
        if "stats" in d:
            d["stats"] = Statistics.from_dict(d["stats"])
        attr = DataAttr4Model(**d)
        return attr

    def to_dict(self):
        d = self.__dict__
        for k, v in d.items():
            if type(v) == Statistics:
                d[k] = v.__dict__
        return d



class ItemDataset(Dataset):
    def __init__(self, item_feat_df: torch.Tensor):
        super().__init__()
        self.item_feat_df = item_feat_df
        self.item_ids = item_feat_df.index.to_numpy()
        self.item_id_2_offset = pd.DataFrame.from_dict(
            {item_id: i for i, item_id in enumerate(self.item_ids)},
            orient="index",
            columns=["offset"]
        )
        self.item_id_col = self.item_feat_df.index.name

    def __getitem__(self, index):
        feat_dict = self.item_feat_df.loc[self.item_ids[index]].to_dict()
        if self.item_id_col is not None:
            feat_dict.update({self.item_id_col: self.item_ids[index]})
        return feat_dict
    
    def __len__(self):
        return len(self.item_ids)
    

    def get_item_feat(self, item_ids: torch.LongTensor):
        # TODO: efficiency issue due to item_ids may be in GPU and the first dimension is batch size
        item_ids_list = item_ids.view(-1).tolist()
        item_ids_list = self.item_ids[item_ids_list]
        feats = self.item_feat_df.loc[item_ids_list].to_dict("list")
        feats = {k: torch.tensor(v).reshape(*item_ids.shape).to(item_ids.device) for k,v in feats.items()}
        return feats


class ConfigProcessor(object):
    """
    This class is used to split the data into train and test sets at file level.

    Args:
        config (Union[dict, str]): Configuration dictionary or path to JSON file
    """
    def __init__(self, config: Union[dict, str]):
        self.config = self._load_config(config)
        self._check_config()
        self.data_dir = self.config['url']
        self.train_start_date, self.train_end_date = self._get_date_range(mode='train')
        self.test_start_date, self.test_end_date = self._get_date_range(mode='test')
        self.train_files = self._get_valid_filenames(self.train_start_date, self.train_end_date)
        self.test_files = self._get_valid_filenames(self.test_start_date, self.test_end_date)
        self.attrs = self.get_attrs()


    def get_attrs(self):
        stats = Statistics.from_dict(self.config['stats'])
        def remove_whitespace(string):
            return string.strip()
        self.config['context_features'] = list(map(remove_whitespace, self.config['context_features']))
        self.config['item_features'] = list(map(remove_whitespace, self.config['item_features']))
        if self.config.get("user_sequential_info", None):
            self.config['seq_features'] = list(map(remove_whitespace, self.config['user_sequential_info']['use_cols']))
        else:
            self.config['seq_features'] = []
        return DataAttr4Model(
            fiid=self.config['item_col'],
            flabels=self.config['labels'],
            features=self.config['context_features']+self.config['item_features'],
            context_features=self.config['context_features'],
            item_features=self.config['item_features'],
            seq_features=self.config['seq_features'],
            num_items=getattr(stats, self.config['item_col']),
            stats=stats,
        )

    def _load_config(self, config: Union[dict, str]):
        """Loads the configuration from either a dictionary or a JSON file."""
        if isinstance(config, dict):
            pass
        elif isinstance(config, str):
            config = read_json(config)
        else:
            raise TypeError("Config must be a dictionary or path to JSON file.")
        # update nested config to default config
        config = nested_dict_update(DEFAULT_CONFIG, config)
        return config

    def _check_config(self):
        """Checks that all required keys are present in the configuration."""
        def _flatten_keys(dictionary):
            # flatten nested keys with /
            valid_keys = []
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    valid_keys.extend([f"{key}/{subkey}" for subkey in _flatten_keys(value)])
                else:
                    valid_keys.append(key)
            return valid_keys
        
        valid_keys = _flatten_keys(self.config)
        missing_keys = []
        for key in REQUIRED_DATA_CONFIG:
            if key not in valid_keys:
                missing_keys.append(key)
        # TODO: confirm this assert, it using key out of the loop
        assert key in valid_keys, f"Missing required keys: {missing_keys}"

    def _get_date_range(self, mode: str):
        """Get start and end date from config
        Args:
            mode (str): train or test
        """
        date_format: str = self.config['date_format']
        start_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['start_date'], date_format)
        end_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['end_date'], date_format)
        return start_date, end_date

    def _get_valid_filenames(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Get all used filenames among given date range in the dataset"""
        data_client = get_client(self.config['type'], self.config['url'])
        filenames = sorted(data_client.list_dir())
        dates = [extract_timestamp(filename) for filename in filenames]
        file_idx = [idx for idx, date in enumerate(dates) if start_date <= date < end_date]
        valid_filenames = [filenames[idx] for idx in file_idx]
        return valid_filenames
    

class DailyDataIterator(object):
    """This class is used to iterate over each day of data (dataframe)
    
    Args:
        config (dict): config dictionary
        filenames (List[str]): list of filenames to iterate over
    """
    def __init__(self, config, filenames):
        self.config = config
        self.data_client = get_client(config['type'], config['url'])
        if config['user_sequential_info']:
            # use given key column and sequence data file
            self.seq_index_col = config['user_sequential_info']['key']
            self.seq_data_client = get_client(config['type'], config['user_sequential_info']['url'])
        else:
            # use given key column, no additional files needed
            self.seq_index_col = None
            self.seq_data_client = None

        if config['item_info']:
            self.item_col = config['item_info']['key']
            self.item_data_client = get_client(config['type'], config['item_info']['url'])
        else:
            self.item_col = None
            self.item_data_client = None

        self.filenames = filenames
        self.cur_idx = 0
        self._data_cache = {'filename': None, 'data': None, 'seq': None}

    def _columns_rename(self, df: pd.DataFrame):
        """ Rename columns to remove whitespace and modify `features` in the config consistently"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            rename_map[col] = new_col
        self.config['features'] = self.config['context_features'] + self.config['item_features']
        return df.rename(columns=rename_map)

    def load_file(self, filename: str):
        """Load one day of data"""
        data = self.data_client.load_file(filename)
        data = self._columns_rename(data)
        # use given columns
        if self.config['features']:
            if self.seq_index_col is not None and self.seq_index_col not in self.config['features']:
                # add key column for merge
                keep_cols = [self.seq_index_col] + self.config['features'] + self.config['labels']
            else:
                keep_cols = self.config['features'] + self.config['labels']
            data = data[keep_cols]
        data = self.post_process(data)
        return data
    
    def load_sequence_file(self, filename: str):
        """Load one day of sequence data"""
        if self.seq_data_client is None:
            return None
        date_str = filename.split('.')[0]
        fileformat = self.config['user_sequential_info'].get('file_format', 'auto')
        if fileformat != 'auto':
            filename = f"{date_str}.{fileformat}"
        else:
            # auto mode: find the file with the same date
            all_files = self.seq_data_client.list_dir()
            for file in all_files:
                if date_str in file:
                    filename = file
                    break
        data = self.seq_data_client.load_file(filename)
        # sequence data is can be DataFrame or Dict
        if isinstance(data, pd.DataFrame):
            # data.set_index(self.seq_index_col, inplace=True)
            # data = data.to_dict(orient='index')
            pass
        elif isinstance(data, dict):
            # L*N -> n number of L
            data = {k: [v[:, i] for i in range(v.shape[-1])] for k, v in data.items()}
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['user_sequential_info']['columns'])
        else:
            raise ValueError("Sequence data must be DataFrame or Dict")
        # data.set_index(self.seq_index_col, inplace=True)
        if self.config['user_sequential_info'].get('use_cols', None) is not None:
            data = data[self.config['user_sequential_info']['use_cols']]
        return data

    def load_item_file(self):
        """Load all item data"""
        if self.config['item_info'] is None:
            return None
        data = self.item_data_client.load_file()
        if isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['item_info']['columns'])
            # data.set_index(self.config['item_info']['key'], inplace=True)
        else:
            raise ValueError("Item data must be DataFrame or Dict")
        # data = {k: torch.tensor(list(v), dtype=torch.int64) for k, v in data.items()}
        if self.config['item_info'].get('use_cols', None) is not None:
            data = data[self.config['item_info']['use_cols']]
        # data.set_index(self.config['item_info']['key'], inplace=True)
        return data
        
    
    def post_process(self, df):
        # filtering data in log table
        # case1: for retrieval model training
        if self.config['filter_settings'] is not None:
            for col, conditions in self.config['filter_settings'].items():
                condition_func = process_conditions(conditions)
                df = df.loc[df[col].apply(condition_func)]
        if self.config['post_process'] is None:
            return df
        else:
            raise NotImplementedError("Post process is not implemented yet")

    def load_one_day_data(self):
        filename = self.filenames[self.cur_idx]
        return self.load_file(filename), self.load_sequence_file(filename)

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx < len(self.filenames):
            cache_filename = self._data_cache.get('filename', None)
            if cache_filename is not None and self.filenames[self.cur_idx] == cache_filename:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from cache")
                data, seq_data = self._data_cache['data'], self._data_cache['seq']
            else:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from source")
                data, seq_data = self.load_one_day_data()
                self._data_cache['filename'] = self.filenames[self.cur_idx]
                self._data_cache['data'] = data
                self._data_cache['seq'] = seq_data
            self.cur_idx += 1
            return data, seq_data
        else:
            raise StopIteration
        
    def preload_start(self):
        def _load_data(queue):
            data, seq = self.load_one_day_data()
            queue.put((data, seq))
        print(f"Start preload file {self.filenames[self.cur_idx]}")
        queue = Queue()
        p = Process(target=_load_data, args=(queue,))
        return queue, p

    def preload_end(self, queue, p):
        data, seq = queue.get()
        self._data_cache['filename'] = self.filenames[self.cur_idx]
        self._data_cache['data'] = data
        self._data_cache['seq'] = seq
        p.join()



class DailyDataset(IterableDataset):
    def __init__(self, daily_iterator: DailyDataIterator, attrs, shuffle=False, preload=False, seed=42, **kwargs):
        super(DailyDataset, self).__init__(**kwargs)
        accelerator = Accelerator()
        self.rank = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.seed = seed
        self.daily_iterator = daily_iterator
        self.config = daily_iterator.config
        self.attrs = attrs
        self.shuffle = shuffle
        self.preload = preload
        self.preload_ratio = 0.8
        self.seq_index_col = daily_iterator.seq_index_col
        item_data = daily_iterator.load_item_file()
        self.item_feat_dataset = ItemDataset(item_data) if item_data is not None else None
        self.attrs.num_items = len(self.item_feat_dataset) if item_data is not None else None


    def __iter__(self):
        self.seed += 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise NotImplementedError("Not support `num_workers`>0 now.")
        for log_df, seq_df in self.daily_iterator:
            if self.shuffle:
                log_df = log_df.sample(frac=1).reset_index(drop=True)
            tensor_dict = df_to_tensor_dict(log_df)
            num_samples = log_df.shape[0]
            
            if self.preload:
                # `preload_index` controls when to preload
                preload_index = int(self.preload_ratio * num_samples)
            for i in range(num_samples):
                if self.preload and i == preload_index:
                    # preload the next-day logs
                    queue, p = self.daily_iterator.preload_start()
                    # p = Process(target=self.daily_iterator.preload)
                    # p.start()
                if self.preload and (i == num_samples-1):
                    # wait for the preload process
                    self.daily_iterator.preload_end(queue, p)
                    # p.join()

                data_dict = {k: v[i] for k, v in tensor_dict.items()}
                if seq_df is not None:
                    seq_data_dict = seq_df.loc[data_dict[self.seq_index_col].item()].to_dict()
                    data_dict['seq'] = seq_data_dict
                yield data_dict

    def get_item_feat(self, item_ids: torch.Tensor):
        """
        Return item features by item_ids

        Args:
            item_ids (torch.Tensor): [B x N] or [N]

        Returns:
            torch.Tensor: [B x N x F] or [N x F]
        """
        return self.item_feat_dataset.get_item_feat(item_ids)
    
    def get_item_loader(self, batch_size, num_workers=0, shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def get_datasets(config: Union[dict, str]):
    cp = ConfigProcessor(config)

    train_data_iterator = DailyDataIterator(cp.config, cp.train_files)
    test_data_iterator = DailyDataIterator(cp.config, cp.test_files)

    train_data = DailyDataset(train_data_iterator, shuffle=True, attrs=cp.attrs, preload=False)
    test_data = DailyDataset(test_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)
    return (train_data, test_data), cp.attrs

class DataCollator():
    pass

# test
if __name__ == '__main__':
    config = "./examples/data/recflow.json"
    cp = ConfigProcessor(config)
    print(cp.train_start_date, cp.train_end_date)
    print(cp.test_start_date, cp.test_end_date)
    print(cp.train_files)

    train_data_iterator = DailyDataIterator(cp.config, cp.train_files)
    test_data_iterator = DailyDataIterator(cp.config, cp.test_files)
    print(len(train_data_iterator), len(test_data_iterator))

    train_data = DailyDataset(train_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)
    test_data = DailyDataset(test_data_iterator, shuffle=False, attrs=cp.attrs, preload=False)

    for i, data in enumerate(train_data):
        print({k: v.shape for k, v in data.items()})
        if i > 3:
            break

    # data_iter = iter(train_data)
    # sample = next(data_iter)
    # sample = next(data_iter)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, num_workers=0)

    for i, batch in enumerate(train_loader):
        print({k: v.shape for k, v in batch.items()})
        # if i > 3:
        #     break
        



        