from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass
from torch.utils.data import DataLoader, IterableDataset, Dataset
from typing import Any, Dict, List, Tuple, Union

from Nexus.abc.training.reranker import AbsRerankerCollator
from Nexus.training.reranker.recommendation.arguments import DataArguments
from Nexus.modules.dataset import get_client, read_json, nested_dict_update, extract_timestamp, process_conditions, df_to_tensor_dict


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
    def __init__(self, config: Union[dict, str, DataArguments]):
        self.config: DataArguments = self._load_config(config)

    def _load_config(self, config: Union[dict, str]) -> DataArguments:
        """Loads the configuration from either a dictionary or a JSON file."""
        if isinstance(config, DataArguments):
            args = config
        elif isinstance(config, dict):
            args = DataArguments.from_dict(config)
        elif isinstance(config, str):
            args = DataArguments.from_json(config)
        else:
            raise TypeError("Config must be a dictionary or path to JSON file.")
        return args

    def split_config(self) -> Tuple[DataArguments, DataArguments]:
        data_client = get_client(self.config.type, self.config.url)
        train_files, eval_files = data_client.get_train_eval_filenames(
            self.config.file_partition,
            self.config.train_period,
            self.config.test_period,
        )
        train_config = deepcopy(self.config)
        eval_config = deepcopy(self.config)
        train_config.files = train_files
        eval_config.files = eval_files
        return train_config, eval_config


class ShardedDataIterator(object):
    """This class is used to iterate over each day of data (dataframe)
    
    Args:
        config (dict): config dictionary
        filenames (List[str]): list of filenames to iterate over
    """
    def __init__(self, config: DataArguments):
        self.config = config
        self.data_client = get_client(config.type, config.url)
        self.number_to_filename = self.data_client.build_file_index(self.config.file_partition)
        self.filename_to_number = {filename: number for number, filename in self.number_to_filename.items()}
        self.seq_data_clients = []
        self.seq_number_to_filename = []
        if len(config.user_sequential_info) > 0:
            # use given key column and sequence data file
            for seq_info in config.user_sequential_info:
                client = get_client(config.type, seq_info["url"])
                self.seq_data_clients.append(client)
                self.seq_number_to_filename.append(client.build_file_index(config.file_partition))

        if config.item_info is not None:
            self.item_col = config.item_info["key"]
            self.item_data_client = get_client(config.type, config.item_info["url"])
        else:
            self.item_col = None
            self.item_data_client = None

        self.filenames = config.files
        self.cur_idx = 0
        data, seq_data = self.load_single_sharded_data()
        self._data_cache = {'filename': self.filenames[self.cur_idx], 'data': data, 'seq': seq_data}

    def _columns_rename(self, df: pd.DataFrame):
        """ Rename columns to remove whitespace and modify `features` in the config consistently"""
        rename_map = {}
        for col in df.columns:
            new_col = col.strip()
            rename_map[col] = new_col
        return df.rename(columns=rename_map)

    def load_file(self, filename: str):
        """Load one day of data"""
        data = self.data_client.load_file(filename)
        data = self._columns_rename(data)
        # use given columns
        # if self.config.context_features:
        seq_index_cols = [seq_info["key"] for seq_info in self.config.user_sequential_info]
        features = self.config.context_features + self.config.item_features + self.config.labels
        keep_cols = seq_index_cols + features
        data = data[keep_cols]
        data = self.post_process(data)
        return data
    
    
    def load_sequence_files(self, filename: str) -> List[pd.DataFrame]:
        """Load one day of sequence data"""
        if len(self.seq_data_clients) <= 0:
            return None
        date_or_number = self.filename_to_number[filename]
        seq_data_list = []
        for i, seq_data_client in enumerate(self.seq_data_clients):
            seq_info = self.config.user_sequential_info[i]
            seq_index_col = seq_info["key"]
            number_to_filename = self.seq_number_to_filename[i]
            seq_filename = number_to_filename[date_or_number]
            seq_data_i = seq_data_client.load_file(seq_filename)

            # sequence data is can be DataFrame or Dict
            if isinstance(seq_data_i, pd.DataFrame):
                seq_data_i.set_index(seq_index_col, inplace=True, drop=True)
            elif isinstance(seq_data_i, dict):
                # L*N -> n number of L
                seq_data_i = {k: [v[:, i] for i in range(v.shape[-1])] for k, v in seq_data_i.items()}
                seq_data_i = pd.DataFrame.from_dict(seq_data_i, orient='index', columns=seq_info["columns"])
            else:
                raise ValueError("Sequence data must be DataFrame or Dict")
            if seq_info["use_cols"] is not None:
                seq_data_i = seq_data_i[seq_info["use_cols"]]
            seq_data_list.append(seq_data_i.sort_index())
        return seq_data_list

    def load_item_file(self) -> pd.DataFrame:
        """Load all item data"""
        if self.config.item_info is None:
            return None
        data = self.item_data_client.load_file()
        if isinstance(data, pd.DataFrame):
            data.set_index(self.config.item_info["key"], inplace=True, drop=True)
        elif isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index', columns=self.config.item_info["columns"])
        else:
            raise ValueError("Item data must be DataFrame or Dict")
        if self.config.item_info["use_cols"] is not None:
            data = data[self.config.item_info["use_cols"]]
        return data
        
    
    def post_process(self, df) -> pd.DataFrame:
        # filtering data in log table
        # case1: for retrieval model training
        if (self.config.filter_settings is not None) and (len(self.config.filter_settings) > 0):
            for col, conditions in self.config.filter_settings.items():
                condition_funcs = process_conditions(conditions)
                # df = df.loc[df[setting.by].map(condition_func)]
                for func in condition_funcs:
                    df = df[func(df[col])]
        return df


    def load_single_sharded_data(self):
        filename = self.filenames[self.cur_idx]
        return self.load_file(filename), self.load_sequence_files(filename)

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
                data, seq_data_list = self._data_cache['data'], self._data_cache['seq']
                print(f"Load dataset sucessfully.")
            else:
                print(f"Load dataset file {self.filenames[self.cur_idx]} from source")
                data, seq_data_list = self.load_single_sharded_data()
                print(f"Load dataset sucessfully.")
                self._data_cache['filename'] = self.filenames[self.cur_idx]
                self._data_cache['data'] = data
                self._data_cache['seq'] = seq_data_list
            self.cur_idx += 1
            return data, seq_data_list
        else:
            raise StopIteration
        


class ShardedDataset(IterableDataset):
    def __init__(self, config: DataArguments, shuffle=False, seed=42, **kwargs):
        super(ShardedDataset, self).__init__()
        self.seed = seed
        self.sharded_data_iterator = ShardedDataIterator(config)
        self.config = config
        self.seq_index_cols = [seq_info['key'] for seq_info in self.config.user_sequential_info]
        self.seq_data_names = [seq_info['name'] for seq_info in self.config.user_sequential_info]
        self.shuffle = shuffle
        item_data = self.sharded_data_iterator.load_item_file()
        self.item_feat_dataset = ItemDataset(item_data) if item_data is not None else None
        # self.long_sequence_selectors: Dict[str, BaseSelector] = {}
        

    # def add_long_sequence_selectors(self, selectors: Union[BaseSelector, List[BaseSelector]]):
    #     """Add long sequence selector function which will be applied after loading data"""
    #     if isinstance(selectors, BaseSelector):
    #         selectors = [selectors]
    #     for selector in selectors:
    #         assert selector.seq_name in self.config.user_sequential_info, f"sequential features `{selector.seq_name}` not provided in config"
    #         assert selector.seq_name not in self.long_sequence_selectors, f"duplicate selector for sequential features `{selector.seq_name}`"
    #         self.long_sequence_selectors[selector.seq_name] = selector
    #         print(f"Long Sequence Selector `{selector.__class__.__name__}` added to feature `{selector.seq_name}`")
    def __len__(self):
        return 1000000000

    def __iter__(self):
        self.seed += 1
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        for log_df, seq_df_list in self.sharded_data_iterator:
            if self.shuffle:
                log_df = log_df.sample(frac=1).reset_index(drop=True)
            log_dict = log_df.to_dict(orient='list')
            num_samples = log_df.shape[0]
            
            for i in range(num_samples):
                if i % num_workers == worker_id:
                    data_dict = {k: v[i] for k, v in log_dict.items()}
                    if len(seq_df_list) > 0:
                        for i, seq_df in enumerate(seq_df_list):
                            seq_data_dict = seq_df.loc[data_dict[self.seq_index_cols[i]]].to_dict()
                            seq_data_dict = {k: np.copy(v) for k, v in seq_data_dict.items()}
                            # if self.seq_data_names[i] in self.long_sequence_selectors:
                            #     target_item_dict: Dict[str, Any] = {k: v for k, v in data_dict.items() if k in self.config.item_features}
                            #     seq_data_dict = self.long_sequence_selectors[self.seq_data_names[i]](seq_data_dict, target_item_dict)
                            data_dict[self.seq_data_names[i]] = seq_data_dict
                    for index_key in self.seq_index_cols:
                        del data_dict[index_key]
                    yield (data_dict, )

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



# class ConfigProcessor(object):
#     """
#     This class is used to split the data into train and test sets at file level.

#     Args:
#         config (Union[dict, str]): Configuration dictionary or path to JSON file
#     """
#     def __init__(self, config: Union[dict, str]):
#         self.config = self._load_config(config)
#         self._check_config()
#         self.data_dir = self.config['url']
#         self.train_start_date, self.train_end_date = self._get_date_range(mode='train')
#         self.test_start_date, self.test_end_date = self._get_date_range(mode='test')
#         self.train_files = self._get_valid_filenames(self.train_start_date, self.train_end_date)
#         self.test_files = self._get_valid_filenames(self.test_start_date, self.test_end_date)
#         self.attrs = self.get_attrs()


#     def get_attrs(self):
#         stats = Statistics.from_dict(self.config['stats'])
#         def remove_whitespace(string):
#             return string.strip()
#         self.config['context_features'] = list(map(remove_whitespace, self.config['context_features']))
#         self.config['item_features'] = list(map(remove_whitespace, self.config['item_features']))
#         if self.config.get("user_sequential_info", None):
#             self.config['seq_features'] = list(map(remove_whitespace, self.config['user_sequential_info']['use_cols']))
#         else:
#             self.config['seq_features'] = []
#         return DataAttr4Model(
#             fiid=self.config['item_col'],
#             flabels=self.config['labels'],
#             features=self.config['context_features']+self.config['item_features'],
#             context_features=self.config['context_features'],
#             item_features=self.config['item_features'],
#             seq_features=self.config['seq_features'],
#             num_items=getattr(stats, self.config['item_col']),
#             stats=stats,
#         )

#     def _load_config(self, config: Union[dict, str]):
#         """Loads the configuration from either a dictionary or a JSON file."""
#         if isinstance(config, dict):
#             pass
#         elif isinstance(config, str):
#             config = read_json(config)
#         else:
#             raise TypeError("Config must be a dictionary or path to JSON file.")
#         # update nested config to default config
#         config = nested_dict_update(DEFAULT_CONFIG, config)
#         return config

#     def _check_config(self):
#         """Checks that all required keys are present in the configuration."""
#         def _flatten_keys(dictionary):
#             # flatten nested keys with /
#             valid_keys = []
#             for key, value in dictionary.items():
#                 if isinstance(value, dict):
#                     valid_keys.extend([f"{key}/{subkey}" for subkey in _flatten_keys(value)])
#                 else:
#                     valid_keys.append(key)
#             return valid_keys
        
#         valid_keys = _flatten_keys(self.config)
#         missing_keys = []
#         for key in REQUIRED_DATA_CONFIG:
#             if key not in valid_keys:
#                 missing_keys.append(key)
#         assert key in valid_keys, f"Missing required keys: {missing_keys}"


#     def _get_date_range(self, mode: str):
#         """Get start and end date from config
#         Args:
#             mode (str): train or test
#         """
#         date_format: str = self.config['date_format']
#         start_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['start_date'], date_format)
#         end_date: datetime = datetime.strptime(self.config[f'{mode}_settings']['end_date'], date_format)
#         return start_date, end_date


#     def _get_valid_filenames(self, start_date: datetime, end_date: datetime) -> List[str]:
#         """Get all used filenames among given date range in the dataset"""
#         data_client = get_client(self.config['type'], self.config['url'])
#         filenames = sorted(data_client.list_dir())
#         dates = [extract_timestamp(filename) for filename in filenames]
#         file_idx = [idx for idx, date in enumerate(dates) if start_date <= date < end_date]
#         valid_filenames = [filenames[idx] for idx in file_idx]
#         return valid_filenames
    

# class DailyDataIterator(object):
#     """This class is used to iterate over each day of data (dataframe)
    
#     Args:
#         config (dict): config dictionary
#         filenames (List[str]): list of filenames to iterate over
#     """
#     def __init__(self, config, filenames):
#         self.config = config
#         self.data_client = get_client(config['type'], config['url'])
#         if config['user_sequential_info']:
#             # use given key column and sequence data file
#             self.seq_index_col = config['user_sequential_info']['key']
#             self.seq_data_client = get_client(config['type'], config['user_sequential_info']['url'])
#         else:
#             # use given key column, no additional files needed
#             self.seq_index_col = None
#             self.seq_data_client = None

#         if config['item_info']:
#             self.item_col = config['item_info']['key']
#             self.item_data_client = get_client(config['type'], config['item_info']['url'])
#         else:
#             self.item_col = None
#             self.item_data_client = None

#         self.filenames = filenames
#         self.cur_idx = 0
#         self._data_cache = {'filename': None, 'data': None, 'seq': None}

#     def _columns_rename(self, df: pd.DataFrame):
#         """ Rename columns to remove whitespace and modify `features` in the config consistently"""
#         rename_map = {}
#         for col in df.columns:
#             new_col = col.strip()
#             rename_map[col] = new_col
#         self.config['features'] = self.config['context_features'] + self.config['item_features']
#         return df.rename(columns=rename_map)

#     def load_file(self, filename: str):
#         """Load one day of data"""
#         data = self.data_client.load_file(filename)
#         data = self._columns_rename(data)
#         # use given columns
#         if self.config['features']:
#             if self.seq_index_col is not None and self.seq_index_col not in self.config['features']:
#                 # add key column for merge
#                 keep_cols = [self.seq_index_col] + self.config['features'] + self.config['labels']
#             else:
#                 keep_cols = self.config['features'] + self.config['labels']
#             data = data[keep_cols]
#         data = self.post_process(data)
#         return data
    
#     def load_sequence_file(self, filename: str):
#         """Load one day of sequence data"""
#         if self.seq_data_client is None:
#             return None
#         date_str = filename.split('.')[0]
#         fileformat = self.config['user_sequential_info'].get('file_format', 'auto')
#         if fileformat != 'auto':
#             filename = f"{date_str}.{fileformat}"
#         else:
#             # auto mode: find the file with the same date
#             all_files = self.seq_data_client.list_dir()
#             for file in all_files:
#                 if date_str in file:
#                     filename = file
#                     break
#         data = self.seq_data_client.load_file(filename)
#         # sequence data is can be DataFrame or Dict
#         if isinstance(data, pd.DataFrame):
#             # data.set_index(self.seq_index_col, inplace=True)
#             # data = data.to_dict(orient='index')
#             pass
#         elif isinstance(data, dict):
#             # L*N -> n number of L
#             data = {k: [v[:, i] for i in range(v.shape[-1])] for k, v in data.items()}
#             data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['user_sequential_info']['columns'])
#         else:
#             raise ValueError("Sequence data must be DataFrame or Dict")
#         # data.set_index(self.seq_index_col, inplace=True)
#         if self.config['user_sequential_info'].get('use_cols', None) is not None:
#             data = data[self.config['user_sequential_info']['use_cols']]
#         return data

#     def load_item_file(self):
#         """Load all item data"""
#         if self.config['item_info'] is None:
#             return None
#         data = self.item_data_client.load_file()
#         if isinstance(data, pd.DataFrame):
#             pass
#         elif isinstance(data, dict):
#             data = pd.DataFrame.from_dict(data, orient='index', columns=self.config['item_info']['columns'])
#             # data.set_index(self.config['item_info']['key'], inplace=True)
#         else:
#             raise ValueError("Item data must be DataFrame or Dict")
#         # data = {k: torch.tensor(list(v), dtype=torch.int64) for k, v in data.items()}
#         if self.config['item_info'].get('use_cols', None) is not None:
#             data = data[self.config['item_info']['use_cols']]
#         # data.set_index(self.config['item_info']['key'], inplace=True)
#         return data
        
    
#     def post_process(self, df):
#         # filtering data in log table
#         # case1: for retrieval model training
#         if self.config['filter_settings'] is not None:
#             for col, conditions in self.config['filter_settings'].items():
#                 condition_func = process_conditions(conditions)
#                 df = df.loc[df[col].apply(condition_func)]
#         if self.config['post_process'] is None:
#             return df
#         else:
#             raise NotImplementedError("Post process is not implemented yet")

#     def load_one_day_data(self):
#         filename = self.filenames[self.cur_idx]
#         return self.load_file(filename), self.load_sequence_file(filename)

#     def __len__(self):
#         return len(self.filenames)

#     def __iter__(self):
#         self.cur_idx = 0
#         return self

#     def __next__(self):
#         if self.cur_idx < len(self.filenames):
#             cache_filename = self._data_cache.get('filename', None)
#             if cache_filename is not None and self.filenames[self.cur_idx] == cache_filename:
#                 print(f"Load dataset file {self.filenames[self.cur_idx]} from cache")
#                 data, seq_data = self._data_cache['data'], self._data_cache['seq']
#             else:
#                 print(f"Load dataset file {self.filenames[self.cur_idx]} from source")
#                 data, seq_data = self.load_one_day_data()
#                 self._data_cache['filename'] = self.filenames[self.cur_idx]
#                 self._data_cache['data'] = data
#                 self._data_cache['seq'] = seq_data
#             self.cur_idx += 1
#             return data, seq_data
#         else:
#             raise StopIteration
        
#     def preload_start(self):
#         def _load_data(queue):
#             data, seq = self.load_one_day_data()
#             queue.put((data, seq))
#         print(f"Start preload file {self.filenames[self.cur_idx]}")
#         queue = Queue()
#         p = Process(target=_load_data, args=(queue,))
#         return queue, p

#     def preload_end(self, queue, p):
#         data, seq = queue.get()
#         self._data_cache['filename'] = self.filenames[self.cur_idx]
#         self._data_cache['data'] = data
#         self._data_cache['seq'] = seq
#         p.join()


# class DailyDataset(IterableDataset):
#     def __init__(self, daily_iterator: DailyDataIterator, attrs, shuffle=False, preload=False, seed=42, **kwargs):
#         super(DailyDataset, self).__init__(**kwargs)
#         accelerator = Accelerator()
#         self.rank = accelerator.process_index
#         self.num_processes = accelerator.num_processes
#         self.seed = seed
#         self.daily_iterator = daily_iterator
#         self.config = daily_iterator.config
#         self.attrs = attrs
#         self.shuffle = shuffle
#         self.preload = preload
#         self.preload_ratio = 0.8
#         self.seq_index_col = daily_iterator.seq_index_col
#         item_data = daily_iterator.load_item_file()
#         self.item_feat_dataset = ItemDataset(item_data) if item_data is not None else None
#         self.attrs.num_items = len(self.item_feat_dataset) if item_data is not None else None
        

#     def get_item_loader(self, item_batch_size):
#         """
#         Returns the evaluation [`~torch.utils.data.DataLoader`].

#         Subclass and override this method if you want to inject some custom behavior.

#         Args:
#             train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
#                 If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
#         """
#         loader = DataLoader(self.item_feat_dataset, batch_size=item_batch_size)
#         return loader
    
    
#     # TODO: 看一下要不要实现
#     # def __len__(self, ):


#     def __iter__(self):
#         self.seed += 1
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is not None:
#             raise NotImplementedError("Not support `num_workers`>0 now.")
#         for log_df, seq_df in self.daily_iterator:
#             if self.shuffle:
#                 log_df = log_df.sample(frac=1).reset_index(drop=True)
#             tensor_dict = df_to_tensor_dict(log_df)
#             num_samples = log_df.shape[0]
            
#             if self.preload:
#                 # `preload_index` controls when to preload
#                 preload_index = int(self.preload_ratio * num_samples)
#             for i in range(num_samples):
#                 if self.preload and i == preload_index:
#                     # preload the next-day logs
#                     queue, p = self.daily_iterator.preload_start()
#                 if self.preload and (i == num_samples-1):
#                     # wait for the preload process
#                     self.daily_iterator.preload_end(queue, p)

#                 data_dict = {k: v[i] for k, v in tensor_dict.items()}
#                 if seq_df is not None:
#                     seq_data_dict = seq_df.loc[data_dict[self.seq_index_col].item()].to_dict()
#                     data_dict['seq'] = seq_data_dict
#                     # data_dict.update(seq_data_dict)
                    
                
#                 yield (data_dict,)
                    
                    
@dataclass
class AbsRecommenderRerankerCollator(AbsRerankerCollator):
    """
    The abstract reranker collator.
    """    
    def __call__(self, features):
        features = [f[0] for f in features]
        all_dicts = {}
        
        for k, v in features[0].items():
            if isinstance(v, dict):
                all_dicts[k] = {}
                for _k in v.keys():
                    all_dicts[k][_k] = []
            else:
                all_dicts[k] = []
                
        for data_dict in features:
            for k, v in data_dict.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        if not isinstance(_v, torch.Tensor):
                            all_dicts[k][_k].append(torch.tensor(_v))
                        else:
                            all_dicts[k][_k].append(_v)
                else:
                    if not isinstance(v, torch.Tensor):
                        all_dicts[k].append(torch.tensor(v))
                    else:
                        all_dicts[k].append(v)
                
        for k, v in features[0].items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    all_dicts[k][_k] = torch.stack(all_dicts[k][_k], dim=0)
            else:
                all_dicts[k] = torch.stack(all_dicts[k], dim=0)
            
        
        return all_dicts
        # return features
        
def get_datasets(config: Union[dict, str]):
    config_processor = ConfigProcessor(config)
    train_config, eval_config = config_processor.split_config()

    train_data = ShardedDataset(train_config, shuffle=True, preload=False)
    test_data = ShardedDataset(eval_config, shuffle=False, preload=False)
    attr = train_config.to_attr()
    if train_data.item_feat_dataset is not None:
        # when candidate item dataset is given, the number of items is set to the number of items in the dataset
        # instead of the max item id in the dataset
        attr.num_items = len(train_data.item_feat_dataset)
    return (train_data, test_data), attr
