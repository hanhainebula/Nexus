import json
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Union, get_args, get_origin
from dataclasses import dataclass, asdict, fields

logger = logging.getLogger(__name__)

DEFINITE_LOCAL_PATH_KEYS = {
    "train_data",
    "eval_data",
    "output_dir",
    "dataset_dir",
    "media_root",
    "image_root",
    "video_root",
    "cache_path",
    "cache_dir",
    "corpus_embd_save_dir",
    "eval_output_dir",
    "eval_output_path",
}
MAYBE_LOCAL_OR_REMOTE_KEYS = {
    "model_name_or_path",
    "processor_name_or_path",
    "embedder_name_or_path",
}


def init_argument(type_, x):
    if x is None:
        return None
    origin = get_origin(type_)
    args = get_args(type_)

    if type_ in [Any, object]:
        return x

    if origin is Union:
        for tmp_type_ in args:
            if tmp_type_ is type(None):
                continue
            try:
                return init_argument(tmp_type_, x)
            except TypeError:
                continue
        raise TypeError(f"Failed to init argument {x} ({type(x)}) to {type_}.")

    if isinstance(x, dict):
        try:
            # is subclass of AbsArguments
            return type_.from_dict(x)
        except:
            # dict
            return dict(x)
    elif origin in [list, tuple]:
        item_type = args[0] if len(args) > 0 else Any
        converted = [init_argument(item_type, i) for i in x]
        return converted if origin is list else tuple(converted)
    elif isinstance(x, list):
        return list(x)
    else:
        tmp_x = None
        try:
            if isinstance(type_, type) and isinstance(x, type_):
                tmp_x = x
            else:
                tmp_x = type_(x)
                logger.warning(f"Init argument: Convert {x} ({type(x)}) to {tmp_x} ({type(tmp_x)}).")
        except:
            try:
                for tmp_type_ in args:
                    if tmp_type_ is None:
                        continue
                    if tmp_type_ is type(None):
                        continue
                    if isinstance(x, tmp_type_):
                        tmp_x = x
                        break
                    else:
                        tmp_x = tmp_type_(x)
                        logger.warning(f"Init argument: Convert {x} ({type(x)}) to {tmp_x} ({type(tmp_x)}).")
            except (AttributeError, TypeError):
                raise TypeError(f"Failed to init argument {x} ({type(x)}) to {type_}.")
        if tmp_x is None:
            raise TypeError(f"Failed to init argument {x} ({type(x)}) to {type_}.")
        return tmp_x


def _looks_like_remote_reference(value: str) -> bool:
    return "://" in value


def _resolve_local_path_value(value: str, base_dir: Path, force: bool = False) -> str:
    if value in [None, ""]:
        return value

    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or _looks_like_remote_reference(expanded):
        return expanded

    if not force:
        candidate = base_dir / expanded
        if not (expanded.startswith(".") or expanded.startswith("..") or candidate.exists()):
            return value

    return str((base_dir / expanded).resolve())


def _resolve_config_paths(config_dict: dict, base_dir: Path) -> dict:
    resolved = dict(config_dict)
    for key, value in list(resolved.items()):
        if value in [None, ""]:
            continue

        if key in DEFINITE_LOCAL_PATH_KEYS:
            if isinstance(value, list):
                resolved[key] = [_resolve_local_path_value(item, base_dir, force=True) for item in value]
            elif isinstance(value, str):
                resolved[key] = _resolve_local_path_value(value, base_dir, force=True)
            continue

        if key in MAYBE_LOCAL_OR_REMOTE_KEYS and isinstance(value, str):
            resolved[key] = _resolve_local_path_value(value, base_dir, force=False)
    return resolved


@dataclass
class AbsArguments:

    def to_dict(self):
        return asdict(self)

    def to_json(self, save_path: Union[str, Path], overwrite: bool = False):
        if isinstance(save_path, str):
            save_path = Path(save_path)

        if save_path.exists() and not overwrite:
            raise FileExistsError(f"{save_path} already exists. Set `overwrite=True` to overwrite the file.")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, _dict: dict):
        _fields_w_type = [_field.name for _field in fields(cls)]
        
        _fields_wo_type = [
            x for x in cls.__dict__.keys()
            if not x.startswith("__") and not x.endswith("__") and x not in _fields_w_type
        ]
        _fields_wo_type_dict = {
            _field_name: _dict.pop(_field_name)
            for _field_name in _fields_wo_type if _field_name in _dict
        }
        
        for k in _dict.keys():
            if k not in [_field.name for _field in fields(cls)]:
                raise ValueError(f'{k} is not in fields({cls}).')

        for _field in fields(cls):
            if _field.name not in _dict:
                continue
            _dict[_field.name] = init_argument(_field.type, _dict[_field.name])
        
        _instance = cls(**_dict)
        
        for _field_name, _field_value in _fields_wo_type_dict.items():
            setattr(_instance, _field_name, _field_value)
        return _instance

    @classmethod
    def from_json(cls, load_path: Union[str, Path]):
        if isinstance(load_path, str):
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"{load_path} does not exist.")

        with open(load_path, "r", encoding="utf-8") as f:
            _dict = _resolve_config_paths(json.load(f), load_path.parent)
            return cls.from_dict(_dict)

    @classmethod
    def from_yaml(cls, load_path: Union[str, Path]):
        if isinstance(load_path, str):
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"{load_path} does not exist.")

        with open(load_path, "r", encoding="utf-8") as f:
            _dict = _resolve_config_paths(yaml.safe_load(f), load_path.parent)
            return cls.from_dict(_dict)
