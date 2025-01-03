import json
import yaml
import logging
from pathlib import Path
from typing import Union
from dataclasses import dataclass, asdict, fields

logger = logging.getLogger(__name__)


def init_argument(type_, x):
    if x is None:
        return None
    tmp_x = None
    try:
        if isinstance(x, type_):
            tmp_x = x
        else:
            tmp_x = type_(x)
            logger.warning(f"Init argument: Convert {x} ({type(x)}) to {tmp_x} ({type(tmp_x)}).")
    except:
        try:
            for tmp_type_ in type_.__args__:
                if tmp_type_ is None:
                    continue
                if isinstance(x, tmp_type_):
                    tmp_x = x
                    break
                else:
                    tmp_x = type_(x)
                    logger.warning(f"Init argument: Convert {x} ({type(x)}) to {tmp_x} ({type(tmp_x)}).")
        except AttributeError:
            raise TypeError(f"Failed to init argument {x} ({type(x)}) to {type_}.")
    if tmp_x is None:
        raise TypeError(f"Failed to init argument {x} ({type(x)}) to {type_}.")
    return tmp_x


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
        for k in _dict.keys():
            if k not in [_field.name for _field in fields(cls)]:
                raise ValueError(f'{k} is not in fields({cls}).')
        for _field in fields(cls):
            if _field.name not in _dict:
                continue
            if isinstance(_dict[_field.name], dict):
                try:
                    if issubclass(_field.type, AbsArguments):
                    # if isinstance(_field, AbsArguments):
                        _dict[_field.name] = _field.type.from_dict(_dict[_field.name])
                except TypeError: 
                    _dict[_field.name] = dict(_dict[_field.name])
            else:
                if isinstance(_dict[_field.name], list):
                    _dict[_field.name] = [init_argument(_field.type, x) for x in _dict[_field.name]]
                else:
                    _dict[_field.name] = init_argument(_field.type, _dict[_field.name])
        return cls(**_dict)

    @classmethod
    def from_json(cls, load_path: Union[str, Path]):
        if isinstance(load_path, str):
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"{load_path} does not exist.")

        with open(load_path, "r", encoding="utf-8") as f:
            _dict = json.load(f)
            return cls.from_dict(_dict)

    @classmethod
    def from_yaml(cls, load_path: Union[str, Path]):
        if isinstance(load_path, str):
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"{load_path} does not exist.")

        with open(load_path, "r", encoding="utf-8") as f:
            _dict = yaml.safe_load(f)
            return cls.from_dict(_dict)
