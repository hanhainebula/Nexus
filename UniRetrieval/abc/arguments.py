import json
import yaml
from pathlib import Path
from typing import Union
from dataclasses import dataclass, asdict, fields


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
        for _field in fields(cls):
            if _field.name in _dict and isinstance(_field.type, type) and issubclass(_field.type, AbsArguments):
                _dict[_field.name] = _field.type.from_dict(_dict[_field.name])
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
