import json
import torch

from datetime import datetime
import torch
from typing import List
from UniRetrieval.training.embedder.recommendation.arguments import REQUIRED_DATA_CONFIG, DEFAULT_CONFIG
from copy import deepcopy
import re


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