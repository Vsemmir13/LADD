import os, random, sys
import torch
from typing import Dict, Any, List
import yaml
import logging

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
