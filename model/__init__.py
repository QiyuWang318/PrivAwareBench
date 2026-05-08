#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from .base import RuntimeArgs
from .qwen25 import Qwen25Model
from .mistral import Mistral7Bv03Model
from .llama31 import Llama31Model


MODEL_REGISTRY = {
    "qwen25_7b": Qwen25Model,
    "mistral_7b_v03": Mistral7Bv03Model,
    "llama31_8b": Llama31Model,
}


def build_model(model_alias: str, local_path: str, runtime_args: RuntimeArgs):
    if model_alias not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model alias: {model_alias}. "
            f"Choices: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_alias](local_path=local_path, runtime_args=runtime_args)