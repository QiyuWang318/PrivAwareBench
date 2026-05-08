#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseLocalChatModel, RuntimeArgs


class Qwen25Model(BaseLocalChatModel):
    def __init__(self, local_path: str, runtime_args: RuntimeArgs):
        super().__init__(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            local_path=local_path,
            runtime_args=runtime_args,
            trust_remote_code=False,
        )