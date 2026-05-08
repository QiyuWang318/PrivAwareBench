#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseLocalChatModel, RuntimeArgs


class Mistral7Bv03Model(BaseLocalChatModel):
    def __init__(self, local_path: str, runtime_args: RuntimeArgs):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            local_path=local_path,
            runtime_args=runtime_args,
            trust_remote_code=False,
        )