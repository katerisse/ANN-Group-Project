"""Registry of Phase 1 unlearning methods and their trainable scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from unlearning.methods import head_only, mask, neggrad, original, retain_only, scrub


@dataclass(frozen=True)
class MethodSpec:
    name: str
    scope: str
    loss: Callable | None
    uses_forget: bool
    uses_teacher: bool


METHODS = {
    "original": MethodSpec("original", "none", None, False, False),
    "mask": MethodSpec("mask", "none", None, False, False),
    "head_only": MethodSpec("head_only", "head", head_only.loss, False, False),
    "retain_only": MethodSpec("retain_only", "conv5_head", retain_only.loss, False, False),
    "neggrad": MethodSpec("neggrad", "conv5_head", neggrad.loss, True, False),
    "scrub": MethodSpec("scrub", "conv5_head", scrub.loss, True, True),
}

__all__ = ["METHODS", "MethodSpec", "mask", "original"]
