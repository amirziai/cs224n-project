#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from functools import reduce
from typing import TypeVar, List, Tuple, Dict, Any

T = TypeVar('T')


def unzip(xs: List[Tuple[List[T], List[T]]]) -> Tuple[List[List[T]], List[List[T]]]:
    return list(zip(*xs))


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def merge_dicts(*args: Dict) -> Dict[Any, Any]:
    return reduce(lambda x, y: {**x, **y}, args)
