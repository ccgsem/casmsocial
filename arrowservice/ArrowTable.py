# Copyright (c) ZeroC, Inc.

# slice2py version 3.8.2

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import IcePy

from arrowservice.ByteSeq import _arrowservice_ByteSeq_t

if TYPE_CHECKING:
    pass


@dataclass
class ArrowTable:
    """
    Notes
    -----
        The Slice compiler generated this dataclass from Slice struct ``::arrowservice::ArrowTable``.
    """
    data: bytes = field(default_factory=bytes)

_arrowservice_ArrowTable_t = IcePy.defineStruct(
    "::arrowservice::ArrowTable",
    ArrowTable,
    (),
    (("data", (), _arrowservice_ByteSeq_t),))

__all__ = ["ArrowTable", "_arrowservice_ArrowTable_t"]
