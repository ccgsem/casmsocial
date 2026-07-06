# Copyright (c) ZeroC, Inc.

# slice2py version 3.8.2

from __future__ import annotations

from dataclasses import dataclass

import IcePy
from Ice.UserException import UserException


@dataclass
class TableNotFound(UserException):
    """
    Notes
    -----
        The Slice compiler generated this exception dataclass from Slice exception ``::arrowservice::TableNotFound``.
    """
    tableName: str = ""

    _ice_id = "::arrowservice::TableNotFound"

_arrowservice_TableNotFound_t = IcePy.defineException(
    "::arrowservice::TableNotFound",
    TableNotFound,
    (),
    None,
    (("tableName", (), IcePy._t_string, False, 0),))

setattr(TableNotFound, '_ice_type', _arrowservice_TableNotFound_t)

__all__ = ["TableNotFound", "_arrowservice_TableNotFound_t"]
