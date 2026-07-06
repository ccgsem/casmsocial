# Copyright (c) ZeroC, Inc.

# slice2py version 3.8.2

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload

import IcePy
from Ice.Object import Object
from Ice.ObjectPrx import ObjectPrx, checkedCast, checkedCastAsync, uncheckedCast
from Ice.OperationMode import OperationMode

from arrowservice.ArrowServer_forward import _arrowservice_ArrowServerPrx_t
from arrowservice.ArrowTable import _arrowservice_ArrowTable_t
from arrowservice.StringSeq import _arrowservice_StringSeq_t
from arrowservice.TableNotFound import _arrowservice_TableNotFound_t

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from Ice.Current import Current

    from arrowservice.ArrowTable import ArrowTable


class ArrowServerPrx(ObjectPrx):
    """
    Notes
    -----
        The Slice compiler generated this proxy class from Slice interface ``::arrowservice::ArrowServer``.
    """

    def uploadTable(self, tableName: str, table: ArrowTable, context: dict[str, str] | None = None) -> None:
        return ArrowServer._op_uploadTable.invoke(self, ((tableName, table), context))

    def uploadTableAsync(self, tableName: str, table: ArrowTable, context: dict[str, str] | None = None) -> Awaitable[None]:
        return ArrowServer._op_uploadTable.invokeAsync(self, ((tableName, table), context))

    def getTable(self, tableName: str, context: dict[str, str] | None = None) -> ArrowTable:
        return ArrowServer._op_getTable.invoke(self, ((tableName, ), context))

    def getTableAsync(self, tableName: str, context: dict[str, str] | None = None) -> Awaitable[ArrowTable]:
        return ArrowServer._op_getTable.invokeAsync(self, ((tableName, ), context))

    def getTableSchema(self, tableName: str, context: dict[str, str] | None = None) -> str:
        return ArrowServer._op_getTableSchema.invoke(self, ((tableName, ), context))

    def getTableSchemaAsync(self, tableName: str, context: dict[str, str] | None = None) -> Awaitable[str]:
        return ArrowServer._op_getTableSchema.invokeAsync(self, ((tableName, ), context))

    def listTableNames(self, context: dict[str, str] | None = None) -> list[str]:
        return ArrowServer._op_listTableNames.invoke(self, ((), context))

    def listTableNamesAsync(self, context: dict[str, str] | None = None) -> Awaitable[list[str]]:
        return ArrowServer._op_listTableNames.invokeAsync(self, ((), context))

    @staticmethod
    def checkedCast(
        proxy: ObjectPrx | None,
        facet: str | None = None,
        context: dict[str, str] | None = None
    ) -> ArrowServerPrx | None:
        return checkedCast(ArrowServerPrx, proxy, facet, context)

    @staticmethod
    def checkedCastAsync(
        proxy: ObjectPrx | None,
        facet: str | None = None,
        context: dict[str, str] | None = None
    ) -> Awaitable[ArrowServerPrx | None ]:
        return checkedCastAsync(ArrowServerPrx, proxy, facet, context)

    @overload
    @staticmethod
    def uncheckedCast(proxy: ObjectPrx, facet: str | None = None) -> ArrowServerPrx:
        ...

    @overload
    @staticmethod
    def uncheckedCast(proxy: None, facet: str | None = None) -> None:
        ...

    @staticmethod
    def uncheckedCast(proxy: ObjectPrx | None, facet: str | None = None) -> ArrowServerPrx | None:
        return uncheckedCast(ArrowServerPrx, proxy, facet)

    @staticmethod
    def ice_staticId() -> str:
        return "::arrowservice::ArrowServer"

IcePy.defineProxy("::arrowservice::ArrowServer", ArrowServerPrx)

class ArrowServer(Object, ABC):
    """
    Notes
    -----
        The Slice compiler generated this skeleton class from Slice interface ``::arrowservice::ArrowServer``.
    """

    _ice_ids: Sequence[str] = ("::Ice::Object", "::arrowservice::ArrowServer", )
    _op_uploadTable: IcePy.Operation
    _op_getTable: IcePy.Operation
    _op_getTableSchema: IcePy.Operation
    _op_listTableNames: IcePy.Operation

    @staticmethod
    def ice_staticId() -> str:
        return "::arrowservice::ArrowServer"

    @abstractmethod
    def uploadTable(self, tableName: str, table: ArrowTable, current: Current) -> None | Awaitable[None]:
        pass

    @abstractmethod
    def getTable(self, tableName: str, current: Current) -> ArrowTable | Awaitable[ArrowTable]:
        pass

    @abstractmethod
    def getTableSchema(self, tableName: str, current: Current) -> str | Awaitable[str]:
        pass

    @abstractmethod
    def listTableNames(self, current: Current) -> Sequence[str] | Awaitable[Sequence[str]]:
        pass

ArrowServer._op_uploadTable = IcePy.Operation(
    "uploadTable",
    "uploadTable",
    OperationMode.Normal,
    None,
    (),
    (((), IcePy._t_string, False, 0), ((), _arrowservice_ArrowTable_t, False, 0)),
    (),
    None,
    (),
    False)

ArrowServer._op_getTable = IcePy.Operation(
    "getTable",
    "getTable",
    OperationMode.Normal,
    None,
    (),
    (((), IcePy._t_string, False, 0),),
    (),
    ((), _arrowservice_ArrowTable_t, False, 0),
    (_arrowservice_TableNotFound_t,),
    False)

ArrowServer._op_getTableSchema = IcePy.Operation(
    "getTableSchema",
    "getTableSchema",
    OperationMode.Normal,
    None,
    (),
    (((), IcePy._t_string, False, 0),),
    (),
    ((), IcePy._t_string, False, 0),
    (_arrowservice_TableNotFound_t,),
    False)

ArrowServer._op_listTableNames = IcePy.Operation(
    "listTableNames",
    "listTableNames",
    OperationMode.Normal,
    None,
    (),
    (),
    (),
    ((), _arrowservice_StringSeq_t, False, 0),
    (),
    False)

__all__ = ["ArrowServer", "ArrowServerPrx", "_arrowservice_ArrowServerPrx_t"]
