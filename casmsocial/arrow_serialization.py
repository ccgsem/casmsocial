"""Arrow IPC serialization helpers for the optional live Arrow/Ice server.

Only depends on pyarrow (already a hard dependency), so this module is safe
to import eagerly -- unlike casmsocial.arrow_server, which touches Ice and
must stay behind a lazy import gated by the optional ``service`` extra.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.ipc as ipc


class ArrowServerUnavailableError(Exception):
    """Raised when the `service` extra (zeroc-ice) is not installed.

    Defined here, rather than in arrow_server.py, so callers can catch it
    around the lazy `import casmsocial.arrow_server` itself -- that import
    is what raises it when zeroc-ice is missing, so the exception type must
    already be safely importable before that import is attempted.
    """


def serialize_table(table: pa.Table) -> bytes:
    """Serialize an Arrow table as IPC file bytes."""
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()
