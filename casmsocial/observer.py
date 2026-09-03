"""Observer base class for CasmSocial models"""

import pyarrow as pa

from casmsocial.model import Model


class Observer:
    """Base class for all observers"""

    # Step observers execute in ascending priority order. The default retains
    # registration order, while transport adapters can run after model-owned
    # loggers and publish the snapshot produced for the current tick.
    step_priority = 0

    def __init__(self, name, model: Model = None):
        self.name = name

    def initialize(self, model: Model) -> None:
        """Initialize the observer with the model."""
        pass

    def on_step(self, model: Model) -> None:
        """Called at each step of the simulation."""
        pass

    def on_end(self, model: Model) -> None:
        """Called at the end of the simulation."""
        pass

    def get_output_tables(self, model: Model) -> dict[str, pa.Table]:
        """Return this observer's current outputs as named Arrow tables.

        Keys should match the channel names used to name the observer's
        on-disk output (e.g. ``"agent_log"``), so callers such as casmservice
        can consume in-memory results the same way they consume the parquet
        outputs. The default implementation returns no tables; observers
        that produce data override this to expose their most recently
        built table(s) without requiring a read back from disk.
        """
        return {}
