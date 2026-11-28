"""Supervised runner under real MPI ranks, driven by the parent test."""

import sys
import time
from pathlib import Path
from threading import Event

import pyarrow as pa
from casmsim.run_state import RunState
from mpi4py import MPI

from casmsocial import grpc_runner


class CollectiveAdapter:
    """Every rank joins a collective; ``mode=cancel`` waits for cancellation."""

    def __init__(self, comm, params):
        self.comm, self.params = comm, params
        self.observer, self.cancelled = None, Event()
        self.state = RunState.Pending

    def add_observer(self, observer):
        self.observer = observer

    def cancel(self):
        self.cancelled.set()

    def get_state(self):
        return self.state, 1.0, 1

    def start(self):
        self.state = RunState.Running
        rows = self.comm.allgather({"rank": self.comm.Get_rank()})
        if self.params.get("mode") == "cancel" and self.comm.Get_rank() == 0:
            deadline = time.monotonic() + 15
            while not self.cancelled.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        self.comm.Barrier()
        if self.observer is not None:
            self.observer.publish("ranks", pa.Table.from_pylist(rows))
            self.observer.flush()
        self.state = RunState.Completed


if __name__ == "__main__":
    grpc_runner.get_comm = lambda: MPI.COMM_WORLD
    sys.argv = ["casmsocial-runner", "--run-dir", str(Path(sys.argv[1]) / "run"), "--supervised",
                "--drain-seconds", "1"]
    raise SystemExit(grpc_runner.main())
