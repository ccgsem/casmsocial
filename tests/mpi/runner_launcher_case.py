"""Real CLI, MPI, gRPC and Flight smoke fixture driven by the parent test."""

import json
import sys
import time
from pathlib import Path
from threading import Event, Thread, current_thread, main_thread

import pyarrow as pa
from casmsim.run_state import RunState
from mpi4py import MPI

from casmsocial import grpc_runner

RECORD = {}
FIXTURE_DIR = None


def require_main_thread():
    assert current_thread() is main_thread(), "MPI accessed from a background thread"


def wait_for_file(path):
    deadline = time.monotonic() + 15
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"fixture timed out waiting for {path.name}")
        Event().wait(0.01)


class CommProbe:
    """Record final-barrier entry and reject every off-main-thread MPI call."""

    def __getattr__(self, name):
        method = getattr(MPI.COMM_WORLD, name)

        def call(*args, **kwargs):
            require_main_thread()
            if name == "Barrier" and RECORD.get("adapter_returned"):
                RECORD["completion_barriers"] += 1
                if RECORD["rank"] == 0:
                    (FIXTURE_DIR / "rank0_completion_barrier").write_text("entered")
            return method(*args, **kwargs)

        return call


class CollectiveAdapter:
    """Minimal adapter with collective construction, execution and flushing."""

    def __init__(self, comm, params):
        require_main_thread()
        self.comm = comm
        self.params = params
        self.observer = None
        self.cancelled = Event()
        self.state = RunState.Pending
        RECORD["params"] = dict(params)
        self.comm.Barrier()
        RECORD["constructed"] = True

    def add_observer(self, observer):
        self.observer = observer
        RECORD["observer_attached"] = True

    def cancel(self):
        RECORD["cancel_calls"] += 1
        self.cancelled.set()

    def get_state(self):
        return self.state, 1.0, 1

    def start(self):
        require_main_thread()
        self.state = RunState.Running
        RECORD["started"] = True
        rows = self.comm.allgather({"rank": RECORD["rank"], "run_id": self.params["simulation.run_id"]})
        if self.observer is not None:
            self.observer.publish("ranks", pa.Table.from_pylist(rows))
        if self.params["mode"] == "cancel":
            if RECORD["rank"] == 0:
                (FIXTURE_DIR / "model_running").write_text("ready")
                wait_for_file(FIXTURE_DIR / "release_cancel")
            RECORD["collective_cancelled"] = bool(self.comm.allreduce(int(self.cancelled.is_set()), op=MPI.SUM))
        if self.params["mode"] == "worker_flush" and RECORD["rank"] == 1:
            (FIXTURE_DIR / "worker_flush_waiting").write_text("ready")
            wait_for_file(FIXTURE_DIR / "release_worker_flush")
        if self.observer is not None:
            self.observer.flush()
        self.state = RunState.Failed if RECORD["collective_cancelled"] else RunState.Completed
        RECORD["adapter_returned"] = True


def main(fixture_dir):
    global FIXTURE_DIR
    FIXTURE_DIR = Path(fixture_dir)
    run_dir = FIXTURE_DIR / "run"
    require_main_thread()
    RECORD.update(
        rank=MPI.COMM_WORLD.Get_rank(),
        thread_level=MPI.Query_thread(),
        constructed=False,
        started=False,
        observer_attached=False,
        cancel_calls=0,
        collective_cancelled=False,
        adapter_returned=False,
        completion_barriers=0,
        control_starts=0,
        flight_starts=0,
        endpoint_writes=0,
    )
    assert RECORD["thread_level"] >= MPI.THREAD_FUNNELED
    comm = CommProbe()
    grpc_runner.get_comm = lambda: comm
    original_write = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.parent == run_dir and path.name in {"runner_endpoints.json", "arrow_endpoint.txt"}:
            RECORD["endpoint_writes"] += 1
        return original_write(path, *args, **kwargs)

    Path.write_text = write_text
    original_control = grpc_runner.start_control_server
    original_flight = grpc_runner.start_broker_flight_server
    original_runner = grpc_runner.start_runner
    runtime = {}
    server_ready, stop_watcher = Event(), Event()

    def start_control(*args, **kwargs):
        require_main_thread()
        assert RECORD["rank"] == 0
        RECORD["control_starts"] += 1
        return original_control(*args, **kwargs)

    def start_flight(*args, **kwargs):
        require_main_thread()
        assert RECORD["rank"] == 0
        RECORD["flight_starts"] += 1
        return original_flight(*args, **kwargs)

    def start_runner(*args, **kwargs):
        control, flights = original_runner(*args, **kwargs)
        runtime.update(control=control, handoff=kwargs["handoff"])
        server_ready.set()
        (FIXTURE_DIR / "transport_ready").write_text("ready")
        return control, flights

    def supervise():
        if not server_ready.wait(15):
            return
        deadline = time.monotonic() + 25
        while not (FIXTURE_DIR / "stop").exists() and not stop_watcher.is_set():
            if time.monotonic() >= deadline:
                runtime["handoff"].shutdown(TimeoutError("fixture supervisor timed out"))
                break
            stop_watcher.wait(0.01)
        runtime["control"].stop(0).wait()

    grpc_runner.start_control_server = start_control
    grpc_runner.start_broker_flight_server = start_flight
    grpc_runner.start_runner = start_runner
    watcher = Thread(target=supervise, daemon=True) if RECORD["rank"] == 0 else None
    if watcher is not None:
        watcher.start()
    sys.argv = ["casmsocial-runner", "--run-dir", str(run_dir)]
    try:
        RECORD["exit_code"] = grpc_runner.main()
    finally:
        stop_watcher.set()
        if watcher is not None:
            watcher.join(timeout=2)
        Path.write_text = original_write
    records = MPI.COMM_WORLD.gather(RECORD, root=0)
    if RECORD["rank"] == 0:
        (FIXTURE_DIR / "result.json").write_text(json.dumps(records))


if __name__ == "__main__":
    main(sys.argv[1])
