"""Loopback runner hosting CASMSocial gRPC control and Arrow Flight.

Single-rank usage (casmservice subprocess launch)::

    python -m casmsocial.grpc_runner --run-dir /tmp/runs/abc123

Multi-rank usage (mpirun / SLURM)::

    mpirun -n 8 python -m casmsocial.grpc_runner --run-dir /tmp/runs/abc123

All ranks execute this entry point.  Rank 0 starts the gRPC control and Arrow
Flight servers, writes ``runner_endpoints.json``, then waits for a ``Start``
RPC.  Ranks 1..N-1 wait on an MPI broadcast from rank 0.  Once the run
parameters arrive, all ranks create and start the model collectively.

MPI thread safety
-----------------
``mpi4py`` initialises MPI with ``MPI_THREAD_SERIALIZED`` by default.  To keep
all MPI collective calls (``bcast``) on the process's main thread, the gRPC
server's ``Start`` handler only stashes the request and fires a
``threading.Event``; rank 0's main thread wakes up, issues the ``bcast``, then
drives the model.  No MPI calls are made from gRPC thread-pool threads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from mpi4py import MPI

from casmsocial.__main__ import load_builtin_models
from casmsocial.factory import Models, load_models
from casmsocial.flight_broker import start_broker_flight_server
from casmsocial.grpc_control import ENDPOINT_FILENAME, SimulatorControlServicer, start_control_server
from casmsocial.observation_broker import ObservationBroker
from casmsocial.repast_observation_broker import RepastObservationBrokerAdapter

# MPI broadcast message tags
_MSG_START = "start"
_MSG_ABORT = "abort"


# ---------------------------------------------------------------------------
# Model runner (collective — called on all ranks)
# ---------------------------------------------------------------------------


def _run_model(
    run_id: str,
    params: dict,
    broker: ObservationBroker,
    comm: MPI.Intracomm,
    *,
    on_model_ready=None,
) -> None:
    """Create and run the model on all MPI ranks collectively.

    ``broker`` and ``on_model_ready`` are only meaningful on rank 0 (the
    broker is None on worker ranks).  All ranks must call this function
    together because ``model.start()`` issues MPI collectives internally.
    """
    load_builtin_models()
    plugins = params.get("model.plugins", [])
    if plugins:
        load_models(plugins)
    model = Models.create_model(params["model.name"])(comm, params)
    if broker is not None:
        model.add_observer(RepastObservationBrokerAdapter(broker))
    if on_model_ready is not None:
        on_model_ready(model)
    model.start()


# ---------------------------------------------------------------------------
# Rank 0 — servers + run lifecycle
# ---------------------------------------------------------------------------


def _rank0_main(
    comm: MPI.Intracomm,
    servicer: SimulatorControlServicer,
    broker: ObservationBroker,
) -> None:
    """Drive the run lifecycle from rank 0's main thread.

    Blocks until a ``Start`` RPC arrives, broadcasts params to all ranks,
    runs the model collectively, then returns so the caller can shut down
    the servers cleanly.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert rank == 0

    logger.info("Rank 0 waiting for Start RPC ({}  rank{})…", size, "s" if size > 1 else "")
    run_id, config_json = servicer.wait_for_run()

    params: dict = json.loads(config_json)
    params["simulation.run_id"] = run_id
    # Broker-backed Flight is the live transport; disable the per-model server.
    params["observers.arrow_server.enabled"] = False

    if size > 1:
        logger.info("Broadcasting run parameters to {} worker ranks.", size - 1)
        comm.bcast((_MSG_START, params), root=0)

    success = False
    try:
        _run_model(run_id, params, broker, comm, on_model_ready=servicer._set_model)
        success = True
    except Exception:
        logger.exception("Model run failed on rank 0.")
        if size > 1:
            # Workers are already running (bcast already sent); they will
            # observe the MPI_COMM_WORLD error or fall through at_end naturally.
            pass
    finally:
        servicer.complete_run(success=success)


# ---------------------------------------------------------------------------
# Ranks 1..N-1 — worker loop
# ---------------------------------------------------------------------------


def _worker_main(comm: MPI.Intracomm) -> None:
    """Wait for run parameters from rank 0, then participate in the collective run."""
    rank = comm.Get_rank()
    assert rank > 0

    logger.info("Worker rank {} waiting for broadcast from rank 0.", rank)
    msg = comm.bcast(None, root=0)
    kind = msg[0]

    if kind == _MSG_ABORT:
        logger.warning("Worker rank {} received abort — rank 0 failed before starting the run.", rank)
        return

    if kind != _MSG_START:
        logger.error("Worker rank {} received unexpected message kind {!r}.", rank, kind)
        return

    params: dict = msg[1]
    logger.info("Worker rank {} received run parameters; joining collective model start.", rank)
    try:
        # broker=None on workers: observations are only published from rank 0.
        _run_model(params["simulation.run_id"], params, broker=None, comm=comm)
    except Exception:
        logger.exception("Model run failed on worker rank {}.", rank)


# ---------------------------------------------------------------------------
# Server startup (rank 0 only)
# ---------------------------------------------------------------------------


def start_runner(run_dir: Path) -> tuple[object, object, SimulatorControlServicer]:
    """Start loopback-only gRPC and Flight endpoints on rank 0.

    Returns ``(grpc_server, flight_server, servicer)``.
    """
    broker = ObservationBroker()
    control, servicer = start_control_server(run_dir, broker)
    flights = start_broker_flight_server(run_dir, broker)
    manifest_path = run_dir / ENDPOINT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["flight"] = {"address": f"127.0.0.1:{flights.port}", "protocol": "arrow.flight"}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    return control, flights, servicer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        control, flights, servicer = start_runner(args.run_dir)
        broker = servicer._broker
        try:
            _rank0_main(comm, servicer, broker)
        except KeyboardInterrupt:
            pass
        finally:
            control.stop(0).wait()
            flights.shutdown()
    else:
        _worker_main(comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
