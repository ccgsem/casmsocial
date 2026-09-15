"""Compatibility launch module for the internal CASMSocial gRPC runner."""

from casmsocial.casmsim.grpc_runner import *  # noqa: F403
from casmsocial.casmsim.grpc_runner import main as main

if __name__ == "__main__":
    raise SystemExit(main())
