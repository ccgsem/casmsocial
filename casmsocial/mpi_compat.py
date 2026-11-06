"""MPI interface shared by CASMSOCIAL runtime components.

Use the same MPI backend selected by Repast4Py, ensuring that
Python MPI objects match Repast4Py's compiled extensions.
"""

from repast4py import MPI

# Repast4Py's single-rank Windows build does not expose COMM_SELF.
# For a one-rank simulation, COMM_WORLD provides the same local
# communicator behavior needed by CASMSOCIAL's MVP tests.
if not hasattr(MPI, "COMM_SELF"):
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError(
            "COMM_SELF is unavailable in a multi-rank MPI environment."
        )
    MPI.COMM_SELF = MPI.COMM_WORLD
