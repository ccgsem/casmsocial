"""MPI interface shared by CASMSOCIAL runtime components.

Use the MPI backend that Repast4Py was built with, so CASMSOCIAL's MPI objects
match Repast4Py's compiled extensions. Repast4Py 1.3+ exposes it as
``repast4py.MPI``: the real :mod:`mpi4py.MPI`, or a pure-Python single-rank
substitute in its no-MPI Windows build. Earlier releases (including the 1.2.1
in uv.lock) have no ``repast4py.MPI`` and always use :mod:`mpi4py`.
"""

try:
    from repast4py import MPI
except ImportError:
    from mpi4py import MPI

# Repast4Py's single-rank build does not expose COMM_SELF. For a one-rank
# simulation, COMM_WORLD provides the same local communicator behavior.
if not hasattr(MPI, "COMM_SELF"):
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("COMM_SELF is unavailable in a multi-rank MPI environment.")
    MPI.COMM_SELF = MPI.COMM_WORLD
