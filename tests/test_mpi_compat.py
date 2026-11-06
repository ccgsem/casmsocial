"""Tests for CASMSOCIAL's MPI compatibility interface."""

from casmsocial.mpi_compat import MPI
from repast4py import MPI as repast_mpi


def test_mpi_backend_matches_repast4py():
    assert MPI is repast_mpi


def test_communicators_exist():
    assert hasattr(MPI, "COMM_WORLD")
    assert hasattr(MPI, "COMM_SELF")
    assert hasattr(MPI, "Comm")
    assert hasattr(MPI, "Intracomm")


def test_comm_self_is_single_rank():
    assert MPI.COMM_SELF.Get_size() == 1
    assert MPI.COMM_SELF.Get_rank() == 0


def test_single_rank_collectives():
    comm = MPI.COMM_SELF

    assert comm.allreduce(5, op=MPI.SUM) == 5
    assert comm.allgather([1, 2]) == [[1, 2]]
    assert comm.alltoall([["test"]]) == [["test"]]
